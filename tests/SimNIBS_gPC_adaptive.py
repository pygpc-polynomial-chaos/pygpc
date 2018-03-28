import sys

import numpy as np

# sys.path.append('/data/pt_01756/software/git/pygpc')
import pygpc

# sys.path.append('/data/pt_01756/software/git/pyfempp')
import pyfempp
import os
import h5py
import multiprocessing
import multiprocessing.pool  # yes, we need both
from _functools import partial
import parmap
from pyfempp.coil import calc_coil_position_pdf
from pyfempp import load_hdf5
import glob
import itertools
from scipy.spatial import Delaunay
import random
import yaml
import pickle
from typing import List, Dict


# from https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def main():
    """
    Define here which parts of the gpc process you want to run.
    
    1. Get E_gPC parameters from Experiment: get_gpc_params()
        As we want the E_gPC only for realistic coil positions, we read all the positions that were taken in 
        the experiment, and transform them into gpc parameter definitions, condition-wise
        As parameters of gpc are uncorrelated, and real data is not, this leads to the problem, that the coil
        may be positioned into the subject's head for some combinations. With an iterative procedure, these 
        combinations are eliminated, the parameter ranges are reduced.
        An run_reg_adaptive_gpc is started for each condition, to save the gpc objects as .yaml.
    
    2. Start E_gPC for each condition: e_gpc()
        Reads reg.yamls and starts run_reg_adaptive_E_gPC with it. As the eGPC may take ages, I implemented a 
        clustersolution (dispy), see below. 
        
        The run_reg_adaptive_E_gPC uses calc_e() to compute the e fields. calc_e() checks if the e already was 
        computed. To do so, the randomness of the gpc_grid has to be defined by seed values.
        Results of the E_gPC are saved to disk.
        
    3. Create MEP fit for each condition: mep_fit()
        Fit a function the MEP IO curves per condition. Save as pickle to disk
        
    4. Start MIC_gPC: mic_gpc()
        The MIC_gPC uses the E_gPC to compute the e fields. GM, WM, CSF and Tensor Scaling are also uncertain parameters
        for the MIC_gPC, all the coil position parameters from the E_gPC are fixed. MEP Fit parameters are added as 
        uncertain parameters. This is done over all conditions.
        
        The E_gPC objects a read to get the e functions. 
        The MEP fit object are read as well, for MEP computation.
        run_reg_adaptive() is started.
            calc_mic() for each random grid row
                uses mic_workhorse() in a pool to get mics for each triangle of midlayer
        
        While the E_gPC is build from all triangles, we want the MIC to be computed in the midlayer from 
        a Freesurfer mask. So we need E in the tetrahedra between WM and GM surface. To calc the midlayer E from that,
        dadt and potential has to be computed there as well.
        
        The MIC_gPC object is saved to disk.
        
    """
    # results_folder = "/data/pt_01756/results/gpc/gpc_4_params"
    results_folder = "/data/pt_01756/results/gpc/"
    data_folder = "/data/pt_01756/pyfempp_testdata/gpc/"
    four_params_gpc = False
    # get_gpc_params(data_folder)
    #
    # e_gpc_4_params(results_folder, data_folder, mp = '', dispy_scheduler_host="rihanna")
    # e_gpc(results_folder, data_folder, mp='dispy', dispy_scheduler_host="rihanna")  # no multiprocessing: mp='', pool mp: mp='mp'
    # fit_meps(results_folder, data_folder)
    tau_gpc(results_folder, data_folder, four_params_gpc)


def get_gpc_params(data_folder):
    """Read experimental coil positions and create gPC parameters from this.
    Then, reduce parameter space, so that the coil is not positioned inside the subjects head.
    Then, start run_reg_adaptive_egpc to save reg object.
        ***Important*** To save the reg objects, these must not exist on disk.
        The GPC ends after saving the reg objects."""

    # data_folder = "/data/pt_01756/pyfempp_testdata/gpc/"
    # data_folder = "d:\\work-konstantin\\work\\py\\pyfempp_testdata\\"

    mesh_fn = data_folder + '15484.08_fixed.msh'
    tensor_fn = data_folder + 'CTI_vn_tensor.nii'
    coil_fn = data_folder + 'MagVenture_MC_B60_1233.ccd'
    hdf5_fn = data_folder + "15484.08-0001_MagVenture_MC_B60_1233.hdf5"

    pdf_paras_location, pdf_paras_orientation_euler, \
    positions_mean, locations_zeromean, svd_v, \
    locations_transform, euler_angles = calc_coil_position_pdf(os.path.join(data_folder, "results_conditions.csv"),
                                                                          os.path.join(data_folder, "simPos.csv"))
    locations_transform = np.concatenate(locations_transform)
    euler_angles = np.concatenate(euler_angles)
    # suppress warning for multiple pools like (parmap.starmap(parmap.starmap(func,a),zip(a,b))
    # warnings.filterwarnings("ignore", message="daemonic processes are not allowed to have children")

    # pdf_paras_location[0] == 1st condition
    # 1st condition == X,Y,Z
    #     X = beta_paras, moments, p_value, uni_paras
    #         beta_paras = p_beta, q_beta, a_beta, b_beta
    #         moments = np.array([data_mean, data_std, beta_mean, beta_std])
    #         p_value = p_value

    # define position and orientation uncertainty parameters for every condition
    n_conditions = 6
    dim = 10  # number of random variables
    dim_fixed = dim - 6  #
    # random vars: WM, GM, CSF, tensor_scaling, POS_X, POS_Y, POS_Z, yaw, pitch, roll

    pdftype = np.repeat('beta', dim)

    # create bounds for wm, gm, csf, tensor scaling as they stay the same over all conditions
    a = [0.1, 0.1, 1.2, 0.4]  # lower bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    b = [0.4, 0.6, 1.8, 0.6]  # upper bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    p = [3, 3, 3, 3]  # first shape parameter of pdf (WM, GM, CSF) and tensor scaling
    q = [3, 3, 3, 3]  # second shape parameter of pdf (WM, GM, CSF) and tensor scaling

    # create delauney object
    with h5py.File(hdf5_fn, 'r') as f:
        points = np.array(f['mesh/nodes/node_coord'])
        node_number_list = np.array(f['mesh/elm/node_number_list'])
        elm_type = np.array(f['mesh/elm/elm_type'])
        regions = np.array(f['mesh/elm/tag1'])
        triangles_regions = regions[elm_type == 2,] - 1000
        triangles = node_number_list[elm_type == 2, 0:3]
    triangles = triangles[triangles_regions == 5]
    surface_points = pyfempp.unique_rows(np.reshape(points[triangles], (3 * triangles.shape[0], 3)))
    limits_scaling_factor = .1

    # Generate Delaunay triangulation object
    dobj = Delaunay(surface_points)
    del points, node_number_list, elm_type, regions, triangles_regions, triangles, surface_points

    # we need to map some indices
    dic_par_name = dict.fromkeys([0, 1, 2], "pdf_paras_location")
    dic_par_name.update(dict.fromkeys([3, 4, 5], "pdf_paras_orientation_euler"))
    dic_limit_id = {0: 2, 2: 3}  # translates from get_bad_param limits id to pdf_paras_* id
    dic_param_id = dict.fromkeys([0, 3], 0)
    dic_param_id.update(dict.fromkeys([1, 4], 1))
    dic_param_id.update(dict.fromkeys([2, 5], 2))

    for cond in range(n_conditions):
        random.seed(1)
        # check if any dipoles for start param values
        print("Check coil dipoles for condition " + str(cond))
        bad_params = get_bad_param(pdf_paras_location, pdf_paras_orientation_euler,
                                   pos_mean=positions_mean,
                                   v=svd_v,
                                   del_obj=dobj,
                                   coil_fn=coil_fn,
                                   condition=cond)

        # change param limits until no dipoles left inside head
        while bad_params:
            # several parameters may lead to a bad positions.
            # take a random parameter from these
            param_id = random.sample(range(len(bad_params)), 1)[0]

            # change last bad_params[-1]' limit
            param_to_change_idx = bad_params[param_id][0]
            param_to_change = dic_par_name[param_to_change_idx]
            limit_to_change = dic_limit_id[bad_params[param_id][1]]
            param_to_change_idx = dic_param_id[param_to_change_idx]

            # grab limits for correct parameter
            limits = np.array([locals()[param_to_change][cond][param_to_change_idx][0][2],
                               None,
                               locals()[param_to_change][cond][param_to_change_idx][0][3]])

            # rescale limit (only important boundary)
            factor = (limits[2] - limits[0]) * limits_scaling_factor
            limits = [limits[0] + factor, None, limits[2] - factor]
            print "Changing " + param_to_change + " [" + str(param_to_change_idx) + "][" + \
                  str(limit_to_change) + "]: " + \
                  str(locals()[param_to_change][cond][param_to_change_idx][0][limit_to_change]) + " -> " + \
                  str(limits[bad_params[param_id][1]])
            locals()[param_to_change][cond][param_to_change_idx][0][limit_to_change] = limits[bad_params[param_id][1]]

            # repeat dipole check
            bad_params = get_bad_param(pdf_paras_location, pdf_paras_orientation_euler,
                                       pos_mean=positions_mean,
                                       v=svd_v,
                                       del_obj=dobj,
                                       coil_fn=coil_fn,
                                       condition=cond)

    # build params vars for all conditions
    a_all = np.zeros((n_conditions, dim))
    a_all = a_all.tolist()  # type: list
    b_all = np.zeros((n_conditions, dim))
    b_all = b_all.tolist()  # type: list
    p_all = np.zeros((n_conditions, dim))
    p_all = p_all.tolist()  # type: list
    q_all = np.zeros((n_conditions, dim))
    q_all = q_all.tolist()  # type: list
    pdf_shape_all = []  # np.zeros((n_conditions, 1 ))
    limits_all = []  # np.zeros((n_conditions, 2 * dim))
    for i in range(n_conditions):
        a_all[i][0:dim_fixed] = a
        b_all[i][0:dim_fixed] = b
        p_all[i][0:dim_fixed] = p
        q_all[i][0:dim_fixed] = q

        for j in range(3):
            # get location
            a_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][2])
            b_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][3])
            p_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][0])
            q_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][1])

            # get orientation
            a_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][2])
            b_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][3])
            p_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][0])
            q_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][1])

        pdf_shape_all.append([list(p_all[i]), list(q_all[i])])
        limits_all.append([list(a_all[i]), list(b_all[i])])

    mic_grid_n = 10  # should be the same size as MIC_grid_N_exp, shouldn't it?

    coil_positions_exp, conditions, _, meps, intensities = pyfempp.read_exp_stimulations(
        os.path.join(data_folder, "results_conditions.csv"),
        os.path.join(data_folder, "simPos.csv"))
    conditions = np.array(conditions)
    # print '{0:0.20f}'.format(pdf_shape_all[4][0][7])
    # print ''
    # get unique conditions in correct order
    _, idx = np.unique(conditions, return_index=True)
    conditions_unique = conditions[idx]
    mic_grid_n_exp = len(coil_positions_exp)

    # mesh_fn = '/data/pt_01756/probands/15484.08/simnbis/15484.08_fixed.msh'
    # tensor_fn = '/data/pt_01756/probands/15484.08/simnbis/d2c_15484.08_PA/CTI_vn_tensor.nii'
    # coil_fn = '/data/pt_01756/coils/ccd/MagVenture_MC_B60_1233.nii'

    results_folder = os.path.join(data_folder, '/results/E_gPC/')  # root folder of all E simulations
    results_folder = "/data/pt_01756/results/gpc"
    # run adaptive gpc (regression) passing the goal function func(x, args()) of E
    ########################################################################################

    # this multiprocessed version starts one gpc per condition
    # Maybe E_gpc or E is to big to be pickled.
    results_folder_cond = []
    regobj_yaml_fn = []
    for i in range(n_conditions):
        results_folder_cond.append(os.path.join(results_folder, 'cond' + str(i)))
        regobj_yaml_fn.append(results_folder_cond[-1] + '_regInfo_10_params_checked.yaml')
    seed = 3  # sets seed in grid_init to have the same e calculated in each programm start.

    # start gpc in parallel
    # parmap.starmap(func, zip([1,3,5], [2,4,6]), 'a') calls:
    #     func(1, 2, 'a')
    #     func(3, 4, 'a')
    #     func(5, 6, 'a')

    eps = 1E-3

    # e_gpc: n_condition gPC objects
    # e: calculated e-fields for each condition

    parmap.starmap(pygpc.run_reg_adaptive_E_gPC,
                   zip(np.tile(pdftype, (n_conditions, 1)),
                       pdf_shape_all,
                       limits_all,
                       np.tile(calc_e, n_conditions),
                       zip(np.repeat(mesh_fn, n_conditions),
                           np.repeat(tensor_fn, n_conditions),
                           results_folder_cond,
                           np.repeat(coil_fn, n_conditions),
                           # range(n_conditions),
                           positions_mean,
                           svd_v),
                       regobj_yaml_fn),
                   order_start=0,
                   order_end=10,
                   eps=eps,
                   print_out=True,
                   seed=seed,
                   do_mp=False,
                   n_cpu=5,
                   dispy=False,
                   dispy_sched_host='cher',
                   random_vars='',
                   pm_pool=NonDaemonicPool(6))


def e_gpc(results_folder, data_folder, mp='dispy', dispy_scheduler_host="ramones"):
    """Runs e_gpc

    3 ways of computing e are implemented:
        - single thread
        - pool
        - dispy (cluster)
    Each condition (of 6) has its own gpc and is startet in a parmap.pool. Provide a NonDaemonicPool object 
        as pm_pool. Parmap version of 'map' allows to provide multiple arguments.
    New params for run_reg_adaptive: 
        - do_mp<boolean>: pooled version of calc_e (dominant over dispy)
        - n_cpu<int>: cpus per calc_e pool. e.g. n_cpu / n_conditions
        - dispy<boolean>: dispy version of calc_e
        - dispy_sched_host<string>: where runs dispy scheduler
        - seed<int>: if set, same gpc coordinates are picked, so already computed e can be reused
        - hdf5_geo_fn<string>: .hdf5 with mesh information in it
    returns filename of generated .hdf5

    """
    # get multiprocessing type:
    if mp == "dispy":
        mp = False
        dispy = True
    elif mp == "mp":
        mp = True
        dispy = False
    else:
        mp = False
        dispy = False

    # start ~/.local/bin/dispyscheduler on dispy_scheduler_host

    # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    # print(user_paths)
    # Define parameters
    ########################################################################################
    # get shape of pdf's from preprocessing for E_gPC
    # data_folder = "d:\\work-konstantin\\work\\py\\pyfempp_testdata\\"

    mesh_fn = os.path.join(data_folder, '15484.08_fixed.msh')
    tensor_fn = os.path.join(data_folder, 'CTI_vn_tensor.nii')
    coil_fn = os.path.join(data_folder, 'MagVenture_MC_B60_1233.ccd')
    hdf5_fn = os.path.join(data_folder, "15484.08-0001_MagVenture_MC_B60_1233.hdf5")
    pdf_paras_location, pdf_paras_orientation_euler, \
    positions_mean, locations_zeromean, svd_v, \
    locations_transform, euler_angles = calc_coil_position_pdf(data_folder)
    locations_transform = np.concatenate(locations_transform)
    euler_angles = np.concatenate(euler_angles)
    # pdf_paras_location[0] == 1st condition
    # 1st condition == X,Y,Z
    #     X = beta_paras, moments, p_value, uni_paras
    #         beta_paras = p_beta, q_beta, a_beta, b_beta
    #         moments = np.array([data_mean, data_std, beta_mean, beta_std])
    #         p_value = p_value

    # define position and orientation uncertainty parameters for every condition
    n_conditions = 6
    dim = 10  # number of random variables
    dim_fixed = dim - 6  #
    # random vars: WM, GM, CSF, tensor_scaling, POS_X, POS_Y, POS_Z, yaw, pitch, roll

    pdftype = np.repeat('beta', dim)

    # create bounds for wm, gm, csf, tensor scaling as they stay the same over all conditions
    a = [0.1, 0.1, 1.2, 0.4]  # lower bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    b = [0.4, 0.6, 1.8, 0.6]  # upper bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    p = [3, 3, 3, 3]  # first shape parameter of pdf (WM, GM, CSF) and tensor scaling
    q = [3, 3, 3, 3]  # second shape parameter of pdf (WM, GM, CSF) and tensor scaling

    # build params vars for all conditions
    a_all = np.zeros((n_conditions, dim))
    a_all = a_all.tolist()  # type: list
    b_all = np.zeros((n_conditions, dim))
    b_all = b_all.tolist()  # type: list
    p_all = np.zeros((n_conditions, dim))
    p_all = p_all.tolist()  # type: list
    q_all = np.zeros((n_conditions, dim))
    q_all = q_all.tolist()  # type: list
    pdf_shape_all = []  # np.zeros((n_conditions, 1 ))
    limits_all = []  # np.zeros((n_conditions, 2 * dim))
    for i in range(n_conditions):
        a_all[i][0:dim_fixed] = a
        b_all[i][0:dim_fixed] = b
        p_all[i][0:dim_fixed] = p
        q_all[i][0:dim_fixed] = q

        for j in range(3):
            # get location
            a_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][2])
            b_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][3])
            p_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][0])
            q_all[i][dim_fixed + j] = float(pdf_paras_location[i][j][0][1])

            # get orientation
            a_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][2])
            b_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][3])
            p_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][0])
            q_all[i][dim_fixed + 3 + j] = float(pdf_paras_orientation_euler[i][j][0][1])

        pdf_shape_all.append([list(p_all[i]), list(q_all[i])])
        limits_all.append([list(a_all[i]), list(b_all[i])])

    coil_positions_exp, conditions, _, meps, intensities = pyfempp.read_exp_stimulations(
        os.path.join(data_folder, "results_conditions.csv"),
        os.path.join(data_folder, "simPos.csv"))
    conditions = np.array(conditions)
    # print '{0:0.20f}'.format(pdf_shape_all[4][0][7])
    # print ''
    # get unique conditions in correct order
    _, idx = np.unique(conditions, return_index=True)
    # conditions_unique = conditions[idx]
    # mic_grid_n_exp = len(coil_positions_exp)

    # mesh_fn = '/data/pt_01756/probands/15484.08/simnbis/15484.08_fixed.msh'
    # tensor_fn = '/data/pt_01756/probands/15484.08/simnbis/d2c_15484.08_PA/CTI_vn_tensor.nii'
    # coil_fn = '/data/pt_01756/coils/ccd/MagVenture_MC_B60_1233.nii'

    # results_folder = data_folder + '/results/E_gPC/'  # root folder of all E simulations
    # results_folder = "/data/pt_01756/results/gpc"
    # run adaptive gpc (regression) passing the goal function func(x, args()) of E
    ########################################################################################

    # this multiprocessed version starts one gpc per condition
    # Maybe E_gpc or E is to big to be pickled.
    results_folder_cond = []
    regobj_yaml_fn = []
    for i in range(n_conditions):
        results_folder_cond.append(os.path.join(results_folder, 'cond' + str(i)))
        regobj_yaml_fn.append(results_folder_cond[-1] + '_regInfo_10_params_checked.yaml')

    seed = 3  # sets seed in grid_init to have the same e calculated in each programm start.

    # start gpc in parallel
    # parmap.starmap(func, zip([1,3,5], [2,4,6]), 'a') calls:
    #     func(1, 2, 'a')
    #     func(3, 4, 'a')
    #     func(5, 6, 'a')
    # n_cpus = multiprocessing.cpu_count()
    # n_cpus = 37
    eps = 1E-3
    interaction_max = 2
    # e_gpc: n_condition gPC objects
    # e: calculated e-fields for each condition

    try:
        parmap.starmap(pygpc.run_reg_adaptive_E_gPC,
                       zip(np.tile(pdftype, (n_conditions, 1)),
                           pdf_shape_all,
                           limits_all,
                           np.tile(calc_e, n_conditions),
                           zip(np.repeat(mesh_fn, n_conditions),
                               np.repeat(tensor_fn, n_conditions),
                               results_folder_cond,
                               np.repeat(coil_fn, n_conditions),
                               positions_mean,
                               svd_v),
                           regobj_yaml_fn),
                       order_start=0,
                       order_end=10,
                       interaction_order_max=interaction_max,
                       eps=eps,
                       print_out=True,
                       seed=seed,
                       do_mp=mp,
                       n_cpu=5,
                       dispy=dispy,
                       dispy_sched_host=dispy_scheduler_host,
                       random_vars='',
                       hdf5_geo_fn=hdf5_fn,
                       pm_pool=NonDaemonicPool(6))

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        quit()

    print "E_gPC is done!"

    return


def e_gpc_4_params(results_folder, data_folder, mp='dispy', dispy_scheduler_host='rihanna'):
    """Runs e_gpc with conductivity and ts only. Mean coil positions are used.

    3 ways of computing e are implemented:
        - single thread
        - pool
        - dispy (cluster)
    Each condition (of 6) has its own gpc and is startet in a parmap.pool. Provide a NonDaemonicPool object 
        as pm_pool. Parmap version of 'map' allows to provide multiple arguments.
    New params for run_reg_adaptive: 
        - do_mp<boolean>: pooled version of calc_e (dominant over dispy)
        - n_cpu<int>: cpus per calc_e pool. e.g. n_cpu / n_conditions
        - dispy<boolean>: dispy version of calc_e
        - dispy_sched_host<string>: where runs dispy scheduler
        - seed<int>: if set, same gpc coordinates are picked, so already computed e can be reused
        - hdf5_geo_fn<string>: .hdf5 with mesh information in it
    returns filename of generated .hdf5

    """
    # get multiprocessing type:
    if mp == "dispy":
        mp = False
        dispy = True
    elif mp == "mp":
        mp = True
        dispy = False
    else:
        mp = False
        dispy = False

    # start ~/.local/bin/dispyscheduler on this host:


    # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    # print(user_paths)
    # Define parameters
    ########################################################################################
    # get shape of pdf's from preprocessing for E_gPC
    # data_folder = "d:\\work-konstantin\\work\\py\\pyfempp_testdata\\"

    mesh_fn = os.path.join(data_folder, '15484.08_fixed.msh')
    tensor_fn = os.path.join(data_folder, 'CTI_vn_tensor.nii')
    coil_fn = os.path.join(data_folder, 'MagVenture_MC_B60_1233.ccd')
    hdf5_fn = os.path.join(data_folder, "15484.08-0001_MagVenture_MC_B60_1233.hdf5")
    pdf_paras_location, pdf_paras_orientation_euler, \
    positions_mean, locations_zeromean, svd_v, \
    locations_transform, euler_angles = calc_coil_position_pdf(data_folder)

    locations_transform = np.concatenate(locations_transform)
    euler_angles = np.concatenate(euler_angles)

    # pdf_paras_location[0] == 1st condition
    # 1st condition == X,Y,Z
    #     X = beta_paras, moments, p_value, uni_paras
    #         beta_paras = p_beta, q_beta, a_beta, b_beta
    #         moments = np.array([data_mean, data_std, beta_mean, beta_std])
    #         p_value = p_value

    # define position and orientation uncertainty parameters for every condition
    n_conditions = 6
    dim = 4  # number of random variables
    # dim_fixed = 4  #
    # random vars: WM, GM, CSF, tensor_scaling, POS_X, POS_Y, POS_Z, yaw, pitch, roll

    pdftype = np.repeat('beta', dim)

    # create bounds for wm, gm, csf, tensor scaling as they stay the same over all conditions
    a = [0.1, 0.1, 1.2, 0.4]  # lower bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    b = [0.4, 0.6, 1.8, 0.6]  # upper bounds of conductivities in S/m (WM, GM, CSF) and tensor scaling
    p = [3, 3, 3, 3]  # first shape parameter of pdf (WM, GM, CSF) and tensor scaling
    q = [3, 3, 3, 3]  # second shape parameter of pdf (WM, GM, CSF) and tensor scaling
    pdf_shape = [p, q]
    limits = [a, b]
    pdf_shape_all = []
    limits_all = []
    for _ in range(n_conditions):
        pdf_shape_all.append(pdf_shape)
        limits_all.append(limits)

    coil_positions_exp, conditions, _, meps, intensities = pyfempp.read_exp_stimulations(
        os.path.join(data_folder, "results_conditions.csv"),
        os.path.join(data_folder, "simPos.csv"))
    conditions = np.array(conditions)
    # print '{0:0.20f}'.format(pdf_shape_all[4][0][7])
    # print ''
    # get unique conditions in correct order
    _, idx = np.unique(conditions, return_index=True)
    # conditions_unique = conditions[idx]
    # mic_grid_n_exp = len(coil_positions_exp)

    # mesh_fn = '/data/pt_01756/probands/15484.08/simnbis/15484.08_fixed.msh'
    # tensor_fn = '/data/pt_01756/probands/15484.08/simnbis/d2c_15484.08_PA/CTI_vn_tensor.nii'
    # coil_fn = '/data/pt_01756/coils/ccd/MagVenture_MC_B60_1233.nii'

    # results_folder = data_folder + '/results/E_gPC/'  # root folder of all E simulations
    # results_folder = "/data/pt_01756/results/gpc"
    # run adaptive gpc (regression) passing the goal function func(x, args()) of E
    ########################################################################################

    # this multiprocessed version starts one gpc per condition
    # Maybe E_gpc or E is to big to be pickled.
    results_folder_cond = []
    regobj_yaml_fn = []
    for i in range(n_conditions):
        results_folder_cond.append(os.path.join(results_folder, 'cond' + str(i)))
        regobj_yaml_fn.append(results_folder_cond[-1] + '_regInfo.yaml')

    seed = 3  # sets seed in grid_init to have the same e calculated in each programm start.

    # start gpc in parallel
    # parmap.starmap(func, zip([1,3,5], [2,4,6]), 'a') calls:
    #     func(1, 2, 'a')
    #     func(3, 4, 'a')
    #     func(5, 6, 'a')
    # n_cpus = multiprocessing.cpu_count()
    # n_cpus = 37
    eps = 1E-3
    interaction_max = 2
    # e_gpc: n_condition gPC objects
    # e: calculated e-fields for each condition

    try:
        parmap.starmap(pygpc.run_reg_adaptive_E_gPC,
                       zip(np.tile(pdftype, (n_conditions, 1)),
                           pdf_shape_all,
                           limits_all,
                           np.tile(calc_e, n_conditions),
                           zip(np.repeat(mesh_fn, n_conditions),
                               np.repeat(tensor_fn, n_conditions),
                               results_folder_cond,
                               np.repeat(coil_fn, n_conditions),
                               positions_mean,
                               svd_v),
                           regobj_yaml_fn),
                       order_start=0,
                       order_end=10,
                       interaction_order_max=interaction_max,
                       eps=eps,
                       print_out=True,
                       seed=seed,
                       do_mp=mp,
                       n_cpu=5,
                       dispy=dispy,
                       dispy_sched_host=dispy_scheduler_host,
                       random_vars='',
                       hdf5_geo_fn=hdf5_fn,
                       pm_pool=NonDaemonicPool(6))

    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        quit()

    print "E_gPC is done!"

    return


def tau_gpc(results_folder, data_folder, four_params_gpc):
    """Builds stuff for the the mic_gpc and calls run_run_adaptive() which then calls calc_mic
    
    
    Params
    -------------------------------
    results_folder: String, root folder of gpc results
    data_folder: String, where to fetch all needed files (fs, MEP, ...)
    four_params_gpc: True -> e_gpc_params[4:] = 0
    """
    n_conditions = 6
    n_samples = 100
    interaction_order_max = 3
    eps = 1e-3

    tet_idx_fn_nodes = os.path.join(data_folder, "15484.08_mask_nodes_3l.txt")
    tet_idx_fn_center = os.path.join(data_folder, "15484.08_mask_3l.txt")
    msh_fn = os.path.join(results_folder, '15484.08-0001_MagVenture_MC_B60_1233.hdf5')
    msh = load_hdf5(msh_fn)
    fsl_gm = os.path.join(data_folder, "lh.pial")
    fsl_wm = os.path.join(data_folder, "lh.white")
    mask_fn = os.path.join(data_folder, "mask_15484.08.mgh")
    surf_points_upper, surf_points, surf_points_lower, surf_con = (None,) * 4

    e_gpc_obj = []
    coeffs_e_norm = []
    coeffs_e_tan = []

    # load e_gpc objects per condition
    fn_coeffs_base = None
    for i in range(n_conditions):
        e_gpc_obj.append(pygpc.load_gpcobj(os.path.join(results_folder, 'cond' + str(i) + "obj.pkl")))
        phi = []
        dadt = []

        fn_coeffs_base = os.path.join(results_folder, 'cond' + str(i) + '_coeffs')
        if os.path.exists(fn_coeffs_base + '_norm.npy') and os.path.exists(fn_coeffs_base + '_tan.npy'):
            print "Condition " + str(i) + ": loading coeffs from file"

            coeffs_e_norm.append(np.load(fn_coeffs_base + '_norm.npy'))
            coeffs_e_tan.append(np.load(fn_coeffs_base + '_tan.npy'))

        else:
            print "Condition " + str(i) + ": loading phi and dAdt from hdf5"

            # read file list in results folder
            fnames_hdf5 = glob.glob(os.path.join(results_folder, 'cond' + str(i), '*.hdf5'))
            # fnames_hdf5 = glob.glob(os.path.join(results_folder, '*.hdf5'))

            # sort filenames by number
            fnames_hdf5.sort()
            # if there's an ln -s to another hdf5
            if '15484' in fnames_hdf5[-1]:
                fnames_hdf5 = fnames_hdf5[:-1]

            # create gm-wm surface only once
            if any(var is None for var in [surf_points_upper, surf_points, surf_points_lower, surf_con]):
                print "Getting midlayer points"
                surf_points_upper, surf_points, surf_points_lower, surf_con = pyfempp.make_GM_WM_surface(
                    gm_surf_fname=fsl_gm,
                    wm_surf_fname=fsl_wm,
                    mask_fn=mask_fn,
                    layer=3)

            # load phi and dAdt from e_gpc results
            for j in range(len(fnames_hdf5)):
                fancy_bar('Read phi and dAdt ', j, len(fnames_hdf5))

                # read phi and dAdt from hdf5 files
                with h5py.File(fnames_hdf5[j], 'r') as f:
                    phi.append(np.array(f['data/potential']).flatten()[:, np.newaxis])
                    dadt.append(np.reshape(np.array(f['data/dAdt']).flatten(), (phi[j].shape[0], 3), order='c'))

            # determine e_norm and e_tan for every simulation
            for j in range(len(fnames_hdf5)):
                fancy_bar('Calculating midlayer e ', j, len(fnames_hdf5))
                e_norm_temp, e_tan_temp = msh.calc_E_on_GM_WM_surface3(phi=phi[j],
                                                                       dAdt=dadt[j],
                                                                       surf_up=surf_points_upper,
                                                                       surf_mid=surf_points,
                                                                       surf_low=surf_points_lower,
                                                                       con=surf_con,
                                                                       fname_ele_idx_center=tet_idx_fn_center,
                                                                       fname_ele_idx_nodes=tet_idx_fn_nodes,
                                                                       verbose=False)

                if j == 0:
                    e_norm = np.zeros((len(fnames_hdf5), 3 * e_norm_temp.shape[0]))
                    e_tan = np.zeros((len(fnames_hdf5), 3 * e_norm_temp.shape[0]))

                e_norm[j, :] = np.reshape(e_norm_temp, (1, 3 * e_norm_temp.shape[0])).flatten()
                e_tan[j, :] = np.reshape(e_tan_temp, (1, 3 * e_tan_temp.shape[0])).flatten()

            del phi, dadt
            # expand for e_norm and e_tan
            print "Expanding gPC object " + str(i)
            coeffs_e_norm.append(e_gpc_obj[i].expand(e_norm))
            coeffs_e_tan.append(e_gpc_obj[i].expand(e_tan))
            np.save(fn_coeffs_base+'_norm', coeffs_e_norm[-1])
            np.save(fn_coeffs_base + '_tan', coeffs_e_tan[-1])
            del e_norm, e_tan

    del msh, surf_points_upper, surf_points, surf_points_lower, surf_con, fn_coeffs_base

    # now get fit iocurve, per condition
    mep_fits = []
    fname_mep_fit_cond = None
    for i in range(n_conditions):
        fname_mep_fit_cond = os.path.join(results_folder, 'mep_fit_cond' + str(i) + '.pkl')
        with open(fname_mep_fit_cond, 'r') as f:
            mep_fits.append(pickle.load(f))

    # get mic_gpc params from mep fit
    pdftype = ['norm' for i in range(len(mep_fits)) for _ in range(mep_fits[i].popt.shape[0])]
    random_vars = ['MEP #{} para #{}'.format(str(i), str(j)) for i in range(len(mep_fits)) for j in
                   range(mep_fits[i].popt.shape[0])]
    pdfshape = [[mep_fits[i].popt[j] for i in range(len(mep_fits)) for j in range(mep_fits[i].popt.shape[0])],
                [mep_fits[i].pstd[j] ** 2 for i in range(len(mep_fits)) for j in range(mep_fits[i].popt.shape[0])]]
    limits = [[0 for i in range(len(mep_fits)) for _ in range(mep_fits[i].popt.shape[0])],
              [0 for i in range(len(mep_fits)) for _ in range(mep_fits[i].popt.shape[0])]]

    # now get params of wm, gm. csf and ts
    e_gpc_params_yml = n_conditions * [0]  # type: List
    for i in range(n_conditions):
        fname_mep_fit_cond = os.path.join(results_folder, 'cond' + str(i) + "_regInfo_10_params_checked.yaml")
        with open(fname_mep_fit_cond, 'r') as f:
            e_gpc_params_yml[i] = yaml.load(f)  # type: Dict[str, str]Dict[str, float]
        del f

    # params used in e_gpc for wm, gm, csf, tensor scaling (same for all conditions):
    a = e_gpc_params_yml[0]['limits'][0][0:4]
    b = e_gpc_params_yml[0]['limits'][1][0:4]
    del e_gpc_params_yml, fname_mep_fit_cond

    # and concat parameters, so that e_gpc stuff is first
    random_vars = ['wm', 'gm', 'csf', 'ts'] + random_vars
    pdftype = ['beta', 'beta', 'beta', 'beta'] + pdftype
    pdfshape = [[3, 3, 3, 3] + pdfshape[0], [3, 3, 3, 3] + pdfshape[1]]
    limits = [a + limits[0], b + limits[1]]

    del a, b

    # set mean position parameter per condition
    fixed_par = []
    for i in range(n_conditions):
        fixed_par.append(np.zeros((1, 6)))

    # start mic_gpc
    gpc, e = pygpc.run_reg_adaptive2(random_vars,
                                     pdftype,
                                     pdfshape,
                                     limits,
                                     calc_tau,
                                     (fixed_par, coeffs_e_norm, coeffs_e_tan, e_gpc_obj, mep_fits,
                                      n_samples, four_params_gpc),
                                     order_start=0,
                                     order_end=10,
                                     eps=eps,
                                     print_out=True,
                                     interaction_order_max=interaction_order_max,
                                     save_res_fn=os.path.join(results_folder, 'tau_10_param.hdf5'),
                                     seed=1)

    pygpc.save_gpcobj(gpc, os.path.join(results_folder, 'mic_gpc.gpc'))
    np.save(os.path.join(results_folder, 'mic_gpc_e'), e)

    # this was the old mic_gpc with 40 parameters:
    """
    # make gPC grid containing random conductivities and pseudo random orientations where MEP amplitudes are measured
    #   in gpc parameter space
    mic_grid_conductivities = pygpc.randomgrid(pdftype=pdftype[0:dim_fixed],
                                               gridshape=[pdf_shape_all[0][0][0:dim_fixed],
                                                          pdf_shape_all[0][1][0:dim_fixed]],
                                               limits=[limits_all[0][0][0:dim_fixed],
                                                       limits_all[0][1][0:dim_fixed]],
                                               N=mic_grid_n_exp)  # n was mic_grid_n

    # append coil positions from experiment in a pseudo random manner
    sample_idx = np.random.choice(range(mic_grid_n_exp), size=mic_grid_n_exp, replace=False)
    # _sample_idx_ is mapping from coil positions to conductivities
    locations_transform = locations_transform[sample_idx,]
    euler_angles = euler_angles[sample_idx,]
    conditions = conditions[sample_idx]
    meps = meps[sample_idx]
    intensities = intensities[sample_idx]
    # create grid 'container'
    mic_grid = pygpc.randomgrid(pdftype=pdftype,
                                gridshape=pdf_shape_all[0],
                                limits=limits_all[0],
                                N=1)

    # insert coords 6 times
    mic_grid.coords = np.hstack(
        (mic_grid_conductivities.coords, locations_transform, euler_angles))
    del mic_grid_conductivities

    # compute norms per conditions,
    # problem is, different norm needed per condition. so which condition is mic_grid.coords[i]?
    # conditions_unique are in original order, n_conditions and mic_grid in sample_idx order
    coords_norm = []
    for i in range(n_conditions):
        coords_norm.append(pygpc.norm(mic_grid.coords[conditions == conditions_unique[i]],
                                      pdftype,
                                      pdf_shape_all[i],
                                      limits_all[i]))
    mic_grid.coords_norm = np.concatenate(coords_norm)
    del coords_norm

    # now repeat the last dim-fixed parameters n_condition times
    # create grid container with correct dimensions
    gridshape_large = [np.concatenate([pdf_shape_all[0][0][:dim_fixed],
                                       np.tile(pdf_shape_all[0][0][dim_fixed:], n_conditions)]),
                       np.concatenate([pdf_shape_all[0][1][:dim_fixed],
                                       np.tile(pdf_shape_all[0][1][dim_fixed:], n_conditions)])]
    limits_large = [np.concatenate([limits_all[0][0][:dim_fixed],
                                    np.tile(limits_all[0][0][dim_fixed:], n_conditions)]),
                    np.concatenate([limits_all[0][1][:dim_fixed],
                                    np.tile(limits_all[0][1][dim_fixed:], n_conditions)])]

    mic_grid_large = pygpc.randomgrid(pdftype=np.repeat('beta', dim_fixed + n_conditions * (dim - dim_fixed)),
                                      gridshape=gridshape_large,
                                      limits=limits_large,
                                      N=1)

    del gridshape_large, limits_large
    # then add fixed dims and 6 * location and orientation
    mic_grid_large.coords = np.hstack((mic_grid.coords[:, : dim_fixed],
                                       np.tile(mic_grid.coords[:, dim_fixed:], n_conditions)))
    mic_grid_large.coords_norm = np.hstack((mic_grid.coords_norm[:, : dim_fixed],
                                            np.tile(mic_grid.coords_norm[:, dim_fixed:], n_conditions)))
    mic_grid_large.N = mic_grid_n_exp

    mic = mic_grid.coords.shape[0] * [0]

    del mic_grid
    # determine MIC for each conductivity combination (all positions each)

    for i in range(len(mic)):
        mic[i] = calc_mic(e_gpc,
                          e_coeffs,
                          mic_grid_large.coords_norm,
                          i,
                          conditions,
                          meps,
                          intensities,
                          conditions_unique)

    # perform gPC expansion
    MIC_gpc = pygpc.reg(pdftype=pdftype,
                        pdfshape=pdfshape,
                        limits=limits,
                        order=4,
                        order_max=4,
                        interaction_order=1,
                        grid=mic_grid)

    MIC_coeffs = MIC_gpc.expand(mic)

    # postprocessing of MIC
    ########################################################################################
    MIC_mean = MIC_gpc.mean(MIC_coeffs)
    MIC_std = MIC_gpc.std(MIC_coeffs)
    MIC_sobol, MIC_sobol_idx = MIC_gpc.sobol(MIC_coeffs)
    MIC_globalsens = MIC_gpc.globalsens(MIC_coeffs)

    # TODO: save in hdf5, call writeXDMF()
    # postprocessing of E (optional)
    # E_mean = E_gpc.mean(E_coeffs)
    # E_std = E_gpc.std(E_coeffs)
    # E_sobol, E_sobol_idx = E_gpc.sobol(E_coeffs)
    # E_globalsens = E_gpc.globalsens(E_coeffs)

    # run adaptive gpc (regression) passing the goal function func(x, args()) of MIC
    ########################################################################################
    # MIC_gpc, MIC = pygpc.run_reg_adaptive(pdftype=pdftype,
    #                                   pdfshape=pdfshape,
    #                                   limits=limits,
    #                                   func=CALC_MIC_AS_FUNCTION_OF_E, # E_gpc.evaluate
    #                                   args=(E_gpc,E_coeffs, ___________),
    #                                   order_start=0,
    #                                   order_end=10,
    #                                   eps=1E-3,
    #                                   print_out=True,)
    #
    # # perform final gpc expansion including all simulations
    # MIC_coeffs = MIC_gpc.expand(MIC)
    """


def calc_e(parameters, mesh_fn, tensor_fn, results_folder, coil_fn, positions_mean, v):
    """
    This is to be called by run_reg_adaptive. It build the SIMNIBS parameters from the gpc parameters.
    It checks whether the e was already computed (== is .hdf5 present on disk)
        if so: it checks, if the .hdf5 has the e values in it
        if not: 
            .hdf5 is deleted
            simnibs is called
            .xdmf is written (does not work)
    
    As this also gets called within a dispy node, all used python packages have to be imported again here.
    Also, PYTHONPATH may not be set correct, so add python packages with sys.path.append()
    
    Then, there are numerical deviances between machines, which lead to different .hdf5 filenames (as they are built
        from the gpc parameters. That's why a search algorithm is implemented to look for filenames which are near to 
        one that was computed by the present call of this function.

    This is written for 15484.08, so be aware of some hardcoded stuff.
    :type v: from svd
    :type positions_mean: np.array((3*?)), mean coil positions from experiment
            
    :type coil_fn: basestring, filename of coil.ccd
    :type results_folder: basestring, where to save simnibs results
    :type tensor_fn: basestring, filename of tensor.msh for simnibs input
    :type parameters: list[*], list of wm,gm,csf cond and coil position parameters from gpc grid 
    :type mesh_fn: basestring, .msh filename for simnibs input
    
    :return: Filename of calculated E
    
    """
    import os
    import numpy as np
    import sys
    sys.path.append('/data/pt_01756/software/git/pyfempp')
    sys.path.append('/data/pt_01756/software/git/pygpc')
    getdp_bin = '/data/pt_01756/software/git/pyfempp/pyfempp/pckg/simnibs/bin/getdp'
    from pyfempp.pckg.simnibs.simulation import nnav
    # from preprocessing.uncertainty_estimation_position import euler_angles_to_rotation_matrix
    # from pygpc.misc import euler_angles_to_rotation_matrix
    import h5py
    # import numpy as np
    # import socket
    import pyfempp
    hdf5_geo_fn = "/data/pt_01756/probands/15484.08/simulations_ani_trans_ts_00/" + \
                  "15484.08-0001_MagVenture_MC_B60_1233.hdf5"
    tensor_scaling = False
    no_positions = False
    # we pass the i_grid counter to this function for the filename.
    if len(parameters) != 2:
        raise NotImplementedError('Provide parameters as [i_grid,grirow]')
    i_grid = parameters[0]
    parameters = parameters[1]
    if len(parameters) >= 9:
        tensor_scaling = parameters[3]
    elif len(parameters) == 4:
        no_positions = True
    else:
        raise NotImplementedError('Unknown number of parameters')

    testing = False
    results_fn = os.path.join(results_folder, str(i_grid).zfill(4))
    hdf5_fn = results_fn + '-0000_' + os.path.basename(coil_fn)[:-4] + '.hdf5'
    # hdf5_fn = os.path.join(results_folder,  '_' + str(i_grid).zfill(4) + '.hdf5')
    # the e calc keeps crashing ramdonly, i have no idea why.
    # so repeat until it's fine
    while True:
        try:
            # # if not testing:
            # #     print(socket.gethostname())
            # # build filename from parameters
            # results_fn = results_folder + '/' + ''.join(str(e) + '_' for e in parameters)[:-1]
            # coil_fn_stripped = os.path.basename(coil_fn)
            # coil_fn_stripped = os.path.splitext(os.path.splitext(coil_fn_stripped)[0])[0]
            # hdf5_fn = results_fn + '-0000_' + coil_fn_stripped + '.hdf5'
            # # print '{0:.20f}'.format(parameters[7])
            #
            # # sometimes there are extremely annoying rounding differences at the last digits
            # fn_len = len(parameters)
            # par_idx = 0
            # error_idx = 0
            # results_fn_org = results_fn
            # numerical_dev = False
            # error_tmp = 0
            # error = [-1e-14, 1e-14, -2e-14, 2e-14, -3e-14, 3e-14, -1e-13, 1e-13]
            # # iterate through all parameters of the filename
            # while not os.path.exists(hdf5_fn):
            #     error_tmp = error[error_idx]
            #     numerical_dev = True
            #     # while par_idx < fn_len:
            #     # quit if no parameter change leads to an existing file.
            #     if par_idx == fn_len:
            #         results_fn = results_fn_org
            #         numerical_dev = False
            #         break
            #
            #     # Add or substract possible numerical deviation
            #     parameters[par_idx] += error_tmp
            #
            #     # create filename with this new parameter
            #     results_fn = results_folder + '/' + ''.join(str(e) + '_' for e in parameters)[:-1]
            #
            #     hdf5_fn = results_fn + '-0000_' + os.path.basename(coil_fn)[:-4] + '.hdf5'
            #
            #     # reset parameter to original value
            #     parameters[par_idx] -= error_tmp
            #     error_idx += 1
            #     if error_idx == len(error):
            #         par_idx += 1
            #         error_idx = 0
            #
            # if numerical_dev:
            #     print("Parameter " + str(par_idx) + " changed by " + str(error_tmp))
            # del fn_len, par_idx, results_fn_org, error, numerical_dev, error_idx, error_tmp

            # do not recompute e if already done
            if testing:
                print(hdf5_fn)
            if os.path.exists(hdf5_fn):
                # if not testing:
                #     print(hdf5_fn + ' already exists.')
                # else:
                #     sys.stdout.write(".")
                #     sys.stdout.flush()

                # try:
                #     # hdf = h5py.File(hdf5_fn, 'r')
                #     with h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                #         with h5py.File(hdf5_fn, 'r') as hdf:
                #             e = hdf['/data/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                # except KeyError:
                try:
                    with h5py.File(hdf5_geo_fn, 'r') as hdf_geo, h5py.File(hdf5_fn, 'r') as hdf:
                            e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                            #                 return e.flatten()
                            return hdf5_fn  # e.flatten()
                except (KeyError, IOError):
                    print(hdf5_fn + " has no E data. "
                                    "Removing all stimulation files for this condition and computing E again")
                    files_to_remove = glob.glob(hdf5_fn[:-4] + '*')
                    map(os.remove, files_to_remove)

            if testing:
                print(hdf5_fn)
                quit()

            # build poslist with one position (from parameters)

            poslist = nnav.POSLIST()
            poslist.cond[0].value = parameters[0]  # WM
            poslist.cond[1].value = parameters[1]  # GM
            poslist.cond[2].value = parameters[2]  # CSF
            poslist.cond[3].value = 0.010  # Skull
            poslist.cond[4].value = 0.25  # Scalp
            poslist.fnamecoil = coil_fn

            pos = nnav.POSITION()
            pos.name = '0'

            if no_positions:
                # build mean matsimnibs from disk
                matsimnibs = pyfempp.calc_coil_transformation_matrix(LOC_mean=positions_mean[0:3, 3],
                                                                     ORI_mean=positions_mean[0:3, 0:3],
                                                                     LOC_var=np.array([0, 0, 0]),
                                                                     ORI_var=np.array([0, 0, 0]),
                                                                     V=v)
            else:
                # get coil transformation matrix
                matsimnibs = pyfempp.calc_coil_transformation_matrix(LOC_mean=positions_mean[0:3, 3],
                                                                     ORI_mean=positions_mean[0:3, 0:3],
                                                                     LOC_var=parameters[4:7],
                                                                     ORI_var=parameters[7:10],
                                                                     V=v)

            # matsimnibs depends on svd. with setting seed, the same svds are created (also multiprocessed).
            pos.matsimnibs = matsimnibs
            getdp_options = dict(getdp_bin=getdp_bin)
            # mesh = gmsh.read_msh(fn_mesh)
            poslist.anisotropy_type = 'vn'
            poslist.fn_tensor = tensor_fn
            poslist.poscoil.append(pos)
            poslist.run_single_simulation(mesh_fn,
                                          results_fn,
                                          cpus=2,  # or 2?
                                          keepall=False,
                                          tensor_scaling=tensor_scaling,
                                          fields=['E'],
                                          verbosity='WARNING',  # nE,J,...
                                          **getdp_options)  # path to getdp

            # try:
            pyfempp.write_xdmf(hdf5_fn, hdf5_geo_fn=hdf5_geo_fn, overwrite_xdmf=True)
            #     with h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
            #         with h5py.File(hdf5_fn, 'r') as hdf:
            #             e = hdf['/data/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
            # except Exception:
            #     print "Write_xdmf failed on " + hdf5_fn
            # with h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
            #     with h5py.File(hdf5_fn, 'r') as hdf:
            #         e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
            # all e where x of first node is inside left hemisphere:
            # e = hdf['/mesh/fields/E/value'][:][np.all([hdf_geo['/mesh/elm/elm_type'][:] == 2, hdf_geo['/mesh/nodes/node_coord'][:][hdf_geo['/mesh/elm/node_number_list'][:,0] - 1][:,0] < 0] ,axis=0)]
            return hdf5_fn  # e.flatten()

        except Exception:
            print "calc_e crashed on " + hdf5_fn + ". Repeating calculation"


def tau_workhorse(elm_idx_list, fitted_mep, mep_params, n_samples,
                  e,
                  e_tan,
                  e_norm):
    """Worker function for TAU computation - call from multiprocessing.pool

    Calculates mic for E, Eort and Epar for given zaps and elements.
    idxLst is used for multiprocessed,
    n_samples are taken from fitted_mep, within the range of the mep object

    Args:
        elm_idx_list:  which elms[*] to calc?
        fitted_mep: List of fited MEP object for this subject
        mep_params: parameters of curve fits used to calculate the MEP
        n_samples: how many samples to take? 
        e: list of n_condition. each list of |e| for roi
        e_tan: same for tangential part
        e_norm: same for normal part

    Returns:
        A 3-tuple: (tau, mic_norm, tau_tan)
    """
    tau = np.empty((len(elm_idx_list)))
    tau_norm = np.empty((len(elm_idx_list)))
    tau_tan = np.empty((len(elm_idx_list)))
    # mine = MINE(alpha, c)  # 10.1093/bioinformatics/bts707
    n_conditions = len(fitted_mep)

    # rearrange mep parameters to individual conditions
    mep_params_cond = []
    start_idx = 0
    for cond in range(n_conditions):
        mep_params_cond.append(mep_params[start_idx:(start_idx+fitted_mep[cond].popt.shape[0])])
        start_idx = start_idx+fitted_mep[cond].popt.shape[0]

    del start_idx

    intensities = []
    amp = []
    # build list of intensity values per condition
    for cond in range(n_conditions):
        intensities.append(np.arange(fitted_mep[cond].x_limits[0],
                                     fitted_mep[cond].x_limits[1],
                                     step=(fitted_mep[cond].x_limits[1] - fitted_mep[cond].x_limits[0]) /
                                          float(n_samples)))
        amp.append(fitted_mep[cond].eval(intensities[-1], mep_params_cond[cond]))

    for idx, elmIdx in enumerate(elm_idx_list):

        e_to_compute = np.empty((n_conditions, intensities[0].shape[0]))
        e_tan_to_compute = np.empty((n_conditions, intensities[0].shape[0]))
        e_norm_to_compute = np.empty((n_conditions, intensities[0].shape[0]))

        # e, etan, norm and amp are lists with len = n_conditions
        for cond in range(n_conditions):
            e_to_compute[cond] = e[cond][elmIdx] * intensities[cond]
            e_tan_to_compute[cond] = e_tan[cond][elmIdx] * intensities[cond]
            e_norm_to_compute[cond] = e_norm[cond][elmIdx] * intensities[cond]

        tau[idx] = cc_workhorse(e_to_compute, amp)
        tau_norm[idx] = cc_workhorse(e_norm_to_compute, amp)
        tau_tan[idx] = cc_workhorse(e_tan_to_compute, amp)
        # print("{}:{}".format(idx, len(elm_idx_list)))
    return tau, tau_norm, tau_tan


def cc_workhorse(e, amp):
    """Cross-correlation measure for multiple MEP slopes per element
    
    np.correlation needs zero padded values, so for each condition one zero padded array for
    E vs MEPamp is created. All arrays have to be of the same length and same stepsize.
    
    The n_conditions-1 mep slopes are cross-correlated with the median mep slope.
    The abs(delta E) with max cross-correlation per condition are weighted by median(E) and summed up.
        This favors elements which have greater E, as these are more likely to produce MEP.
    """

    stepsize = 1e-1
    n_condition = len(amp)
    e_min = np.min(e, axis=1)
    # ceil to .stepsize
    e_min = np.ceil(e_min / stepsize) * stepsize
    e_max = np.max(e, axis=1)
    e_max = np.floor(e_max / stepsize) * stepsize

    # return NaN if xmax-xmin is smaller than stepsize
    if np.any(e_max-e_min <= stepsize):
        return np.nan

    else:
        # stepsize-wise e over all conditions. we only need the length of this and first elm
        e_all_cond = np.arange(np.min(e_min), np.max(e_max) + stepsize, stepsize)

        e_len = len(e_all_cond)
        # mep_arr = []
        mep_y_all_cond = []
        start_ind = np.empty(n_condition, dtype=int)
        stop_ind = np.empty(n_condition, dtype=int)
        # e_x_cond_all = []
        for idx in range(n_condition):
            # x range for e for conditions, stepsize wise
            e_x_cond = np.arange(e_min[idx], e_max[idx], stepsize)
            # e_x_cond_all.append(e_x_cond)

            # interpolate mep values to stepsize width
            mep_y_all_cond.append(np.interp(e_x_cond, e[idx], amp[idx]))
            # mep_y_all_cond.append(mep_y_cond)

            # setup zero spaced array
            # global_e_mep_arr = np.zeros(e_len)
            # mep_arr_cond[:] = np.random.rand(int(e_len)) * 3

            # lower boundary idx of e_x_cond in e_arr
            start_idx = int((e_x_cond[0] - np.min(e_min)) / stepsize)
            stop_idx = start_idx + len(e_x_cond)
            stop_ind[idx] = stop_idx
            start_ind[idx] = start_idx

            # overwrite e_x_cond range of mep_arr_cond with interpolated mep values
            # global_e_mep_arr[start_idx:stop_idx] = mep_y_cond
            # mep_arr_cond[start_idx + len(e_x_cond):] = mep_y_cond[-1] # tailing last
            # mep_arr.append(global_e_mep_arr)

        # find median mep cond
        e_mean = np.mean((e_max + e_min) / 2)

        # get tau distances for all conditions vs median condition
        # distances for ref,i == i,ref. i,i == 0. So only compute upper triangle of matrix
        ref_range = np.arange(n_condition)
        t_cond = np.zeros((n_condition, n_condition))
        idx_range = list(reversed(np.arange(n_condition)))
        # min_err_idx_lst_all = []
        for reference_idx in ref_range:
            # remove this reference index from idx_range
            idx_range.pop()
            # as we always measure the distance of the shorter mep_cond, save idx to store in matrix
            reference_idx_backup = reference_idx
            # min_err_idx_lst = np.zeros((n_condition, 1))
            # t = np.zeros((n_condition, 1))
            for idx in idx_range:
                # print((reference_idx, idx))
                idx_save = idx
                # restore correct reference idx
                reference_idx = reference_idx_backup

                # get lengths of mep_y
                len_mep_idx = mep_y_all_cond[idx].shape[0]
                len_mep_ref = mep_y_all_cond[reference_idx].shape[0]

                # switch ref and idx, as we want to measure from short mep_y
                if len_mep_idx < len_mep_ref:
                    reference_idx, idx = idx, reference_idx
                    len_mep_idx, len_mep_ref = len_mep_ref, len_mep_idx

                # create array: global e + 2* len(mep[idx])
                shift_array = np.zeros(2 * len_mep_idx + e_len)

                # and paste reference mep values. errors will be measured against this array
                shift_array[len_mep_idx +
                            start_ind[reference_idx]: len_mep_idx +
                                                      stop_ind[reference_idx]] = mep_y_all_cond[reference_idx]

                # shift mep[idx] step wise over the shift_array and measure error

                # instead of for loop, I'll use multple slices:
                # slice_indices[0] is 0-shifting
                # slice_indices[1] is 1-shifting,...
                # we start shifting at start_ind[reference_idx], because range left of that is only 0
                # we stop shifting after len_mep_idx + e_len - stop_ind[reference_idx] times
                # slice_indices.shape == (len_mep_idx + e_len - stop_ind[reference_idx], len_mep_idx)
                slice_indices = np.add(np.arange(start_ind[reference_idx],
                                                 len_mep_idx + start_ind[reference_idx]),
                                       np.arange(len_mep_idx + e_len - stop_ind[reference_idx])[:, np.newaxis])

                # compute error vectorized
                # the error is y-difference between mep[idx] and mep[reference].zero_padded
                err = np.sqrt(np.sum((shift_array[slice_indices] - mep_y_all_cond[idx]) ** 2, axis=1))

                # 3 times slower for loop version:
                # for step in range(len_mep_idx + e_len):
                #     err[step] = np.sqrt(np.sum(np.square(shift_array[step:len_mep_idx+step] -
                #                                          mep_y_all_cond[idx]))) / len_mep_idx

                # which shift leads to minimum error. remember that we don't start at 0-shift, so add start index
                min_err_idx = np.argmin(err) + start_ind[reference_idx]
                # min_err_idx_lst[idx] = min_err_idx

                # rescale min_error_idx to real E values
                t_cond[reference_idx_backup, idx_save] = np.abs(stop_ind[idx] - min_err_idx) * stepsize

        # sum all errors and divide by e_mean over all conditions
        return 1 / (np.sqrt(np.sum(np.square(t_cond) * 2)) / e_mean / n_condition / (n_condition - 1))



def compute_chunks(seq, num):
    """
    Calculate chunks which are of similar size.

    Args:
        seq: list of int's
        num: number of chunks to generate
    Returns:
        list of num sublists, each ~ same length.
    """

    avg = len(seq) / float(num)
    if avg < 1:
        raise ValueError("seq/num ration too small: " + str(avg))
    else:
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    if np.isnan(np.dot(arr[:, 1], arr[:, 1])):
        nans = sum(np.isnan(arr[:, 1]))
        print(str(nans) + ' zero-length normal vectors found!' +
              'Replacing by 0-0-0')
        arr = np.nan_to_num(arr)
    return arr


def calc_tau(gpc_params, fixed_egpc_par, coeffs_e_norm, coeffs_e_tan, e_gpc_obj, mep_fits, n_samples,
             four_parameters_e_gpc):
    """Gets called by run_reg_adaptive
    
    Params
    ----------------------------------------------------------
    gpc_params: list of gpc parameters from grid
                WM, GM, CSF, TS, 6*MEP fit parameter
    fixed_egpc_par: List, len == n_egpc_params - 4
    four_parameters_e_gpc: boolean 
        If set, e-gpc objects are provided with 4 params. Otherwise, 4 + len(fixed_egpc_par)  
    
    """

    e = []
    e_tan = []
    e_norm = []
    # get E for all condition, i = 0:5
    for i in range(len(e_gpc_obj)):

        # build e.eval parameters. first 4 are from mic_gpc, rest is fixed. (probably 0)
        if four_parameters_e_gpc:
            e_gpc_params = np.stack(gpc_params[0:4])[np.newaxis, :]
        else:
            e_gpc_params = np.hstack((gpc_params[0:4], fixed_egpc_par[i].flatten()))[np.newaxis,:]

        # normalize gridpoints from mic_gpc to pass to e_gpc object
        e_gpc_params_norm = pygpc.norm(e_gpc_params,
                                       e_gpc_obj[i].pdftype,
                                       e_gpc_obj[i].pdfshape,
                                       e_gpc_obj[i].limits)

        # calculate normal and tangential component from e_gpc
        e_norm_vec = e_gpc_obj[i].evaluate(coeffs_e_norm[i], e_gpc_params_norm)
        e_tan_vec = e_gpc_obj[i].evaluate(coeffs_e_tan[i], e_gpc_params_norm)

        e_norm_vec = np.reshape(e_norm_vec, (e_norm_vec.shape[1] / 3, 3), order='c')
        e_tan_vec = np.reshape(e_tan_vec, (e_tan_vec.shape[1] / 3, 3), order='c')

        # get magnitudes
        e_norm.append(np.linalg.norm(e_norm_vec, axis=1))
        e_tan.append(np.linalg.norm(e_tan_vec, axis=1))
        e.append(np.linalg.norm(e_norm_vec + e_tan_vec, axis=1))

    # get mic for every element of roi over all condition
    n_cpu = multiprocessing.cpu_count()
    n_cpu = min(n_cpu, 30)
    elm_idx_list = compute_chunks(range(len(e_norm[0])), n_cpu - 1)
    tau_func = partial(tau_workhorse,
                       fitted_mep=mep_fits,
                       mep_params=gpc_params[4:],
                       n_samples=n_samples,
                       e=e,
                       e_tan=e_tan,
                       e_norm=e_norm)

    # start = time.time()
    pool = multiprocessing.Pool(n_cpu)
    # print('      pool: ' + str(time.time() - start) + ' sec')

    # start = time.time()
    tau, tau_norm, tau_tan = zip(*pool.map(tau_func, elm_idx_list))
    pool.close()
    pool.join()

    tau_all = np.hstack(tau)
    tau_norm_all = np.hstack(tau_norm)
    tau_tan_all = np.hstack(tau_tan)

    return np.hstack((tau_all, tau_norm_all, tau_tan_all))


def get_bad_param(pdf_paras_location, pdf_paras_orientation_euler,
                  pos_mean,
                  v,
                  del_obj,
                  coil_fn,
                  condition,
                  save_hdf5=False, folder="/data/pt_01756/misc/coils_pos/"):
    """
    Finds gpc parameter combinations, which place coil dipoles inside subjects brain.
    
    Only endpoints (and midpoints) of the parameter ranges are examined. 
    
    :param pdf_paras_location: 
    :type pdf_paras_location: 
    :param pdf_paras_orientation_euler: 
    :type pdf_paras_orientation_euler: 
    :param pos_mean: 
    :type pos_mean: 
    :param v: from svd
    :type v: 
    :param del_obj: Delaunay object of subjects head
    :type del_obj: Delaunay
    :param coil_fn: 
    :type coil_fn: 
    :param condition: 
    :type condition: 
    :param save_hdf5: Coil dipoles from each parameter combination may be into .hdf5 files. .xdmf are created as well. 
    :type save_hdf5: bool
    :type folder: Where to save .hdf5 files
    
    :return: Parameter combinations, which led to dipoles in head. Empty if none
    :rtype: list
    """

    # for every condition:
    results = []

    # for i in range(6):
    limits_pos_x = [pdf_paras_location[condition][0][0][2], pdf_paras_location[condition][0][0][3]]
    limits_pos_y = [pdf_paras_location[condition][1][0][2], pdf_paras_location[condition][1][0][3]]
    limits_pos_z = [pdf_paras_location[condition][2][0][2], pdf_paras_location[condition][2][0][3]]
    limits_yaw = [pdf_paras_orientation_euler[condition][0][0][2], pdf_paras_orientation_euler[condition][0][0][3]]
    limits_pitch = [pdf_paras_orientation_euler[condition][1][0][2], pdf_paras_orientation_euler[condition][1][0][3]]
    limits_roll = [pdf_paras_orientation_euler[condition][2][0][2], pdf_paras_orientation_euler[condition][2][0][3]]

    limits_pos_x = add_center(limits_pos_x)
    limits_pos_y = add_center(limits_pos_y)
    limits_pos_z = add_center(limits_pos_z)
    limits_yaw = add_center(limits_yaw)
    limits_pitch = add_center(limits_pitch)
    limits_roll = add_center(limits_roll)

    temp_list = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    combinations = list(itertools.product(*temp_list))
    del temp_list
    for combination in combinations:

        # create matsimnibs
        loc_var = np.array([limits_pos_x[combination[0]],
                            limits_pos_y[combination[1]],
                            limits_pos_z[combination[2]]])
        ori_var = np.array([limits_yaw[combination[3]],
                            limits_pitch[combination[4]],
                            limits_roll[combination[5]]])
        mat = pyfempp.calc_coil_transformation_matrix(LOC_mean=pos_mean[condition][0:3, 3],
                                                      ORI_mean=pos_mean[condition][0:3, 0:3],
                                                      LOC_var=loc_var,
                                                      ORI_var=ori_var,
                                                      V=v[condition])

        # get dipole points for coil for actual matsimnibs
        coil_dipoles = pyfempp.get_coil_dipole_pos(coil_fn, mat)
        if save_hdf5:

            xdmf_fn = folder + '_' + str(condition) + '' + str(combination) + ".hdf5"
            with h5py.File(xdmf_fn, 'w') as f:
                f.create_dataset("/dipoles/", data=coil_dipoles)
            with open(folder + '_' + str(condition) + '' + str(combination) + ".xdmf", 'w') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
                f.write('<Domain>\n')

                # one collection grid
                f.write('<Grid\nCollectionType="Spatial"\nGridType="Collection"\nName="Collection">\n')

                f.write('<Grid Name="coil" GridType="Uniform">\n')
                f.write('<Topology NumberOfElements="' + str(len(coil_dipoles)) +
                        '" TopologyType="Polyvertex" Name="Tri">\n')
                f.write('<DataItem Format="XML" Dimensions="' + str(len(coil_dipoles)) + ' 1">\n')
                # f.write(hdf5_fn + ':' + path + '/triangle_number_list\n')
                np.savetxt(f, range(len(coil_dipoles)), fmt='%d',
                           delimiter=' ')  # 1 2 3 4 ... N_Points
                f.write('</DataItem>\n')
                f.write('</Topology>\n')

                # nodes
                f.write('<Geometry GeometryType="XYZ">\n')
                f.write('<DataItem Format="HDF" Dimensions="' + str(len(coil_dipoles)) + ' 3">\n')
                f.write(xdmf_fn + ':' + '/dipoles\n')
                f.write('</DataItem>\n')
                f.write('</Geometry>\n')

                f.write('</Grid>\n')
                f.write('</Grid>\n')
                f.write('</Domain>\n')
                f.write('</Xdmf>\n')

        # check hull and add to results
        results.append(pyfempp.in_hull(coil_dipoles, del_obj))

    # find parameter which drives dipoles into head
    combinations = np.array(combinations)
    fail_params = []
    if np.sum(results) > 0:
        comb_idx = np.where(np.sum(results, axis=1) > 0)[0]
        params_idx = np.where(np.var(combinations[comb_idx], axis=0) ==
                              min(np.var(combinations[comb_idx], axis=0)))[0]
        for param_idx in params_idx:
            comb_idx = comb_idx[combinations[comb_idx, params_idx[0]] != 1]
            fail_params.append((param_idx, combinations[comb_idx, params_idx[0]][0]))

            # this gives all combinations for the failed parameters which lead to dipoles inside head
            # combinations[np.array(comb_idx)[:, np.newaxis], params_idx]

            # # find consecutive idx:
            # from itertools import groupby
            # from operator import itemgetter
            # successives = []
            # for k, g in groupby(enumerate(comb_idx), lambda (i, x): i - x):
            #     successives.append(map(itemgetter(1), g))
            #     print map(itemgetter(1), g)
            #
            #     find parameter
            #     for successive in successives:
            #         combinations[successive]

    return fail_params


def add_center(var):
    """Add center to argument list.
        
        var: list of float <[f1,f2]>
        
        returns list of float <[f1,mean(f1,f2),f2]>"""
    return [var[0], sum(var) / 2, var[1]]


def fit_meps(results_folder, data_folder):
    """ Create MEP fit-objects from experimental data, one per condition.
    These are loaded in tau_gpc() for amplitude computation.
    
    :param data_folder: folder with results_conditions.csv and simPos.csv in it
    :type data_folder: str
    :param results_folder: where to save fit objects
    :type results_folder: str
    :return: results_folder/mep_fit_cond*.pkl'
    :rtype: file
    """

    results_conditions_fn = os.path.join(data_folder, 'results_conditions.csv')
    sim_pos_fn = os.path.join(data_folder, 'simPos.csv')

    positions_all, conditions, position_list, mep_amp, intensities = pyfempp.read_exp_stimulations(
        results_conditions_fn, sim_pos_fn)
    conditions = np.array(conditions)

    # sort data by condition, but keep original order of conditions
    data_sorted = pyfempp.sort_data_by_condition(conditions, False, mep_amp, intensities)
    _, idx = np.unique(conditions, return_index=True)
    # fit mep amplitudes to function
    mep = []

    for i in range(len(np.unique(conditions))):
        mep.append(pyfempp.Mep(intensities=data_sorted[1][i],
                               mep=data_sorted[0][i]))
        mep[i].fit_mep_to_function_multistart(p0=[70, 0.6, 1])
        mep[i].plot(label=conditions[np.sort(idx)][i],
                    sigma=2,
                    plot_samples=True,
                    show_plot=False,
                    fname_plot=os.path.join(data_folder, 'mep_vs_intensity_fit_cond_' + str(i) + '.png'))
        pyfempp.save_mepobj(mep[i], os.path.join(results_folder, 'mep_fit_cond' + str(i) + '.pkl'))


def fancy_bar(text, i, n_i):
    if not isinstance(text, basestring):
        i = str(i)

    assert isinstance(text, basestring)
    assert isinstance(n_i, int)
    if not text.endswith(' '):
        text += ' '

    sys.stdout.write('\r')
    sys.stdout.write(text + i.zfill(4) + " from " + str(n_i))
    # this prints [50-spaces], i% * =
    sys.stdout.write(" [%-40s] %d%%" % (
        '=' * int((float(i) + 1) / n_i * 100 / 2.5), float(i) / n_i * 100))
    sys.stdout.flush()
    if int(i) == n_i:
        print ""


if __name__ == '__main__':
    main()
