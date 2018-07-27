# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:21:45 2016

@author: Konstantin Weise
"""
import numpy as np
import dill           # module for saving and loading object instances
import pickle         # module for saving and loading object instances
import h5py
import sys
import time
import scipy
import random
import os
import sys
import warnings
import yaml
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

from _functools import partial
import multiprocessing
import multiprocessing.pool

from .grid import quadrature_jacobi_1D
from .grid import quadrature_hermite_1D
from .grid import randomgrid
from .grid import tensgrid
from .grid import sparsegrid

from .misc import unique_rows
from .misc import allVL1leq
from .misc import euler_angles_to_rotation_matrix
from .misc import fancy_bar

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
class NondaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def save_gpcyml(gobj, fname):
    """save gpc infos about input pdfs, polynomials, grid etc. as .yml file

    save_gpcyml(gobj, fname)

    Parameters:
    ---------------------------
    gobj: object
        gpc object including infos to save
    fname: str
        filename
    """

    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # write information to dictionary
    info = dict(random_vars=gobj.random_vars,
                pdftype=gobj.pdftype.tolist(),
                pdfshape=gobj.pdfshape,
                limits=gobj.limits,
                order=gobj.order.tolist(),
                order_max=gobj.order_max,
                interaction_order=gobj.interaction_order,
                grid_coords=gobj.grid.coords.tolist(),
                grid_coords_norm=gobj.grid.coords_norm.tolist(),
                gpc_kind=gobj.__class__.__name__,
                grid_kind=gobj.grid.__class__.__name__)

    # add grid specific attributes to dictionary
    if gobj.grid.__class__.__name__ == 'randomgrid':
        info['seed'] = gobj.grid.seed

    if gobj.grid.__class__.__name__ == 'tensgrid':
        info['gridtype'] = gobj.grid.gridtype
        info['N'] = gobj.grid.N
        info['weights'] = gobj.grid.weights.tolist()

    if gobj.grid.__class__.__name__ == 'sparsegrid':
        info['gridtype'] = gobj.grid.gridtype
        info['level'] = gobj.grid.level
        info['level_max'] = gobj.grid.level_max
        info['order_sequence_type'] = gobj.grid.order_sequence_type
        info['weights'] = gobj.grid.weights.tolist()
        info['level_sequence'] = gobj.grid.level_sequence
        info['order_sequence'] = gobj.grid.order_sequence
        info['l_level'] = gobj.grid.l_level.tolist()

    # write in file
    with open(fname, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


def load_gpcyml(fname):
    """load gpc infos about input pdfs, polynomials, grid etc. as .yml file and initialize gpc object

    gpcobject = load_gpcyml(fname)

    Parameters:
    ---------------------------
    fname: str
        filename
    """

    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # read yml file
    with open(fname, 'r') as f:
        info = yaml.load(f)

    # initialize grid object
    if info['grid_kind'] == 'randomgrid':
        grid = randomgrid(pdftype=info['pdftype'],
                          gridshape=info['pdfshape'],
                          limits=info['limits'],
                          N=0,
                          seed=info['seed'])

    elif info['grid_kind'] == 'tensgrid':
        grid = tensgrid(pdftype=info['pdftype'],
                        gridtype=info['gridtype'],
                        gridshape=info['pdfshape'],
                        limits=info['limits'],
                        N=info['N'])
        grid.weights = np.asarray(info['weights'])

    elif info['grid_kind'] == 'sparsegrid':
        grid = sparsegrid(pdftype=info['pdftype'],
                          gridtype=info['gridtype'],
                          gridshape=info['pdfshape'],
                          limits=info['limits'],
                          level=info['level'],
                          level_max=info['level_max'],
                          interaction_order=info['interaction_order'],
                          order_sequence_type=info['order_sequence_type'],
                          make_grid=False)
        grid.weights = np.asarray(info['weights'])
        grid.level_sequence = info['level_sequence']
        grid.order_sequence = info['order_sequence']
        grid.l_level = np.asarray(info['l_level'])
    else:
        raise NotImplementedError

    grid.coords = np.asarray(info['grid_coords'])
    grid.coords_norm = np.asarray(info['grid_coords_norm'])

    # initialize gpc object
    gobj = eval(info['gpc_kind'])(random_vars=info['random_vars'],
                                  pdftype=info['pdftype'],
                                  pdfshape=info['pdfshape'],
                                  limits=info['limits'],
                                  order=info['order'],
                                  order_max=info['order_max'],
                                  interaction_order=info['interaction_order'],
                                  grid=grid)
    return gobj


def save_gpcobj(gobj, fname):
    """ saving gpc object including infos about input pdfs, polynomials, grid etc. as pickle file

    save_gpcobj_pkl(gobj, fname)

    Parameters:
    ---------------------------
    gobj: object
        gpc object to save
    fname: str
        filename with .pkl extension
    """

    with open(fname, 'wb') as output:
        pickle.dump(gobj, output, -1)


def load_gpcobj(fname):
    """ loading gpc object including infos about input pdfs, polynomials, grid etc...

    object = load_gpcobj(filename)

    Parameters:
    ---------------------------
    fname: str
        filename with .pkl extension
    """

    with open(fname, 'rb') as input:
        return pickle.load(input)  


def save_data_txt(data, fname):
    """ saving data (quantity of interest) in .txt file (e.g. coeffs, mean, std, ...)

    save_data_txt(qoi, filename)

    Parameters:
    ---------------------------
    data: 2D np.array()
        data to save
    fname: str
        filename with .hdf5 extension
    """

    np.savetxt(fname, data, fmt='%.10e', delimiter='\t', newline='\n', header='', footer='')

def load_data_hdf5(fname, loc):
    """ loading quantity of interest from .hdf5 file (e.g. coeffs, mean, std, ...)

    load_data_hdf5(data, fname, loc)

    Parameters:
    ---------------------------
    data: 2D np.array()
        data to save
    fname: str
        filename with .hdf5 extension
    loc: str
        location (folder and name) in hdf5 file (e.g. data/phi)
    """

    with h5py.File(fname, 'r') as f:
        d=f[loc]        
        return d
        
def save_data_hdf5(data, fname, loc):
    """ saving quantity of interest in .hdf5 file (e.g. coeffs, mean, std, ...)

    save_data_hdf5(data, fname, loc)

    Parameters:
    ---------------------------
    data: 2D np.array()
        data to save
    fname: str
        filename with .hdf5 extension
    loc: str
        location (folder and name) in hdf5 file (e.g. data/phi)
    """

    with h5py.File(fname, 'a') as f:
        f.create_dataset(loc, data=data)

def save_sobol_idx(sobol_idx, fname):
    """ saving sobol_idx list in file

    save_sobol_idx(sobol_idx, filename)

    Parameters:
    ---------------------------
    sobol_idx: list of np.array [N_sobol]
        List of parameter label indices belonging to Sobol indices
    fname: str
        filename with .txt extension containing the saved sobol indices
    """

    f = open(fname, 'w')
    f.write('# Parameter index list of Sobol indices:\n')
    for i_line in range(len(sobol_idx)):
        for i_entry in range(len(sobol_idx[i_line])):
            if i_entry > 0:
                f.write(', ')
            f.write('{}'.format(sobol_idx[i_line][i_entry]))
        if i_line < list(range(len(sobol_idx))):    
            f.write('\n')
        
    f.close()    
    
def read_sobol_idx(fname):
    """  reading sobol_idx list from file

    read_sobol_idx(filename)

    Parameters:
    ---------------------------
    fname: str
        filename with .txt extension containing the saved sobol indices

    Returns:
    ---------------------------
    sobol_idx: list of np.array [N_sobol]
        List of parameter label indices belonging to Sobol indices
    """

     
    f = open(fname,'r')

    line = f.readline().strip('\n')
    sobol_idx = []

    while line:
        
        # ignore comments in textfile
        if line[0] == '#':
           line = f.readline().strip('\n')
           continue
        
        else:
            # read comma separated indices and convert to nparray
            sobol_idx.append(np.array([int(x) for x in line.split(',') if x]))
      
        line = f.readline().strip('\n')
    
    return sobol_idx    


def extract_sobol_order(sobol, sobol_idx, order=1):
    """ Extract Sobol indices with specified order from Sobol data

    extract_sobol_order(sobol, sobol_idx, order=1)

    Parameters:
    ----------------------------------
        sobol: np.array() [N_sobol x N_out]
            Sobol indices of N_out output quantities
        sobol_idx: list of np.array [N_sobol]
            List of parameter label indices belonging to Sobol indices
        order: int
            Sobol index order to extract

    Returns:
    ----------------------------------
        sobol_1st: np.array() [N_sobol x N_out]
            1st order Sobol indices of N_out output quantities

        sobol_idx_1st: list of np.array [DIM]
            List of parameter label indices belonging to 1st order Sobol indices
    """

    # make mask of 1st order (linear) sobol indices
    mask = np.asarray([int(i) for i in range(len(sobol_idx)) if sobol_idx[i].shape[0] == order])

    # extract from dataset
    sobol_1st = sobol[mask, :]
    sobol_idx_1st = np.vstack([sobol_idx[i] for i in mask])

    # sort sobol indices according to parameter indices in ascending order
    sort_idx = np.argsort(sobol_idx_1st, axis=0)[:, 0]
    sobol_1st = sobol_1st[sort_idx, :]
    sobol_idx_1st = sobol_idx_1st[sort_idx, :]

    return sobol_1st, sobol_idx_1st


def wrap_function(function, x, args):
    """ function wrapper to call anonymous function with variable number of arguments (tuple)

    wrap_function(function, x, args)

    Parameters:
    ---------------------------
    function: function
        anonymous function to call
    x: parameters
        parameters of function
    args: arguments
        arguments of function

    Returns:
    ---------------------------
    function_wrapper: function
        wrapped function
    """

    def function_wrapper(*wrapper_args):
        return function(*(wrapper_args + x + args))
    
    return function_wrapper
            
def calc_Nc(order, dim):
    """ calc_Nc calculates the number of PCE coefficients by the used order and dimension.

    Nc   = (order+dim)! / (order! * dim!)

    Nc = calc_Nc( order , dim )

    Parameters:
    ---------------------------
    order: int
        global order of expansion
    dim: int
        Number of random variables

    Returns:
    ---------------------------
    Nc: int
        Number of coefficients and polynomials
    """

    return scipy.special.factorial(order+dim) / (scipy.special.factorial(order) * scipy.special.factorial(dim))


def calc_Nc_sparse(p_d, p_g, p_i, dim):
    """ calc_Nc_sparse calculates the number of PCE coefficients for a specific maximum order in each dimension p_d,
        maximum order of interacting polynomials p_g and the interaction order p_i.

    Nc = calc_Nc_sparse(p_d, p_g, p_i, dim)

    Parameters:
    ---------------------------
    p_d: int or np.array of int
        maximum order in each dimension
    p_g: int
        maximum global order of interacting polynomials
    p_i: int
        interaction order
    dim: int
        Number of random variables

    Returns:
    ---------------------------
    Nc: int
        Number of coefficients and polynomials
    """

    p_d = np.array(p_d)
    
    if p_d.size==1:
        p_d = p_d*np.ones(dim)
            
    # generate multi-index list up to maximum order
    if dim == 1:
        poly_idx = np.array([np.linspace(0,p_d,p_d+1)]).astype(int).transpose()
    else:
        poly_idx = allVL1leq(int(dim), p_g)
        
    for i_dim in range(dim):
        # add multi-indexes to list when not yet included
        if p_d[i_dim] > p_g:
            poly_add_dim = np.linspace(p_g+1, p_d[i_dim], p_d[i_dim]-(p_g+1) + 1)
            poly_add_all = np.zeros([poly_add_dim.shape[0],dim])
            poly_add_all[:,i_dim] = poly_add_dim               
            poly_idx = np.vstack([poly_idx,poly_add_all.astype(int)])
                
        # delete multi-indexes from list when they exceed individual max order of parameter     
        elif p_d[i_dim] < p_g:    
            poly_idx = poly_idx[poly_idx[:,i_dim]<=p_d[i_dim],:]
                
    # Consider interaction order (filter out multi-indices exceeding it)
    poly_idx = poly_idx[np.sum(poly_idx>0,axis=1)<=p_i,:]
        
    return poly_idx.shape[0]


def pdf_beta(x, p, q, a, b):
    """ Calculate the probability density function of the beta distribution in the interval [a,b]

    pdf = pdf_beta( x , p , q , a , b )

    pdf = (gamma(p)*gamma(q)/gamma(p+q).*(b-a)**(p+q-1))**(-1) *
              (x-a)**(p-1) * (b-x)**(q-1);

    Parameters:
    ---------------------------
    x: np.array of float
        values of random variable
    a: float
        MIN boundary
    b: float
        MAX boundary
    p: float
        parameter defining the distribution shape
    q: float
        parameter defining the distribution shape

    Returns:
    ---------------------------
    pdf: np.array of float
        probability density
    """
    return (scipy.special.gamma(p)*scipy.special.gamma(q)/scipy.special.gamma(p+q)
            * (b-a)**(p+q-1))**(-1) * (x-a)**(p-1) * (b-x)**(q-1)


def run_reg_adaptive_E_gPC(pdftype, pdfshape, limits, func, args=(), fn_gpcobj=None,
                           order_start=0, order_end=10, interaction_order_max=None, eps=1E-3, print_out=False,
                           seed=None, do_mp=False, n_cpu=4, dispy=False, dispy_sched_host='localhost',
                           random_vars='', hdf5_geo_fn=''):
    """  
    Adaptive regression approach based on leave one out cross validation error estimation
    
    Parameters
    ---------------------------
    random_vars : list of str
        string labels of the random variables
    pdftype : list
        Type of probability density functions of input parameters,
        i.e. ["beta", "norm",...]
    pdfshape : list of lists
        Shape parameters of probability density functions
        s1=[...] "beta": p, "norm": mean
        s2=[...] "beta": q, "norm": std
        pdfshape = [s1,s2]
    limits : list of lists
        Upper and lower bounds of random variables (only "beta")
        a=[...] "beta": lower bound, "norm": n/a define 0
        b=[...] "beta": upper bound, "norm": n/a define 0
        limits = [a,b]
    func : callable func(x,*args)
        The objective function to be minimized.
    args : tuple, optional
        Extra arguments passed to func, i.e. f(x,*args).
    fn_gpcobj : String, None
        If fn_gpcobj exists, regobj will be created from it. If not exist, it is created.
    order_start : int, optional
        Initial gpc expansion order (maximum order)
    order_end : int, optional
        Maximum gpc expansion order to expand to
    interaction_order_max: int
        define maximum interaction order of parameters (default: all interactions)
    eps : float, optional
        Relative mean error bound of leave one out cross validation
    print_out : boolean, optional
        Print output of iterations and subiterations (True/False)
    seed : int, optional
        Set np.random.seed(seed) in random_grid()
    do_mp : boolean, optional
        Do each func(x, *(args)) in each iteration with parmap.starmap(func)
    n_cpu : int, 4, optional
        If (multiprocessing), utilize n_cpu cores
    dispy : boolean, False 
        Compute func(x) in with dispy cluster
    dispy_sched_host : String, localhost
        Host name where dispyscheduler is run. None = localhost 
    hdf5_geo_fn: String, ''
        hdf5 filename with spatial information: /mesh/elm/*
    
    Returns
    ---------------------------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]        
           
    """
    import sys
    sys.path.append('/data/pt_01756/software/git/pyfempp')
    import pyfempp
    try:
        import setproctitle
        eps_gpc = [eps + 1]
        i_grid = 0
        i_iter = 0
        interaction_order_count = 0
        # interaction_order_max = -1
        dim = len(pdftype)
        if not interaction_order_max:
            interaction_order_max = dim
        order = order_start
        run_subiter = True

        # mesh_fn, tensor_fn, results_folder, coil_fn, positions_mean, v = args
        fn_cfg, subject, results_folder, _, _, _ = args

        with open(fn_cfg, 'r') as f:
            config = yaml.load(f)

        mesh_fn = subject.mesh[config['mesh_idx']]['fn_mesh_msh']

        save_res_fn = False
        results = None
        setproctitle.setproctitle("run_reg_adaptive_E_gPC_" + results_folder[-5:])

        # load surface data from skin surface
        msh = pyfempp.read_msh(mesh_fn)
        points = msh.nodes.node_coord
        triangles = msh.elm.node_number_list[((msh.elm.elm_type == 2) & (msh.elm.tag1 == 1005)), 0:3]
        skin_surface_points = pyfempp.unique_rows(np.reshape(points[triangles], (3 * triangles.shape[0], 3)))

        # generate Delaunay grid object of head surface
        skin_surface = scipy.spatial.Delaunay(skin_surface_points)

        if dispy:
            import socket
            import dispy
            import sys
            import time
            dispy.MsgTimeout = 90
            dispy_schedular_ip = socket.gethostbyname(dispy_sched_host)

            #  ~/.local/bin/dispyscheduler.py on this machine
            #  ~/.local/bin/dispynode.py on any else (no

            if print_out:
                print(("Trying to connect to dispyschedular on " + dispy_sched_host))
            while True:
                try:
                    cluster = dispy.SharedJobCluster(func, port=0, scheduler_node=str(dispy_schedular_ip),
                                                     reentrant=True)  # loglevel=dispy.logger.DEBUG,
                    break
                except socket.error as e:
                    time.sleep(1)
                    sys.stdout.write('.')
                    sys.stdout.flush()
            assert cluster

        # make dummy grid
        grid_init = randomgrid(pdftype, pdfshape, limits, 1, seed=seed)

        # make initial regobj
        regobj = reg(pdftype,
                     pdfshape,
                     limits,
                     order * np.ones(dim),
                     order_max=order,
                     interaction_order=interaction_order_max,
                     grid=grid_init,
                     random_vars=random_vars)

        # determine number of coefficients
        n_c_init = regobj.poly_idx.shape[0]

        # make initial grid
        grid_init = randomgrid(pdftype, pdfshape, limits, np.ceil(1.2 * n_c_init))

        if fn_gpcobj:
            # if .yaml does exist: load from yaml
            if os.path.exists(fn_gpcobj):
                print(results_folder + ": Loading regobj from file: " + fn_gpcobj)
                regobj = load_gpcobj(fn_gpcobj)

            # if not: create regobj, save to yaml
            else:
                # regobj = reg(pdftype, pdfshape, limits, (order*np.ones(DIM)).tolist(),
                #              order_max=order, interaction_order=DIM, grid=grid_init)

                # re-initialize reg object with appropriate number of grid-points
                regobj = reg(pdftype,
                             pdfshape,
                             limits,
                             order * np.ones(dim),
                             order_max=order,
                             interaction_order=interaction_order_max,
                             grid=grid_init,
                             random_vars=random_vars)

                save_gpcobj(regobj, fn_gpcobj)

                # if dispy:
                #     cluster.close()
                # return

        # run simulations on initial grid
        print("Iteration #{} (initial grid)".format(i_iter))
        print("=============")

        for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
            # if print_out:
            # print "   Performing simulation #{}\n".format(i_grid+1)
            # if i_grid == 24:
            #     stop = 1
            # read conductivities from grid
            x = [i_grid, regobj.grid.coords[i_grid, :]]

            # evaluate function at grid point
            results_fn = func(x, *(args))

            # append result to solution matrix (RHS)
            try:
                with h5py.File(results_fn, 'r') as hdf: # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                    # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                    e = hdf['/data/potential'][:]
            except Exception:
                print("Fail on " + results_fn)
            if results is None:
                results = e.flatten()
            else:
                results = np.vstack([results, e.flatten()])
            del e


        # increase grid counter by one for next iteration (to not repeat last simulation)
        i_grid = i_grid + 1

        # perform leave one out cross validation
        regobj.LOOCV(results)

        if print_out:
            print("    -> relerror_LOOCV = {}\n".format(regobj.relerror_loocv[-1]))

        # main interations (order)
        while (regobj.relerror_loocv[-1] > eps) and order < order_end:

            i_iter = i_iter + 1
            order = order + 1

            print("Iteration #{}".format(i_iter))
            print("=============")

            # determine new possible polynomials
            poly_idx_all_new = allVL1leq(dim, order)
            poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new, axis=1) == order]
            interaction_order_current_max = np.max(poly_idx_all_new)

            # reset current interaction order before subiterations
            interaction_order_current = 1

            # subiterations (interaction orders)
            while (interaction_order_current <= interaction_order_current_max) and \
                    (interaction_order_current <= interaction_order_max) and \
                    run_subiter:
                print("   Subiteration #{}".format(interaction_order_current))
                print("   ================")

                interaction_order_list = np.sum(poly_idx_all_new > 0, axis=1)

                # filter out polynomials of interaction_order = interaction_order_count
                poly_idx_added = poly_idx_all_new[interaction_order_list == interaction_order_current, :]

                # add polynomials to gpc expansion
                regobj.enrich_polynomial_basis(poly_idx_added)

                # generate new grid-points
                # regobj.enrich_gpc_matrix_samples(1.2)

                if seed:
                    seed += 1

                n_g_old = regobj.grid.coords.shape[0]

                regobj.enrich_gpc_matrix_samples(1.2, seed=seed)

                n_g_new = regobj.grid.coords.shape[0]
                # n_g_added = n_g_new - n_g_old

                # check if coil position of new grid points are valid and do not lie inside head
                # TODO: adapt this part to 'x' 'y' 'z' 'psi' 'theta' 'phi'...
                if regobj.grid.coords.shape[1] >= 9:
                    for i in range(n_g_old, n_g_new):

                        valid_coil_position = False

                        while not valid_coil_position:
                            # get coil transformation matrix
                            coil_trans_mat = \
                                pyfempp.calc_coil_transformation_matrix(LOC_mean=positions_mean[0:3, 3],
                                                                        ORI_mean=positions_mean[0:3, 0:3],
                                                                        LOC_var=regobj.grid.coords[i, 4:7],
                                                                        ORI_var=regobj.grid.coords[i, 7:10],
                                                                        V=v)
                            # get actual coordinates of magnetic dipole
                            dipole_coords = pyfempp.get_coil_dipole_pos(coil_fn, coil_trans_mat)
                            valid_coil_position = pyfempp.check_coil_position(dipole_coords, skin_surface)

                            # replace bad sample with new one until it works (should actually never be the case)
                            if not valid_coil_position:
                                warnings.warn(results_folder +
                                              ": Invalid coil position found: " + str(regobj.grid.coords[i]))
                                regobj.replace_gpc_matrix_samples(idx=np.array(i), seed=seed)

                if do_mp:
                    # run repeated simulations
                    x = []
                    if print_out:
                        print(results_folder + \
                              "   Performing simulations #{} to {}".format(i_grid + 1,
                                                                             regobj.grid.coords.shape[0]))
                    for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
                        x.append([i_grid, regobj.grid.coords[i_grid, :]])

                    func_part = partial(func,
                                        mesh_fn=mesh_fn, tensor_fn=tensor_fn,
                                        results_folder=results_folder,
                                        coil_fn=coil_fn,
                                        POSITIONS_mean=positions_mean,
                                        V=v)
                    p = NondaemonicPool(n_cpu)
                    results_fns = np.array(p.map(func_part, x))
                    p.close()
                    p.join()

                    # append result to solution matrix (RHS)
                    for hdf5_fn in results_fns:
                        try:
                            with h5py.File(hdf5_fn, 'r') as hdf, h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                                # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                                e = hdf['/data/potential'][:]
                        except Exception:
                            print("Fail on " + hdf5_fn)
                        results = np.vstack([results, e.flatten()])
                        del e

                elif dispy:
                    # compute with dispy cluster
                    assert cluster
                    if print_out:
                        # print("Scheduler connected. Now start dispynodes anywhere in the network")
                        print("   Performing simulations #{} to {}".format(i_grid + 1,
                                                                             regobj.grid.coords.shape[0]))
                    # build job list
                    jobs = []
                    for i_grid in range(i_grid, regobj.grid.coords.shape[0]):

                        job = cluster.submit([i_grid, regobj.grid.coords[i_grid, :]], *(args))
                        job.id = i_grid
                        jobs.append(job)

                    # get results from single jobs
                    results_fns = []
                    for job in jobs:
                        # res = job()
                        results_fns.append(job())
                        if print_out:
                            # print(str(job.id) + " done in " + str(job.end_time - job.start_time))
                            # print(job.stdout)
                            if job.exception is not None:
                                print((job.exception))
                                return

                    for hdf5_fn in results_fns:
                        try:
                            with h5py.File(hdf5_fn, 'r') as hdf: # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                                # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                                e = hdf['/data/potential'][:]
                        except Exception:
                            print("Fail on " + hdf5_fn)
                        results = np.vstack([results, e.flatten()])
                        del e

                else:  # no multiprocessing
                    for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
                        if print_out:
                            print("   Performing simulation #{}".format(i_grid+1))
                        # read conductivities from grid
                        x = [i_grid, regobj.grid.coords[i_grid, :]]

                        # evaluate function at grid point
                        results_fn = func(x, *(args))

                        # append result to solution matrix (RHS)
                        try:
                            with h5py.File(results_fn, 'r') as hdf: # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                                # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                                e = hdf['/data/potential'][:]
                        except Exception:
                            print("Fail on " + results_fn)
                        results = np.vstack([results, e.flatten()])
                        del e

                # increase grid counter by one for next iteration (to not repeat last simulation)
                i_grid = i_grid + 1

                # perform leave one out cross validation
                regobj.LOOCV(results)
                if print_out:
                    print(results_folder + "    -> relerror_LOOCV = {}".format(regobj.relerror_loocv[-1]))

                if regobj.relerror_loocv[-1] < eps:
                    run_subiter = False

                # increase current interaction order
                interaction_order_current += 1

        if print_out:
            print(results_folder + "DONE ##############################################################")

        if dispy:
            try:
                cluster.close()
            except UnboundLocalError:
                pass
            
        # save gPC object
        save_gpcobj(regobj, fn_gpcobj)

        # save results of forward simulation
        np.save(os.path.splitext(fn_gpcobj)[0] + "_res", results)

    except Exception as e:
        if dispy:
            try:
                cluster.close()
            except UnboundLocalError:
                pass
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print((exc_type, fname, exc_tb.tb_lineno))
        quit()
    return  # regobj, RES


def run_reg_adaptive(random_vars, pdftype, pdfshape, limits, func, args=(),
                     order_start=0, order_end=10, eps=1E-3,
                     print_out=False):
    """
    Adaptive regression approach based on leave one out cross validation error
    estimation

    Parameters
    ----------
    random_vars : list of str
        string labels of the random variables
    pdftype : list
              Type of probability density functions of input parameters,
              i.e. ["beta", "norm",...]
    pdfshape : list of lists
               Shape parameters of probability density functions
               s1=[...] "beta": p, "norm": mean
               s2=[...] "beta": q, "norm": std
               pdfshape = [s1,s2]
    limits : list of lists
             Upper and lower bounds of random variables (only "beta")
             a=[...] "beta": lower bound, "norm": n/a define 0
             b=[...] "beta": upper bound, "norm": n/a define 0
             limits = [a,b]
    func : callable func(x,*args)
           The objective function to be minimized.
    args : tuple, optional
           Extra arguments passed to func, i.e. f(x,*args).
    order_start : int, optional
                  Initial gpc expansion order (maximum order)
    order_end : int, optional
                Maximum gpc expansion order to expand to
    eps : float, optional
          Relative mean error of leave one out cross validation
    print_out : boolean, optional
          Print output of iterations and subiterations (True/False)

    Returns
    -------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]
    """

    # initialize iterators
    eps_gpc = eps + 1
    i_grid = 0
    i_iter = 0
    interaction_order_count = 0
    interaction_order_max = -1
    DIM = len(pdftype)
    order = order_start

    while eps_gpc > eps:

        # reset sub iteration if all polynomials of present order were added to gpc expansion
        if interaction_order_count > interaction_order_max:
            interaction_order_count = 0
            i_iter = i_iter + 1
            if print_out:
                print("Iteration #{}".format(i_iter))
                print("=============")

        if i_iter == 1:
            # initialize gPC object in first iteration
            grid_init = randomgrid(pdftype, pdfshape, limits, np.ceil(1.2 * calc_Nc(order, DIM)))
            regobj = reg(pdftype, pdfshape, limits, order * np.ones(DIM), order_max=order, interaction_order=DIM,
                         grid=grid_init,
                         random_vars=random_vars)
        else:

            # generate new multi-indices if maximum gpc order increased
            if interaction_order_count == 0:
                order = order + 1
                if order > order_end:
                    return regobj
                poly_idx_all_new = allVL1leq(DIM, order)
                poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new, axis=1) == order]
                interaction_order_list = np.sum(poly_idx_all_new > 0, axis=1)
                interaction_order_max = np.max(interaction_order_list)
                interaction_order_count = 1

            if print_out:
                print("   Subiteration #{}".format(interaction_order_count))
                print("   ================")
            # filter out polynomials of interaction_order = interaction_order_count
            poly_idx_added = poly_idx_all_new[interaction_order_list == interaction_order_count, :]

            # add polynomials to gpc expansion
            regobj.enrich_polynomial_basis(poly_idx_added)

            # generate new grid-points
            regobj.enrich_gpc_matrix_samples(1.2)

            interaction_order_count = interaction_order_count + 1

            # run repeated simulations
        for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
            if print_out:
                print("   Performing simulation #{}".format(i_grid + 1))
            # read conductivities from grid
            x = regobj.grid.coords[i_grid, :]

            # evaluate function at grid point
            res = func(x, *(args))

            # append result to solution matrix (RHS)
            if i_grid == 0:
                res_complete = res
            else:
                res_complete = np.vstack([res_complete, res])

        # increase grid counter by one for next iteration (to not repeat last simulation)
        i_grid = i_grid + 1

        # perform leave one out cross validation
        eps_gpc = regobj.LOOCV(res_complete)
        if print_out:
            print("    -> relerror_LOOCV = {}".format(eps_gpc))

    return regobj, res_complete


def run_reg_adaptive2(random_vars, pdftype, pdfshape, limits, func, args=(), order_start=0, order_end=10,
                      interaction_order_max=None, eps=1E-3, print_out=False, seed=None,
                      save_res_fn=''):
    """
    Adaptive regression approach based on leave one out cross validation error
    estimation

    Parameters
    ----------
    random_vars : list of str
        string labels of the random variables
    pdftype : list
              Type of probability density functions of input parameters,
              i.e. ["beta", "norm",...]
    pdfshape : list of lists
               Shape parameters of probability density functions
               s1=[...] "beta": p, "norm": mean
               s2=[...] "beta": q, "norm": std
               pdfshape = [s1,s2]
    limits : list of lists
             Upper and lower bounds of random variables (only "beta")
             a=[...] "beta": lower bound, "norm": n/a define 0
             b=[...] "beta": upper bound, "norm": n/a define 0
             limits = [a,b]
    func : callable func(x,*args)
           The objective function to be minimized.
    args : tuple, optional
           Extra arguments passed to func, i.e. f(x,*args).
    order_start : int, optional
                  Initial gpc expansion order (maximum order)
    order_end : int, optional
                Maximum gpc expansion order to expand to
    interaction_order_max: int
        define maximum interaction order of parameters (default: all interactions)
    eps : float, optional
          Relative mean error of leave one out cross validation
    print_out : boolean, optional
          Print output of iterations and subiterations (True/False)
    seed : int, optional
        Set np.random.seed(seed) in random_grid()
    save_res_fn : string, optional
          If provided, results are saved in hdf5 file
          
    Returns
    -------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]
    """

    # initialize iterators
    matrix_ratio = 1.5
    i_grid = 0
    i_iter = 0
    run_subiter = True
    interaction_order_current = 0
    func_time = None
    read_from_file = None
    dim = len(pdftype)
    if not interaction_order_max:
        interaction_order_max = dim
    order = order_start
    res_complete = None
    if save_res_fn and save_res_fn[-5:] != '.hdf5':
        save_res_fn += '.hdf5'

    # make dummy grid
    grid_init = randomgrid(pdftype, pdfshape, limits, 1, seed=seed)

    # make initial regobj
    regobj = reg(pdftype,
                 pdfshape,
                 limits,
                 order * np.ones(dim),
                 order_max=order,
                 interaction_order=interaction_order_max,
                 grid=grid_init)

    # determine number of coefficients
    n_c_init = regobj.poly_idx.shape[0]

    # make initial grid
    grid_init = randomgrid(pdftype, pdfshape, limits, np.ceil(1.2*n_c_init), seed=seed)

    # re-initialize reg object with appropriate number of grid-points
    regobj = reg(pdftype,
                 pdfshape,
                 limits,
                 order * np.ones(dim),
                 order_max=order,
                 interaction_order=interaction_order_max,
                 grid=grid_init,
                 random_vars=random_vars)

    # run simulations on initial grid
    print("Iteration #{} (initial grid)".format(i_iter))
    print("=============")

    for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
        if print_out:
            print("    Performing simulation #{}".format(i_grid + 1))

        if save_res_fn and os.path.exists(save_res_fn):
            try:
                with h5py.File(save_res_fn, 'a') as f:
                    # get datset
                    ds = f['res']
                    # ... then read res from file
                    if i_grid == 0:
                        res_complete = ds[i_grid, :]
                    else:
                        res_complete = np.vstack([res_complete, ds[i_grid, :]])

                    print("Reading data row from " + save_res_fn)
                    # if read from file, skip to next i_grid
                    continue

            except (KeyError, ValueError):
                pass

        # read conductivities from grid
        x = regobj.grid.coords[i_grid, :]

        # evaluate function at grid point
        start_time = time.time()
        res = func(x, *(args))
        print(('        function evaluation: ' + str(time.time() - start_time) + 'sec'))
        # append result to solution matrix (RHS)
        if i_grid == 0:
            res_complete = res
            if save_res_fn:
                if os.path.exists(save_res_fn):
                    raise OSError('Given results filename exists.')
                with h5py.File(save_res_fn, 'a') as f:
                    f.create_dataset('res', data=res[np.newaxis, :], maxshape=(None, len(res)))
        else:
            res_complete = np.vstack([res_complete, res])
            if save_res_fn:
                with h5py.File(save_res_fn, 'a') as f:
                    ds = f['res']
                    ds.resize(ds.shape[0] + 1, axis=0)
                    ds[ds.shape[0]-1, :] = res[np.newaxis, :]

    # increase grid counter by one for next iteration (to not repeat last simulation)
    i_grid = i_grid + 1

    # perform leave one out cross validation for elements that are never nan
    non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
    regobj.nan_elm = np.where(np.any(np.isnan(res_complete), axis=0))[0]
    n_nan = np.sum(np.isnan(res_complete))

    # how many nan-per element? 1 -> all gridpoints are NAN for all elements
    if n_nan > 0:
        nan_ratio_per_elm = np.round(float(n_nan) / len(regobj.nan_elm) / res_complete.shape[0], 3)
        if print_out:
            print(("Number of NaN elms: {} from {}, ratio per elm: {}".format(len(regobj.nan_elm),
                                                                     res_complete.shape[1],
                                                                     nan_ratio_per_elm)))
    regobj.LOOCV(res_complete[:, non_nan_mask])

    if print_out:
        print(("    -> relerror_LOOCV = {}").format(regobj.relerror_loocv[-1]))

    # main interations (order)
    while (regobj.relerror_loocv[-1] > eps) and order < order_end:

        i_iter = i_iter + 1
        order = order + 1

        print("Iteration #{}".format(i_iter))
        print("=============")

        # determine new possible polynomials
        poly_idx_all_new = allVL1leq(dim, order)
        poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new, axis=1) == order]
        interaction_order_current_max = np.max(poly_idx_all_new)

        # reset current interaction order before subiterations
        interaction_order_current = 1

        # subiterations (interaction orders)
        while (interaction_order_current <= interaction_order_current_max) and \
                (interaction_order_current <= interaction_order_max) and \
                run_subiter:

            print("   Subiteration #{}".format(interaction_order_current))
            print("   ================")

            interaction_order_list = np.sum(poly_idx_all_new > 0, axis=1)

            # filter out polynomials of interaction_order = interaction_order_count
            poly_idx_added = poly_idx_all_new[interaction_order_list == interaction_order_current, :]

            # add polynomials to gpc expansion
            regobj.enrich_polynomial_basis(poly_idx_added)

            if seed:
                seed += 1
            # generate new grid-points
            regobj.enrich_gpc_matrix_samples(matrix_ratio, seed=seed)

            # run simulations
            print(("   " + str(i_grid) + " to " + str(regobj.grid.coords.shape[0])))
            for i_grid in range(i_grid, regobj.grid.coords.shape[0]):
                if print_out:
                    if func_time:
                        more_text = "Function evaluation took: " + func_time + "s"
                    elif read_from_file:
                        more_text = "Read data row from " + read_from_file
                    else:
                        more_text = None
                    # print "{}, {}   Performing simulation #{:4d} of {:4d}".format(i_iter,
                    #                                                       interaction_order_current,
                    #                                                       i_grid + 1,
                    #                                                       regobj.grid.coords.shape[0])
                    fancy_bar("It/Subit: {}/{} Performing simulation".format(i_iter,
                                                                             interaction_order_current),
                              i_grid + 1,
                              regobj.grid.coords.shape[0],
                              more_text)

                # try to read from file
                if save_res_fn:
                    try:
                        with h5py.File(save_res_fn, 'a') as f:
                            # get datset
                            ds = f['res']
                            # ... then read res from file
                            if i_grid == 0:
                                res_complete = ds[i_grid, :]
                            else:
                                res_complete = np.vstack([res_complete, ds[i_grid, :]])

                            # print "Read data row from " + save_res_fn
                            read_from_file = save_res_fn
                            func_time = None
                            # if read from file, skip to next i_grid
                            continue

                    except (KeyError, ValueError):
                        pass

                # code below is only exectued if save_res_fn does not contain results for i_grid

                # read conductivities from grid
                x = regobj.grid.coords[i_grid, :]

                # evaluate function at grid point
                start_time = time.time()
                res = func(x, *(args))
                # print('   function evaluation: ' + str(time.time() - start) + ' sec\n')
                func_time = str(time.time() - start_time)
                read_from_file = None
                # append result to solution matrix (RHS)
                if i_grid == 0:
                    res_complete = res
                else:
                    res_complete = np.vstack([res_complete, res])

                if save_res_fn:
                    with h5py.File(save_res_fn, 'a') as f:
                        ds = f['res']
                        ds.resize(ds.shape[0] + 1, axis=0)
                        ds[ds.shape[0] - 1, :] = res[np.newaxis, :]

            # increase grid counter by one for next iteration (to not repeat last simulation)
            i_grid = i_grid + 1
            func_time = None
            non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
            regobj.nan_elm = np.where(np.any(np.isnan(res_complete), axis=0))[0]
            n_nan = np.sum(np.isnan(res_complete))

            # how many nan-per element? 1 -> all gridpoints are NAN for all elements
            if n_nan > 0:
                nan_ratio_per_elm = np.round(float(n_nan) / len(regobj.nan_elm) / res_complete.shape[0], 3)
                if print_out:
                    print("Number of NaN elms: {} from {}, ratio per elm: {}".format(len(regobj.nan_elm),
                                                                             res_complete.shape[1],
                                                                             nan_ratio_per_elm))

            regobj.LOOCV(res_complete[:, non_nan_mask])

            if print_out:
                print("    -> relerror_LOOCV = {}".format(regobj.relerror_loocv[-1]))
            if regobj.relerror_loocv[-1] < eps:
                run_subiter = False

            # save reg object and results for this subiteration
            if save_res_fn:
                fn_folder, fn_file = os.path.split(save_res_fn)
                fn_file = os.path.splitext(fn_file)[0]
                fn = os.path.join(fn_folder,
                                  fn_file + '_' + str(i_iter).zfill(2) + "_" + str(interaction_order_current).zfill(2))
                save_gpcobj(regobj, fn + '_gpc.pkl')
                np.save(fn + '_results', res_complete)

            # increase current interaction order
            interaction_order_current = interaction_order_current + 1

    return regobj, res_complete

#%%############################################################################
# gpc object class
###############################################################################

class gpc:
    def __init__(self):
        """ Initialize gpc class """
        self.random_vars = []
        self.pdfshape = []
        self.pdftype = []
        self.poly = []
        self.poly_idx = []
        self.DIM = []
        self.poly_norm = []
        self.poly_norm_basis = []
        self.order = []
        self.limits = []
        self.N_poly = []
        self.mean_random_vars = []

    def setup_polynomial_basis(self):
        """ Setup polynomial basis functions for a maximum order expansion """
        #print 'Setup polynomial basis functions ...'
        # Setup list of polynomials and their coefficients up to the desired order
        #
        #  poly    |     DIM_1     DIM_2    ...    DIM_M
        # -----------------------------------------------
        # Poly_1   |  [coeffs]  [coeffs]   ...  [coeffs]          
        # Poly_2   |  [coeffs]  [coeffs]   ...  [coeffs]
        #   ...    |  [coeffs]  [coeffs]   ...   [0]
        #   ...    |  [coeffs]  [coeffs]   ...   [0]
        #   ...    |  [coeffs]  [coeffs]   ...   ...
        # Poly_No  |   [0]      [coeffs]   ...   [0]
        #
        # size: [Max individual order x DIM]   (includes polynomials also not used)
        
        Nmax = int(np.max(self.order))
        
        # 2D list of polynomials (lookup)
        self.poly      = [[0 for x in range(self.DIM)] for x in range(Nmax+1)]
        
        # 2D array of polynomial normalization factors (lookup)
        # [Nmax+1 x DIM]
        self.poly_norm = np.zeros([Nmax+1,self.DIM])
        
        for i_DIM in range(self.DIM):
            for i_order in range(Nmax+1):
                if self.pdftype[i_DIM] == "beta": 
                    p = self.pdfshape[0][i_DIM] # beta-distr: alpha=p /// jacobi-poly: alpha=q-1  !!!
                    q = self.pdfshape[1][i_DIM] # beta-distr: beta=q  /// jacobi-poly: beta=p-1   !!!
                        
                    # determine polynomial normalization factor
                    beta_norm = (scipy.special.gamma(q)*scipy.special.gamma(p)/scipy.special.gamma(p+q)*(2.0)**(p+q-1))**(-1)
                    jacobi_norm = 2**(p+q-1) / (2.0*i_order+p+q-1)*scipy.special.gamma(i_order+p)*scipy.special.gamma(i_order+q) / (scipy.special.gamma(i_order+p+q-1)*scipy.special.factorial(i_order))
                    self.poly_norm[i_order,i_DIM] = (jacobi_norm * beta_norm)
                        
                    # add entry to polynomial lookup table
                    self.poly[i_order][i_DIM] = scipy.special.jacobi(i_order, q-1, p-1, monic=0)/np.sqrt(self.poly_norm[i_order,i_DIM]) 
                    
#==============================================================================
#                     alpha = self.pdfshape[0][i_DIM]-1 # here: alpha' = p-1
#                     beta = self.pdfshape[1][i_DIM]-1  # here: beta' = q-1
#                         
#                     # determine polynomial normalization factor
#                     beta_norm = (scipy.special.gamma(beta+1)*scipy.special.gamma(alpha+1)/scipy.special.gamma(beta+1+alpha+1)*(2.0)**(beta+1+alpha+1-1))**(-1)
#                     jacobi_norm = 2**(alpha+beta+1) / (2.0*i_order+alpha+beta+1)*scipy.special.gamma(i_order+alpha+1)*scipy.special.gamma(i_order+beta+1) / (scipy.special.gamma(i_order+alpha+beta+1)*scipy.special.factorial(i_order))
#                     self.poly_norm[i_order,i_DIM] = (jacobi_norm * beta_norm)
#                         
#                     # add entry to polynomial lookup table
#                     self.poly[i_order][i_DIM] = scipy.special.jacobi(i_order, beta, alpha, monic=0) # ! alpha beta changed definition in scipy!                  
#==============================================================================
                    
                if self.pdftype[i_DIM] == "normal" or self.pdftype[i_DIM] == "norm":
                        
                    # determine polynomial normalization factor
                    hermite_norm = scipy.special.factorial(i_order)
                    self.poly_norm[i_order,i_DIM] = hermite_norm
                        
                    # add entry to polynomial lookup table
                    self.poly[i_order][i_DIM] = scipy.special.hermitenorm(i_order, monic=0)/np.sqrt(self.poly_norm[i_order,i_DIM]) 
                        
        
        # Determine 2D multi-index array (order) of basis functions w.r.t. 2D array
        # of polynomials self.poly
        #
        # poly_idx |     DIM_1       DIM_2       ...    DIM_M
        # -------------------------------------------------------
        # basis_1  |  [order_D1]  [order_D2]     ...  [order_DM]    
        # basis_2  |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        # basis_Nb |  [order_D1]  [order_D2]     ...  [order_DM]
        #
        # size: [No. of basis functions x DIM]
        
        # generate multi-index list up to maximum order
        if self.DIM == 1:
            self.poly_idx = np.array([np.linspace(0,self.order_max,self.order_max+1)]).astype(int).transpose()
        else:
            self.poly_idx = allVL1leq(self.DIM, self.order_max)
            
        
        for i_DIM in range(self.DIM):
            # add multi-indexes to list when not yet included
            if self.order[i_DIM] > self.order_max:
                poly_add_dim = np.linspace(self.order_max+1, self.order[i_DIM], self.order[i_DIM]-(self.order_max+1) + 1)
                poly_add_all = np.zeros([poly_add_dim.shape[0],self.DIM])
                poly_add_all[:,i_DIM] = poly_add_dim               
                self.poly_idx = np.vstack([self.poly_idx,poly_add_all.astype(int)])
            # delete multi-indexes from list when they exceed individual max order of parameter     
            elif self.order[i_DIM] < self.order_max:    
                self.poly_idx = self.poly_idx[self.poly_idx[:,i_DIM]<=self.order[i_DIM],:]
                
        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.DIM:        
            self.poly_idx = self.poly_idx[np.sum(self.poly_idx>0,axis=1)<=self.interaction_order,:]        
        
        self.N_poly = self.poly_idx.shape[0]
           
#==============================================================================
#         x1 = [np.array(range(self.order[i]+1)) for i in range(self.DIM)]
#         self.poly_idx = []
#         
#         for element in itertools.product(*x1):
#             if np.sum(element) <= self.maxorder:
#                 self.poly_idx.append(element)
#                 
#         self.poly_idx = np.array(self.poly_idx) 
#==============================================================================
        
#==============================================================================
#         x1 = [np.array(range(self.order[i]+1)) for i in range(self.DIM)]
#         order_idx    = misc.combvec(x1)
#         
#         # filter for individual maximum expansion order
#         for i in range(self.DIM):
#             order_idx = order_idx[order_idx[:,i] <= self.order[i]]
# 
#         # filter for total maximum order
#         self.poly_idx = order_idx[np.sum(order_idx,axis = 1) <= self.order_max]
#==============================================================================
        
        # construct array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [Npolybasis x 1]
        self.poly_norm_basis = np.ones([self.poly_idx.shape[0],1])
        for i_poly in range(self.poly_idx.shape[0]):
            for i_DIM in range(self.DIM):
                self.poly_norm_basis[i_poly] *= self.poly_norm[self.poly_idx[i_poly, i_DIM], i_DIM]
    
    def enrich_polynomial_basis(self, poly_idx_added):
        """ Enrich polynomial basis functions and add new columns to gpc matrix

        enrich_polynomial_basis(poly_idx_added)

        Parameters:
        ----------------------------------
        poly_idx_added: np.array of int [N_poly_added x DIM]
            array of added polynomials (order)
        """

        # determine if polynomials in poly_idx_added are already present in self.poly_idx if so, delete them
        poly_idx_tmp = []
        for new_row in poly_idx_added:
            not_in_poly_idx = True
            for row in self.poly_idx:
                if np.allclose(row, new_row):
                    not_in_poly_idx = False
            if not_in_poly_idx:
                poly_idx_tmp.append(new_row)
        
        # if all polynomials are already present end routine
        if len(poly_idx_tmp) == 0:        
            return
        else:            
            poly_idx_added = np.vstack(poly_idx_tmp)
        
        # determine highest order added        
        order_max_added = np.max(np.max(poly_idx_added))
        
        # get current maximum order 
        order_max_current = len(self.poly)-1
        
        # Append list of polynomials and their coefficients up to the desired order
        #
        #  poly    |     DIM_1     DIM_2    ...    DIM_M
        # -----------------------------------------------
        # Poly_1   |  [coeffs_old]  [coeffs_old]   ...  [coeffs_old]          
        # Poly_2   |  [coeffs_old]  [coeffs_old]   ...  [coeffs_old]
        #   ...    |  [coeffs_old]  [coeffs_old]   ...  [coeffs_old]
        #   ...    |  [coeffs_old]  [coeffs_old]   ...  [coeffs_old]
        #   ...    |      ...           ...        ...      ...
        # Poly_No  |  [coeffs_new]  [coeffs_new]   ...  [coeffs_new]
        #
        # size: [Max order x DIM]
        
        # preallocate new rows to polynomial lists
        for i in range(order_max_added-order_max_current):
            self.poly.append([0 for x in range(self.DIM)])
            self.poly_norm = np.vstack([self.poly_norm, np.zeros(self.DIM)])
                
        for i_DIM in range(self.DIM):
            for i_order in range(order_max_current+1,order_max_added+1):
                if self.pdftype[i_DIM] == "beta":
                    p = self.pdfshape[0][i_DIM]
                    q = self.pdfshape[1][i_DIM]
                        
                    # determine polynomial normalization factor
                    beta_norm = (scipy.special.gamma(p)*scipy.special.gamma(q)/
                                 scipy.special.gamma(p+q)*(2.0)**(p+q-1))**(-1) # 1/B(p,q)
                    jacobi_norm = 2**(p+q-1) / \
                                  (2.0*i_order+q+p-1)*scipy.special.gamma(i_order+q)*scipy.special.gamma(i_order+p) / \
                                  (scipy.special.gamma(i_order+q+p-1)*scipy.special.factorial(i_order))
                    self.poly_norm[i_order,i_DIM] = (jacobi_norm * beta_norm)
                        
                    # add entry to polynomial lookup table
                    self.poly[i_order][i_DIM] = scipy.special.jacobi(i_order, q-1, p-1, monic=0)/\
                                                np.sqrt(self.poly_norm[i_order,i_DIM])
                    # ! beta = p-1 and alpha=q-1 (consider definition in scipy.special.jacobi !!)
                    
                if self.pdftype[i_DIM] == "normal" or self.pdftype[i_DIM] == "norm":
                        
                    # determine polynomial normalization factor
                    hermite_norm = scipy.special.factorial(i_order)
                    self.poly_norm[i_order, i_DIM] = hermite_norm
                        
                    # add entry to polynomial lookup table
                    self.poly[i_order][i_DIM] = scipy.special.hermitenorm(i_order, monic=0)/np.sqrt(self.poly_norm[i_order,i_DIM]) 
        
        # append new multi-indexes to old poly_idx array
        self.poly_idx = np.vstack([self.poly_idx, poly_idx_added])
        #self.poly_idx = unique_rows(self.poly_idx)
        self.N_poly = self.poly_idx.shape[0]
        
        # extend array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [Npolybasis x 1]
        N_poly_new = poly_idx_added.shape[0]
        poly_norm_basis_new = np.ones([N_poly_new,1])
        for i_poly in range(N_poly_new):
            for i_DIM in range(self.DIM):
                poly_norm_basis_new[i_poly] *= self.poly_norm[poly_idx_added[i_poly, i_DIM], i_DIM]
        
        self.poly_norm_basis = np.vstack([self.poly_norm_basis, poly_norm_basis_new])
        
        # append new columns to gpc matrix [N_grid x N_poly_new]
        A_new_columns = np.zeros([self.N_grid, N_poly_new])
        for i_poly_new in range(N_poly_new):
            A1 = np.ones(self.N_grid)
            for i_DIM in range(self.DIM):
                A1 *= self.poly[poly_idx_added[i_poly_new][i_DIM]][i_DIM](self.grid.coords_norm[:, i_DIM])
            A_new_columns[:, i_poly_new] = A1
        
        self.A = np.hstack([self.A, A_new_columns])
        self.Ainv = np.linalg.pinv(self.A)
    
    def enrich_gpc_matrix_samples(self, N_samples_N_poly_ratio, seed=[]):
        """ Add sample points according to input pdfs to grid and enrich the gpc matrix such that the ratio of
        rows/columns is N_samples_N_poly_ratio

        enrich_gpc_matrix_samples(N_samples_N_poly_ratio, seed=[]):

        Parameters:
        ----------------------------------
        N_samples_N_poly_ratio: float
            Ratio between number of samples and number of polynomials the matrix will be enriched until
        seed (optional): float
            Random seeding point
        """
        
        # Number of new grid points
        N_grid_new = int(np.ceil(N_samples_N_poly_ratio * self.A.shape[1] - self.A.shape[0]))
        
        if N_grid_new > 0:
            # Generate new grid points
            newgridpoints = randomgrid(self.pdftype, self.pdfshape, self.limits, N_grid_new, seed=seed)
            
            # append points to existing grid
            self.grid.coords = np.vstack([self.grid.coords, newgridpoints.coords])
            self.grid.coords_norm = np.vstack([self.grid.coords_norm, newgridpoints.coords_norm])
            self.N_grid = self.grid.coords.shape[0]
            
            # determine new row of gpc matrix
            a = np.zeros([N_grid_new, self.N_poly])
            for i_poly in range(self.N_poly):
                a1 = np.ones(N_grid_new)
                for i_DIM in range(self.DIM):
                    a1 *= self.poly[self.poly_idx[i_poly][i_DIM]][i_DIM](newgridpoints.coords_norm[:, i_DIM])
                a[:, i_poly] = a1
            
            # append new row to gpc matrix    
            self.A = np.vstack([self.A, a])
            
            # invert gpc matrix Ainv [N_basis x N_grid]
            self.Ainv = np.linalg.pinv(self.A)

    def replace_gpc_matrix_samples(self, idx, seed):
        """ Replace distinct sample points from the gpc matrix

        replace_gpc_matrix_samples(idx, seed=seed)

        Parameters:
        ----------------------------------
        idx: np.array of int
            array of grid indices of obj.grid.coords[idx,:] which are going to be replaced
            (rows of gPC matrix will be replaced by new ones)
        seed (optional): float
            Random seeding point
        """

        # Generate new grid points
        newgridpoints = randomgrid(self.pdftype, self.pdfshape, self.limits, idx.size, seed=seed)

        # append points to existing grid
        self.grid.coords[idx, :] = newgridpoints.coords
        self.grid.coords_norm[idx, :] = newgridpoints.coords_norm
        self.N_grid = self.grid.coords.shape[0]

        # determine new row of gpc matrix
        a = np.zeros([idx.size, self.N_poly])
        for i_poly in range(self.N_poly):
            a1 = np.ones(idx.size)
            for i_DIM in range(self.DIM):
                a1 *= self.poly[self.poly_idx[i_poly][i_DIM]][i_DIM](newgridpoints.coords_norm[:, i_DIM])
            a[:, i_poly] = a1

        # append new row to gpc matrix
        self.A[idx,:] = a

        # invert gpc matrix Ainv [N_basis x N_grid]
        self.Ainv = np.linalg.pinv(self.A)

    def construct_gpc_matrix(self):
        """ Construct the gpc matrix self.A [N_grid x N_poly] and invert it using the Moore Penrose pseudo inverse self.Ainv """

        #print 'Constructing gPC matrix ...'
        self.A = np.zeros([self.N_grid,self.N_poly])
        
        for i_poly in range(self.N_poly):
            A1 = np.ones(self.N_grid)
            for i_DIM in range(self.DIM):
                A1 *= self.poly[self.poly_idx[i_poly][i_DIM]][i_DIM](self.grid.coords_norm[:,i_DIM])
            self.A[:,i_poly] = A1
        
        # invert gpc matrix Ainv [N_basis x N_grid]
        self.Ainv  = np.linalg.pinv(self.A)    

#%%############################################################################
# Postprocessing methods
###############################################################################
    def mean(self,coeffs):
        """ Calculate the expected value

        mean = mean(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        mean: np.array of float [1 x N_out]
            mean
        """

        mean = coeffs[0,:]
        mean = mean[np.newaxis,:]
        return mean
        
    def std(self, coeffs):
        """ Calculate the standard deviation

        std = std(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        std: np.array of float [1 x N_out]
            standard deviation
        """

        # return np.sqrt(np.sum(np.multiply(np.square(self.coeffs[1:,:]),self.poly_norm_basis[1:,:]),axis=0))
        std = np.sqrt(np.sum(np.square(coeffs[1:,:]),axis=0))
        std = std[np.newaxis,:]
        return std
        
    def MC_sampling(self, coeffs, N_samples, output_idx=[]):
        """ Randomly sample the gpc expansion to determine output pdfs in specific points

        xi, y = MC_sampling(coeffs, N_samples, output_idx=[])

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        N_samples: int
            number of random samples drawn from the respective input pdfs
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
        xi: np.array of float [N_samples x DIM]
            generated samples in normalized coordinates
        y: np.array of float [N_samples x N_out]
            gpc solutions
        """

        self.N_out = coeffs.shape[1]
             
        # if output index list is not provided, sample all gpc outputs
        if not output_idx:
            output_idx = np.linspace(0, self.N_out-1, self.N_out)
            output_idx = output_idx[np.newaxis,:]
            
        np.random.seed()        
        
        # generate random samples for each random input variable [N_samples x DIM]
        xi = np.zeros([N_samples, self.DIM])
        for i_DIM in range(self.DIM):
            if self.pdftype[i_DIM] == "beta":
                xi[:, i_DIM] = (np.random.beta(self.pdfshape[0][i_DIM],
                                               self.pdfshape[1][i_DIM], [N_samples, 1])*2.0 - 1)[:, 0]
            if self.pdftype[i_DIM] == "norm" or self.pdftype[i_DIM] == "normal":
                xi[:, i_DIM] = (np.random.normal(0, 1, [N_samples, 1]))[:, 0]
        
        y = self.evaluate(coeffs, xi, output_idx)
        return xi, y

    def evaluate_cpu(self, coeffs, xi, output_idx):
        """ Calculate gpc approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]) on
            the cpu

        y = evaluate(self, coeffs, xi, output_idx)

        example: y = evaluate( [[xi_1_p1 ... xi_DIM_p1] ,
                                [xi_1_p2 ... xi_DIM_p2]],
                                np.array([[0,5,13]])    )

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x DIM]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
        y: np.array of float [N_xi x N_out]
            gpc approximation at normalized coordinates xi
        """

        if len(xi.shape) == 1:
            xi = xi[:, np.newaxis]

        self.N_out = coeffs.shape[1]
        self.N_poly = self.poly_idx.shape[0]

        # if point index list is not provided, evaluate over all points
        if np.array(output_idx).size == 0:
            output_idx = np.linspace(0, self.N_out - 1, self.N_out)
            output_idx = output_idx[np.newaxis, :].astype(int)

        if np.array(output_idx).ndim == 1:
            output_idx = output_idx[np.newaxis, :]

        N_out_eval = output_idx.shape[1]
        N_x = xi.shape[0]

        y = np.zeros([N_x, N_out_eval])
        for i_poly in range(self.N_poly):
            A1 = np.ones(N_x)
            for i_DIM in range(self.DIM):
                A1 *= self.poly[self.poly_idx[i_poly][i_DIM]][i_DIM](xi[:, i_DIM])
            y += np.outer(A1, coeffs[i_poly, output_idx])
        return y

    def evaluate_gpu(self, coeffs, xi, output_idx):
        """ Calculate gpc approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]) on
            the gpu

        y = evaluate(self, coeffs, xi, output_idx)

        example: y = evaluate( [[xi_1_p1 ... xi_DIM_p1] ,
                                [xi_1_p2 ... xi_DIM_p2]],
                                np.array([[0,5,13]])    )

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x DIM]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
        y: np.array of float [N_xi x N_out]
            gpc approximation at normalized coordinates xi
        """
        # FIXME: is output_idx needed?
        # initialize matrices # FIXME: Save poly_idx as np.int32
        polynomial_index = self.poly_idx.astype(np.int32)
        y = np.zeros([xi.shape[0], coeffs.shape[1]])

        # transform list of lists of polynom objects into np.ndarray # FIXME: save polynom objects as np.ndarray
        number_of_variables = len(self.poly[0])
        highest_degree = len(self.poly)
        number_of_polynomial_coeffs = number_of_variables * (highest_degree + 1) * (highest_degree + 2) / 2
        polynomial_coeffs = np.empty([number_of_polynomial_coeffs])
        for degree in range(highest_degree):
            degree_offset = number_of_variables * degree * (degree + 1) / 2
            single_degree_coeffs = np.empty([degree + 1, number_of_variables])
            for var in range(number_of_variables):
                single_degree_coeffs[:, var] = np.flipud(self.poly[degree][var].c)
            polynomial_coeffs[degree_offset:degree_offset + single_degree_coeffs.size] = \
                single_degree_coeffs.flatten(order='C')

        # handle pointer
        polynomial_coeffs_pointer = polynomial_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        polynomial_index_pointer = polynomial_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        xi_pointer = xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sim_result_pointer = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sim_coeffs_pointer = coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        number_of_xi_size_t = ctypes.c_size_t(xi.shape[0])
        number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
        number_of_psi_size_t = ctypes.c_size_t(coeffs.shape[0])
        highest_degree_size_t = ctypes.c_size_t(highest_degree)
        number_of_result_vectors_size_t = ctypes.c_size_t(coeffs.shape[1])

        # handle shared object
        dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'pce.so'), mode=ctypes.RTLD_GLOBAL)
        cuda_pce = dll.polynomial_chaos_matrix
        cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
                            [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_size_t] * 5

        # evaluate CUDA implementation
        cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, sim_result_pointer,
                 sim_coeffs_pointer, number_of_psi_size_t, number_of_result_vectors_size_t, number_of_variables_size_t,
                 highest_degree_size_t, number_of_xi_size_t)
        return y

    def evaluate(self, coeffs, xi, output_idx=[], cpu=True):
        """ WRAPPER

        Wrapper function to calculate gpc approximation in points with output_idx and normalized parameters xi
        (interval: [-1, 1])


        y = evaluate(self, coeffs, xi, output_idx)

        example: y = evaluate( [[xi_1_p1 ... xi_DIM_p1] ,
                                [xi_1_p2 ... xi_DIM_p2]],
                                np.array([[0,5,13]])    )

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x DIM]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)
        cpu (optional): bool
            Choice if the matrices should be processed on the CPU or GPU (Default: CPU)

        Returns:
        ----------------------------------
        y: np.array of float [N_xi x N_out]
            gpc approximation at normalized coordinates xi
        """
        if cpu:
            return self.evaluate_cpu(coeffs=coeffs, xi=xi, output_idx=output_idx)
        else:
            return self.evaluate_gpu(coeffs=coeffs, xi=xi, output_idx=output_idx)

    def sobol(self, coeffs, eval=False, fn_plot=None, verbose=True):
        """ Determine the available sobol indices and evaluate results (optional)

        sobol, sobol_idx = sobol(self, coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        sobol: np.array of float [N_sobol x N_out]
            Not normalized sobol_indices
        sobol_idx: list of np.array of int [N_sobol x DIM]
            List containing the parameter combinations in rows of sobol
        sobol_rel_order_mean: nparray of float
            Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,)
            over all output quantities
        sobol_rel_order_std: nparray of float
            Standard deviation of the proportion of the Sobol indices of the different order to the total variance
            (1st, 2nd, etc..,) over all output quantities
        sobol_rel_1st_order_mean: nparray of float
            Average proportion of the random variables of the 1st order Sobol indices to the total variance over all
            output quantities
        sobol_rel_1st_order_std: nparray of float
            Standard deviation of the proportion of the random variables of the 1st order Sobol indices to the total
            variance over all output quantities
        """

        if verbose:
            print("Determining Sobol indices")

        N_sobol_theoretical = 2**self.DIM - 1
        N_coeffs = coeffs.shape[0]
        
        if N_coeffs == 1:
            raise Exception('Number of coefficients is 1 ... no sobol indices to calculate ...')
            
        # Generate boolean matrix of all basis functions where order > 0 = True
        # size: [N_coeffs x DIM] 
        sobol_mask = self.poly_idx != 0
        
        # look for unique combinations (i.e. available sobol combinations)
        # size: [N_sobol x DIM]
        sobol_idx_bool = unique_rows(sobol_mask)
        
        # delete the first row where all polys are order 0 (no sensitivity)
        sobol_idx_bool = np.delete(sobol_idx_bool,[0],axis=0)
        N_sobol_available = sobol_idx_bool.shape[0] 
        
        # check which basis functions contribute to which sobol coefficient set 
        # True for specific coeffs if it contributes to sobol coefficient
        # size: [N_coeffs x N_sobol]
        sobol_poly_idx = np.zeros([N_coeffs,N_sobol_available])
        for i_sobol in range(N_sobol_available):
            sobol_poly_idx[:,i_sobol] =  np.all(sobol_mask == sobol_idx_bool[i_sobol], axis=1)
            
        # calculate sobol coefficients matrix by summing up the individual
        # contributions to the respective sobol coefficients
        # size [N_sobol x N_points]    
        sobol = np.zeros([N_sobol_available,coeffs.shape[1]])        
        for i_sobol in range(N_sobol_available):
            sobol[i_sobol,:] = np.sum(np.square(coeffs[sobol_poly_idx[:,i_sobol]==1,:]),axis=0)
            # not normalized polynomials:             
            # sobol[i_sobol,:] = np.sum(np.multiply(np.square(coeffs[sobol_poly_idx[:,i_sobol]==1,:]),self.poly_norm_basis[sobol_poly_idx[:,i_sobol]==1,:]),axis=0)  
           
        # sort sobol coefficients in descending order (w.r.t. first output only ...)
        idx_sort_descend_1st = np.argsort(sobol[:,0],axis=0)[::-1]
        sobol = sobol[idx_sort_descend_1st,:]
        sobol_idx_bool = sobol_idx_bool[idx_sort_descend_1st]
        
        # get list of sobol indices
        sobol_idx = [0 for x in range(sobol_idx_bool.shape[0])]
        for i_sobol in range(sobol_idx_bool.shape[0]):      
            sobol_idx[i_sobol] = np.array([i for i, x in enumerate(sobol_idx_bool[i_sobol,:]) if x])

        # evaluate proportion of the sobol indices to the total variance
        if eval:
            order_max = np.max(np.sum(sobol_idx_bool, axis=1))

            # total variance
            var = np.sum(sobol, axis=0).flatten()

            # get NaN values
            not_nan_mask = np.logical_not(np.isnan(var))

            sobol_rel_order_mean = []
            sobol_rel_order_std = []
            sobol_rel_1st_order_mean = []
            sobol_rel_1st_order_std = []
            str_out = []

            # get maximum length of random_vars label
            max_len = max([len(self.random_vars[i]) for i in range(len(self.random_vars))])

            for i in range(order_max):
                # extract sobol coefficients of order i
                sobol_extracted, sobol_extracted_idx = extract_sobol_order(sobol, sobol_idx, i+1)

                sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten())
                                            / np.sum(var[not_nan_mask]))
                sobol_rel_order_std.append(0)

                # # determine ratio to total variance
                # sobol_rel = np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten() / var[not_nan_mask]
                #
                # # determine mean and std over all output quantities
                # sobol_rel_order_mean.append(np.mean(sobol_rel))
                # sobol_rel_order_std.append(np.std(sobol_rel))

                if verbose:
                    print("\tRatio: Sobol indices order {} / total variance: {:.4f}".format(i+1,
                                                                                            sobol_rel_order_mean[i]))

                # for first order indices, determine ratios of all random variables
                if i == 0:
                    sobol_extracted_idx_1st = copy.deepcopy(sobol_extracted_idx)
                    for j in range(sobol_extracted.shape[0]):

                        sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                        / np.sum(var[not_nan_mask]))
                        sobol_rel_1st_order_std.append(0)

                        # sobol_rel_1st = sobol_extracted[j, not_nan_mask].flatten() / var[not_nan_mask]
                        #
                        # sobol_rel_1st_order_mean.append(np.mean(sobol_rel_1st))
                        # sobol_rel_1st_order_std.append(np.std(sobol_rel_1st))

                        str_out.append("\t{}{}: {:.4f}".format((max_len -
                                                                len(self.random_vars[sobol_extracted_idx_1st[j]]))*' ',
                                                                self.random_vars[sobol_extracted_idx_1st[j]],
                                                                sobol_rel_1st_order_mean[j]))

            sobol_rel_order_mean = np.array(sobol_rel_order_mean)
            sobol_rel_1st_order_mean = np.array(sobol_rel_1st_order_mean)

            # print output of 1st order Sobol indice ratios of parameters
            if verbose:
                for j in range(len(str_out)):
                    print(str_out[j])

            # write logfile
            if fn_plot:
                log = open(os.path.splitext(fn_plot)[0] + '.txt', 'w')
                log.write("Sobol indices:\n")
                log.write("==============\n")
                log.write("\n")

                # print order ratios
                log.write("Ratio: order / total variance over all output quantities:\n")
                log.write("---------------------------------------------------------\n")
                for i in range(len(sobol_rel_order_mean)):
                    log.write("Order {}: {:.4f}\n".format(i+1, sobol_rel_order_mean[i]))

                log.write("\n")

                # print 1st order ratios of parameters
                log.write("Ratio: 1st order Sobol indices of parameters / total variance over all output quantities\n")
                log.write("----------------------------------------------------------------------------------------\n")

                random_vars = []
                for i in range(len(sobol_rel_1st_order_mean)):
                    log.write("{}{:s}: {:.4f}\n".format(
                                                    (max_len-len(self.random_vars[sobol_extracted_idx_1st[i]]))*' ',
                                                    self.random_vars[sobol_extracted_idx_1st[i]],
                                                    sobol_rel_1st_order_mean[i]))
                    random_vars.append(self.random_vars[sobol_extracted_idx_1st[i]])

                log.close()

                # prepare plots

                # set the global colors
                mpl.rcParams['text.color'] = '000000'
                mpl.rcParams['figure.facecolor'] = '111111'

                # set a global style
                plt.style.use('seaborn-talk')

                cmap = plt.cm.rainbow

                # make bar plot of order ratios
                labels = ['order=' + str(i) for i in range(1, len(sobol_rel_order_mean) + 1)]
                mask = np.where(sobol_rel_order_mean >= 0.05)[0]
                mask_not = np.where(sobol_rel_order_mean < 0.05)[0]
                labels = [labels[idx] for idx in mask]
                if mask_not.any():
                    labels.append('misc.')
                    values = np.hstack((sobol_rel_order_mean[mask], np.sum(sobol_rel_order_mean[mask_not])))
                else:
                    values = sobol_rel_order_mean

                colors = cmap(np.linspace(0.1, 0.9, len(labels)))

                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                ax.set_title('Sobol indices (order)')
                ax.pie(values, labels=labels, colors=colors,
                       autopct='%1.2f%%', shadow=True, explode=[0.1]*len(labels))
                plt.savefig(os.path.splitext(fn_plot)[0] + '_order.png', facecolor='#ffffff')

                # make bar plot of 1st order parameter ratios
                mask = np.where(sobol_rel_1st_order_mean >= 0.05)[0]
                mask_not = np.where(sobol_rel_1st_order_mean < 0.05)[0]
                labels = [random_vars[idx] for idx in mask]
                if mask_not.any():
                    labels.append('misc.')
                    values = np.hstack((sobol_rel_1st_order_mean[mask], np.sum(sobol_rel_1st_order_mean[mask_not])))
                else:
                    values = sobol_rel_1st_order_mean

                colors = cmap(np.linspace(0., 1., len(labels)))

                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                ax.set_title('Sobol indices 1st order (parameters)')
                ax.pie(values, labels=labels, colors = colors,
                       autopct='%1.2f%%', shadow=True, explode=[0.1]*len(labels))
                plt.savefig(os.path.splitext(fn_plot)[0] + '_parameters.png', facecolor='#ffffff')

            return sobol, sobol_idx, \
                   sobol_rel_order_mean, sobol_rel_order_std, \
                   sobol_rel_1st_order_mean, sobol_rel_1st_order_std

        else:
            return sobol, sobol_idx
        
    def globalsens(self, coeffs):
        """ Determine the global derivative based sensitivity coefficients

        Reference:
        D. Xiu, Fast Numerical Methods for Stochastic Computations: A Review,
        Commun. Comput. Phys., 5 (2009), pp. 242-272 eq. (3.14) page 255

        globalsens = calc_globalsens(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        globalsens: np.array of float [DIM x N_out]
            global derivative based sensitivity coefficients
        """

        Nmax = int(len(self.poly))
        
        self.poly_der = [[0 for x in range(self.DIM)] for x in range(Nmax)]
        poly_der_int = [[0 for x in range(self.DIM)] for x in range(Nmax)]
        poly_int = [[0 for x in range(self.DIM)] for x in range(Nmax)]
        knots_list_1D = [0 for x in range(self.DIM)]
        weights_list_1D = [0 for x in range(self.DIM)]
        
        # generate quadrature points for numerical integration for each random
        # variable separately (2*Nmax points for high accuracy)
        
        for i_DIM in range(self.DIM):
            if self.pdftype[i_DIM] == 'beta':    # Jacobi polynomials
                knots_list_1D[i_DIM], weights_list_1D[i_DIM] = quadrature_jacobi_1D(2*Nmax,self.pdfshape[0][i_DIM]-1, self.pdfshape[1][i_DIM]-1)
            if self.pdftype[i_DIM] == 'norm' or self.pdftype[i_DIM] == "normal":   # Hermite polynomials
                knots_list_1D[i_DIM], weights_list_1D[i_DIM] = quadrature_hermite_1D(2*Nmax)
        
        # preprocess polynomials        
        for i_DIM in range(self.DIM):
            for i_order in range(Nmax):
                
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_DIM] = np.polyder(self.poly[i_order][i_DIM])
                
                # evaluate poly and poly_der at quadrature points and integrate w.r.t. pdf (multiply with weights and sum up)
                # saved like self.poly [N_order x DIM]
                poly_int[i_order][i_DIM]     = np.sum(np.dot(self.poly[i_order][i_DIM](knots_list_1D[i_DIM]), weights_list_1D[i_DIM]))
                poly_der_int[i_order][i_DIM] = np.sum(np.dot(self.poly_der[i_order][i_DIM](knots_list_1D[i_DIM]) , weights_list_1D[i_DIM]))
        
        N_poly = self.poly_idx.shape[0]
        poly_der_int_mult = np.zeros([self.DIM, N_poly])
        
        for i_sens in range(self.DIM):        
            for i_poly in range(N_poly):
                A1 = 1
                
                # evaluate complete integral (joint basis function)                
                for i_DIM in range(self.DIM):
                    if i_DIM == i_sens:
                        A1 *= poly_der_int[self.poly_idx[i_poly][i_DIM]][i_DIM]
                    else:
                        A1 *= poly_int[self.poly_idx[i_poly][i_DIM]][i_DIM]
                
                
                poly_der_int_mult[i_sens,i_poly] = A1
        
        # sum up over all coefficients        
        # [DIM x N_points]  = [DIM x N_poly] * [N_poly x N_points]
        globalsens = np.dot(poly_der_int_mult, coeffs)/(2**self.DIM)
        
        return globalsens
        
    def localsens(self, coeffs, xi):
        """ Determine the local derivative based sensitivity coefficients in the point of operation xi
        in normalized coordinates.

        localsens = calc_localsens(coeffs, xi)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x DIM]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)

        Returns:
        ----------------------------------
        localsens: np.array of float [DIM x N_out]
            local sensitivity
        """

        Nmax = len(self.poly)
        
        self.poly_der = [[0 for x in range(self.DIM)] for x in range(Nmax+1)]
        poly_der_xi = [[0 for x in range(self.DIM)] for x in range(Nmax+1)]
        poly_opvals = [[0 for x in range(self.DIM)] for x in range(Nmax+1)]
        
        # preprocess polynomials        
        for i_DIM in range(self.DIM):
            for i_order in range(Nmax+1):
                
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_DIM] = np.polyder(self.poly[i_order][i_DIM])
                
                # evaluate poly and poly_der at point of operation
                poly_opvals[i_order][i_DIM] =  self.poly[i_order][i_DIM](xi[1,i_DIM])
                poly_der_xi[i_order][i_DIM] =  self.poly_der[i_order][i_DIM](xi[1,i_DIM])
        
        N_vals = 1
        poly_sens = np.zeros([self.DIM, self.N_poly])
        
        for i_sens in range(self.DIM):        
            for i_poly in range(self.N_poly):
                A1 = np.ones(N_vals)
                
                # construct polynomial basis according to partial derivatives                
                for i_DIM in range(self.DIM):
                    if i_DIM == i_sens:
                        A1 *= poly_der_xi[self.poly_idx[i_poly][i_DIM]][i_DIM]
                    else:
                        A1 *= poly_opvals[self.poly_idx[i_poly][i_DIM]][i_DIM]
                poly_sens[i_sens,i_poly] = A1
        
        # sum up over all coefficients        
        # [DIM x N_points]  = [DIM x N_poly]  *   [N_poly x N_points]
        localsens = np.dot(poly_sens,coeffs)   
        
        return localsens
    
    def pdf(self, coeffs, N_samples, output_idx=[]):
        """ Determine the estimated pdfs of the output quantities

        pdf = pdf(coeffs, N_samples, output_idx)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        N_samples: int
            Number of samples used to estimate output pdf
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
            pdf_x: nparray [100 x N_out]
                x-coordinates of output pdf (output quantity),
            pdf_y: nparray [100 x N_out]
                y-coordinates of output pdf (probability density of output quantity)
        """

        self.N_out = coeffs.shape[1]
        
        # if output index array is not provided, determine pdfs of all outputs 
        if not output_idx:
            output_idx = np.linspace(0,self.N_out-1,self.N_out)
            output_idx = output_idx[np.newaxis,:]
    
        # sample gPC expansion
        samples_in, samples_out = self.MC_sampling(coeffs, N_samples, output_idx)
        
        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100,self.N_out])
        pdf_y = np.zeros([100,self.N_out])
        
        for i_out in range(coeffs.shape[1]):
            kde = scipy.stats.gaussian_kde(samples_out.transpose(), bw_method=0.1/samples_out[:,i_out].std(ddof=1))
            pdf_x[:,i_out] = np.linspace(samples_out[:,i_out].min(), samples_out[:,i_out].max(), 100)
            pdf_y[:,i_out] = kde(pdf_x[:,i_out])
            
        return pdf_x, pdf_y

    def get_mean_random_vars(self):
        """ Determine the average values of the input random variables from their pdfs

        Returns:
        --------
        mean_random_vars: nparray of float [N_random_vars]
            Average values of the input random variables
        """
        mean_random_vars = np.zeros(self.DIM)

        for i_DIM in range(self.DIM):
            if self.pdftype[i_DIM] == 'norm' or self.pdftype[i_DIM] == 'normal':
                mean_random_vars[i_DIM] = self.pdfshape[0][i_DIM]

            if self.pdftype[i_DIM] == 'beta':
                mean_random_vars[i_DIM] = (float(self.pdfshape[0][i_DIM]) /
                                                (self.pdfshape[0][i_DIM] + self.pdfshape[1][i_DIM])) * \
                                                (self.limits[1][i_DIM] - self.limits[0][i_DIM]) + \
                                                (self.limits[0][i_DIM])

        return mean_random_vars

#%%############################################################################
# Regression based gpc object subclass
###############################################################################

class reg(gpc):
    def __init__(self, pdftype, pdfshape, limits, order, order_max, interaction_order, grid, random_vars=['']):
        """
        regression gpc subclass
        -----------------------
        reg(random_vars, pdftype, pdfshape, limits, order, order_max, interaction_order, grid)
        
        Parameters:
        -----------------------
            random_vars: list of str [DIM]
                string labels of the random variables
            pdftype: list of str [DIM]
                type of pdf 'beta' or 'norm'
            pdfshape: list of list of float
                shape parameters of pdfs
                beta-dist:   [[alpha_1, ...], [beta_1, ...]    ]
                normal-dist: [[mean_1, ...],  [std_1, ...]]
            limits: list of list of float
                upper and lower bounds of random variables
                beta-dist:   [[min_1, ...], [max_1, ...]]
                normal-dist: [[0, ... ], [0, ... ]] (not used)
            order: list of int [DIM]
                maximum individual expansion order
                generates individual polynomials also if maximum expansion order in order_max is exceeded
            order_max: int
                maximum expansion order (sum of all exponents)
                the maximum expansion order considers the sum of the orders of combined polynomials only
            interaction_order: int
                number of random variables, which can interact with each other
                all polynomials are ignored, which have an interaction order greater than the specified
            grid: object
                grid object generated in .grid.py including grid.coords and grid.coords_norm
        """
        gpc.__init__(self)
        self.random_vars    = random_vars
        self.pdftype        = pdftype
        self.pdfshape       = pdfshape
        self.limits         = limits 
        self.order          = order                     
        self.order_max      = order_max
        self.interaction_order = interaction_order
        self.DIM            = len(pdftype)
        self.grid           = grid        
        self.N_grid         = grid.coords.shape[0]
        self.relerror_loocv = []
        self.nan_elm        = []  # which elements were dropped due to NAN

        # setup polynomial basis functions
        self.setup_polynomial_basis()    
        
        # construct gpc matrix [Ngrid x Npolybasis]
        self.construct_gpc_matrix()

        # get mean values of input random variables
        self.mean_random_vars = self.get_mean_random_vars()

    def expand(self, data):
        """ Determine the gPC coefficients by the regression method

        coeffs = expand(self, data)

        Parameters:
        ----------------------------------
        data: np.array of float [N_grid x N_out]
            results from simulations with N_out output quantities,

        Returns:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gPC coefficients
        """

        #print 'Determine gPC coefficients ...'
        self.N_out = data.shape[1]
        
        if data.shape[0] != self.Ainv.shape[1]:
            if data.shape[1] != self.Ainv.shape[1]:
                print("Please check format of input data: matrix [N_grid x N_out] !")
            else:
                data = data.T
        # coeffs    ... [N_coeffs x N_points] 
        # Ainv      ... [N_coeffs x N_grid]
        # data      ... [N_grid   x N_points]

        return np.dot(self.Ainv, data)

    def LOOCV(self, data, n_loocv_points=50):
        """ Perform leave one out cross validation of gPC and adds result to self.relerror_loocv

        Parameters:
        ----------------------------------
        data: np.array() [N_grid x N_out]
            Results from N_grid simulations with N_out output quantities
        n_loocv_points: int
            Number of repetitions in leave one out cross validation to determine stable error

        Returns:
        ----------------------------------
        relerror_LOOCV: float
            relative mean error of leave one out cross validation
        """

        # define number of performed cross validations (max 100)
        n_loocv_points = np.min((data.shape[0], n_loocv_points))

        # make list of indices, which are randomly sampled
        loocv_point_idx = random.sample(list(range(data.shape[0])), n_loocv_points)

        start = time.time()
        relerror = np.zeros(n_loocv_points)
        # data_temp = np.zeros(data[i,:].shape)
        for i in range(n_loocv_points):
            # get mask of eliminated row
            # a = time.time()
            mask = np.arange(data.shape[0]) != loocv_point_idx[i]

            # invert reduced gpc matrix
            Ainv_LOO = np.linalg.pinv(self.A[mask, :])

            # determine gpc coefficients (this takes a lot of time for large problems)
            coeffs_LOO = np.dot(Ainv_LOO, data[mask, :])
            data_temp = data[loocv_point_idx[i], ]
            relerror[i] = scipy.linalg.norm(data_temp - np.dot(self.A[loocv_point_idx[i], :], coeffs_LOO)) / scipy.linalg.norm(data_temp)
            fancy_bar("LOOCV", int(i+1), int(n_loocv_points))

        # store result in relerror_loocv
        self.relerror_loocv.append(np.mean(relerror))

        print(" (" + str(time.time()-start) + ")")

        return self.relerror_loocv[-1]
             
#%%############################################################################
# Quadrature based gpc object subclass
###############################################################################

class quad(gpc):
    def __init__(self, pdftype, pdfshape, limits, order, order_max, interaction_order, grid, random_vars=['']):
        """
        regression gpc subclass
        -----------------------
        quad(random_vars, pdftype, pdfshape, limits, order, order_max, interaction_order, grid)

        Parameters:
        -----------------------
            pdftype: list of str [DIM]
                type of pdf 'beta' or 'norm'
            pdfshape: list of list of float
                shape parameters of pdfs
                beta-dist:   [[alpha], [beta]    ]
                normal-dist: [[mean],  [variance]]
            limits: list of list of float
                upper and lower bounds of random variables
                beta-dist:   [[a1 ...], [b1 ...]]
                normal-dist: [[0 ... ], [0 ... ]] (not used)
            order: list of int [DIM]
                maximum individual expansion order
                generates individual polynomials also if maximum expansion order in order_max is exceeded
            order_max: int
                maximum expansion order (sum of all exponents)
                the maximum expansion order considers the sum of the orders of combined polynomials only
            interaction_order: int
                number of random variables, which can interact with each other
                all polynomials are ignored, which have an interaction order greater than the specified
            grid: object
                grid object generated in .grid.py including grid.coords and grid.coords_norm
            random_vars: list of str [DIM]
                string labels of the random variables
        """
        gpc.__init__(self)
        self.random_vars    = random_vars
        self.pdftype        = pdftype
        self.pdfshape       = pdfshape
        self.limits         = limits
        self.order          = order
        self.order_max      = order_max
        self.interaction_order = interaction_order
        self.DIM            = len(pdftype)
        self.grid           = grid  
        self.N_grid         = grid.coords.shape[0]
        
        # setup polynomial basis functions
        self.setup_polynomial_basis()
        
        # construct gpc matrix [Ngrid x Npolybasis]
        self.construct_gpc_matrix()

        # get mean values of input random variables
        self.mean_random_vars = self.get_mean_random_vars()
           
    def expand(self, data):
        """ Determine the gPC coefficients by the quadrature method

        coeffs = expand(self, data)

        Parameters:
        ----------------------------------
        data: np.array of float [N_grid x N_out]
            results from simulations with N_out output quantities,

        Returns:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gPC coefficients
        """

        #print 'Determine gPC coefficients ...'
        self.N_out = data.shape[1]
        
        # check if quadrature rule (grid) fits to the distribution (pdfs)
        grid_pdf_fit = 1
        for i_DIM in range(self.DIM):
            if self.pdftype[i_DIM] == 'beta':
                if not (self.grid.gridtype[i_DIM] == 'jacobi'):
                    grid_pdf_fit = 0
                    break
            elif (self.pdftype[i_DIM] == 'norm') or (self.pdftype[i_DIM] == 'normal'):
                if not (self.grid.gridtype[i_DIM] == 'hermite'):
                    grid_pdf_fit = 0
                    break
    
        # if not, calculate joint pdf
        if not(grid_pdf_fit):
            joint_pdf = np.ones(self.grid.coords_norm.shape)
            
            for i_DIM in range(self.DIM):
                if self.pdftype[i_DIM] == 'beta':
                    joint_pdf[:, i_DIM] = pdf_beta(self.grid.coords_norm[:, i_DIM],
                                                   self.pdfshape[0][i_DIM],
                                                   self.pdfshape[1][i_DIM], -1, 1)

                if self.pdftype[i_DIM] == 'norm' or self.pdftype[i_DIM] == 'normal':
                    joint_pdf[:, i_DIM] = scipy.stats.norm.pdf(self.grid.coords_norm[:, i_DIM])
            
            joint_pdf = np.array([np.prod(joint_pdf, axis=1)]).transpose()
            
            # weight data with the joint pdf
            data = data*joint_pdf*2**self.DIM
        
        # scale rows of gpc matrix with quadrature weights
        A_weighted = np.dot(np.diag(self.grid.weights), self.A)
        # scale = np.outer(self.grid.weights, 1./self.poly_norm_basis)        
        # A_weighted = np.multiply(self.A, scale)
        
        # determine gpc coefficients [N_coeffs x N_output]
        return np.dot(data.transpose(), A_weighted).transpose()           
