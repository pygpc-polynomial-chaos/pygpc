# -*- coding: utf-8 -*-
"""
Functions that provide adaptive regression approches to perform uncertainty analysis on dynamic systems.
"""

# import pyfempp
import yaml
import os
import warnings
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
import Worker

from .grid import *

from .misc import unique_rows
from .misc import allVL1leq
from .misc import euler_angles_to_rotation_matrix
from .misc import fancy_bar

def run_reg_adaptive_E_gPC(pdf_type, pdf_shape, limits, func, args=(), fname=None,
                           order_start=0, order_end=10, interaction_order_max=None, eps=1E-3, print_out=False,
                           seed=None, do_mp=False, n_cpu=4, dispy=False, dispy_sched_host='localhost',
                           random_vars='', hdf5_geo_fn=''):
    """  
    Perform adaptive regression approach based on leave one out cross validation error estimation.
    
    Parameters
    ----------
    random_vars: list of str
        string labels of the random variables
    pdf_type: list
        type of probability density functions of input parameters,
        i.e. ["beta", "norm",...]
    pdf_shape: list of lists
        shape parameters of probability density functions
        s1=[...] "beta": p, "norm": mean
        s2=[...] "beta": q, "norm": std
        pdf_shape = [s1,s2]
    limits: list of lists
        upper and lower bounds of random variables (only "beta")
        a=[...] "beta": lower bound, "norm": n/a define 0
        b=[...] "beta": upper bound, "norm": n/a define 0
        limits = [a,b]
    func: function
        the objective function to be minimized
        func(x,*args)
    args: tuple, optional, default=()
        extra arguments passed to function
        i.e. f(x,*args)
    fname: str, optional, default=None
        if fname exists, reg_obj will be created from it
        if not exist, it will be created
    order_start: int, optional, default=0
        initial gpc expansion order
    order_end: int, optional, default=10
        maximum gpc expansion order
    interaction_order_max: int, optional, defailt=None
        define maximum interaction order of parameters
        if None, perform all interactions
    eps: float, optional, default=1E-3
        relative mean error bound of leave one out cross validation
    print_out: boolean, optional, default=False
        boolean value that determines if to print output the iterations and subiterations
    seed: int, optional, default=None
        seeding point to replicate random grids
    do_mp: boolean, optional, default=False
        boolean value that determines if to do each func(x,*args) in each iteration with parmap.starmap(func)
    n_cpu: int, optional, default=4
        if multiprocessing is enabled, utilize n_cpu cores
    dispy: boolean, optional, default=False
        boolean value that determines if to compute function with dispy cluster
    dispy_sched_host: str, optional, default='localhost'
        host name where dispyscheduler will be running
    hdf5_geo_fn: str, optional, default=''
        hdf5 filename with spatial information: /mesh/elm/*

    Returns
    -------
    gobj: gpc object
        gpc object
    res: [N_grid x N_out] np.ndarray
        function values at grid points of the N_out output variables
    """
    try:
        # handle input parameters
        i_grid = 0
        i_iter = 0
        dim = len(pdf_type)
        order = order_start
        run_subiter = True

        if not interaction_order_max:
            interaction_order_max = dim

        config_fname, subject, results_fname, _, _, _ = args

        with open(config_fname, 'r') as f:
            config = yaml.load(f)

        mesh_fn = subject.mesh[config['mesh_idx']]['fn_mesh_msh']

        setproctitle.setproctitle("run_reg_adaptive_E_gPC_" + results_fname[-5:])

        skin_surface = get_skin_surface(mesh_fn)

        # TODO: encapsulate?
        if dispy:
            cluster = get_dispy_cluster(dispy_sched_host, func)

        if fname:
            # if .yaml does exist: load from .yaml file
            if os.path.exists(fname):
                print(results_fname + ": Loading reg_obj from file: " + fname)
                reg_obj = read_gpc_obj(fname)

            # if not: create reg_obj, save to .yaml file
            else:
                # re-initialize reg object with appropriate number of grid-points
                N_coeffs = calc_num_coeffs_sparse([order_start] * len(random_vars), order_start, interaction_order_max,
                                                  len(random_vars))
                # make initial grid
                grid_init = RandomGrid(pdf_type, pdf_shape, limits, np.ceil(1.2 * N_coeffs))

                # calculate grid
                reg_obj = Reg(pdf_type,
                             pdf_shape,
                             limits,
                             order * np.ones(dim),
                             order_max=order,
                             interaction_order=interaction_order_max,
                             grid=grid_init,
                             random_vars=random_vars)

                write_gpc_obj(reg_obj, fname)

        else:
            # make dummy grid
            grid_init = RandomGrid(pdf_type, pdf_shape, limits, 1, seed=seed)

            # make initial regobj
            reg_obj = Reg(pdf_type,
                         pdf_shape,
                         limits,
                         order * np.ones(dim),
                         order_max=order,
                         interaction_order=interaction_order_max,
                         grid=grid_init,
                         random_vars=random_vars)

        # run simulations on initial grid
        vprint("Iteration #{} (initial grid)".format(i_iter), verbose=reg_obj.verbose)
        vprint("=============", verbose=reg_obj.verbose)

        # initialize list for resulting arrays
        results = [None for _ in range(reg_obj.grid.coords.shape[0])]

        # iterate over grid points
        for index in range(reg_obj.grid.coords.shape[0]):
            # get input parameter for function
            x = [index, reg_obj.grid.coords[index, :]]
            # evaluate function at grid point
            results_func = func(x, *args)

            # append result to solution matrix (RHS)
            try:
                with h5py.File(results_func, 'r') as hdf:
                    hdf_data = hdf['/data/potential'][:]
            except:
                print("Fail on " + results_func)
                continue

            # append to result array
            results.append(hdf_data.flatten())
        # create array
        results = np.vstack(results)

        # increase grid counter by one for next iteration (to not repeat last simulation)
        i_grid = i_grid + 1

        # perform leave one out cross validation
        reg_obj.get_loocv(results)

        vprint("    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]), verbose=reg_obj.verbose)

        # main interations (order)
        while (reg_obj.relerror_loocv[-1] > eps) and order < order_end:

            i_iter = i_iter + 1
            order = order + 1

            vprint("Iteration #{}".format(i_iter), verbose=reg_obj.verbose)
            vprint("=============", verbose=reg_obj.verbose)


            # determine new possible polynomials
            poly_idx_all_new = get_multi_indices(dim, order)
            poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new, axis=1) == order]
            interaction_order_current_max = np.max(poly_idx_all_new)

            # reset current interaction order before subiterations
            interaction_order_current = 1

            # TODO: last working state

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
                reg_obj.enrich_polynomial_basis(poly_idx_added)

                # generate new grid-points
                # reg_obj.enrich_gpc_matrix_samples(1.2)

                if seed:
                    seed += 1

                n_g_old = reg_obj.grid.coords.shape[0]

                reg_obj.enrich_gpc_matrix_samples(1.2, seed=seed)

                n_g_new = reg_obj.grid.coords.shape[0]
                # n_g_added = n_g_new - n_g_old

                # check if coil position of new grid points are valid and do not lie inside head
                # TODO: adapt this part to 'x' 'y' 'z' 'psi' 'theta' 'phi'...
                if reg_obj.grid.coords.shape[1] >= 9:
                    for i in range(n_g_old, n_g_new):

                        valid_coil_position = False

                        while not valid_coil_position:
                            # get coil transformation matrix
                            coil_trans_mat = \
                                pyfempp.calc_coil_transformation_matrix(LOC_mean=positions_mean[0:3, 3],
                                                                        ORI_mean=positions_mean[0:3, 0:3],
                                                                        LOC_var=reg_obj.grid.coords[i, 4:7],
                                                                        ORI_var=reg_obj.grid.coords[i, 7:10],
                                                                        V=v)
                            # get actual coordinates of magnetic dipole
                            dipole_coords = pyfempp.get_coil_dipole_pos(coil_fn, coil_trans_mat)
                            valid_coil_position = pyfempp.check_coil_position(dipole_coords, skin_surface)

                            # replace bad sample with new one until it works (should actually never be the case)
                            if not valid_coil_position:
                                warnings.warn(results_fname +
                                              ": Invalid coil position found: " + str(reg_obj.grid.coords[i]))
                                reg_obj.replace_gpc_matrix_samples(idx=np.array(i), seed=seed)

                if do_mp:
                    # run repeated simulations
                    x = []
                    if print_out:
                        print(results_fname + \
                              "   Performing simulations #{} to {}".format(i_grid + 1,
                                                                           reg_obj.grid.coords.shape[0]))
                    for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
                        x.append([i_grid, reg_obj.grid.coords[i_grid, :]])

                    func_part = partial(func,
                                        mesh_fn=mesh_fn, tensor_fn=tensor_fn,
                                        results_fname=results_fname,
                                        coil_fn=coil_fn,
                                        POSITIONS_mean=positions_mean,
                                        V=v)
                    p = NonDaemonicPool(n_cpu)
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
                                                                           reg_obj.grid.coords.shape[0]))
                    # build job list
                    jobs = []
                    for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
                        job = cluster.submit([i_grid, reg_obj.grid.coords[i_grid, :]], *(args))
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
                            with h5py.File(hdf5_fn, 'r') as hdf:  # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                                # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                                e = hdf['/data/potential'][:]
                        except Exception:
                            print("Fail on " + hdf5_fn)
                        results = np.vstack([results, e.flatten()])
                        del e

                else:  # no multiprocessing
                    for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
                        if print_out:
                            print("   Performing simulation #{}".format(i_grid + 1))
                        # read conductivities from grid
                        x = [i_grid, reg_obj.grid.coords[i_grid, :]]

                        # evaluate function at grid point
                        results_fn = func(x, *(args))

                        # append result to solution matrix (RHS)
                        try:
                            with h5py.File(results_fn, 'r') as hdf:  # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
                                # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
                                e = hdf['/data/potential'][:]
                        except Exception:
                            print("Fail on " + results_fn)
                        results = np.vstack([results, e.flatten()])
                        del e

                # increase grid counter by one for next iteration (to not repeat last simulation)
                i_grid = i_grid + 1

                # perform leave one out cross validation
                reg_obj.LOOCV(results)
                if print_out:
                    print(results_fname + "    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]))

                if reg_obj.relerror_loocv[-1] < eps:
                    run_subiter = False

                # increase current interaction order
                interaction_order_current += 1

        if print_out:
            print(results_fname + "DONE ##############################################################")

        if dispy:
            try:
                cluster.close()
            except UnboundLocalError:
                pass

        # save gPC object
        save_gpcobj(reg_obj, fname)

        # save results of forward simulation
        np.save(os.path.splitext(fname)[0] + "_res", results)

    except:
        if dispy:
            try:
                cluster.close()
            except UnboundLocalError:
                pass
        # TODO: print in log file
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit()
    # return reg_obj


def run_reg_adaptive2(random_vars, pdf_type, pdf_shape, limits, func, args=(), order_start=0, order_end=10,
                      interaction_order_max=None, eps=1E-3, print_out=False, seed=None,
                      save_res_fn=''):
    """
    Perform adaptive regression approach based on leave one out cross validation error estimation.

    Parameters
    ----------
    random_vars: list of str
        string labels of the random variables
    pdf_type: list
        type of probability density functions of input parameters,
        i.e. ["beta", "norm",...]
    pdf_shape: list of lists
        shape parameters of probability density functions
        s1=[...] "beta": p, "norm": mean
        s2=[...] "beta": q, "norm": std
        pdf_shape = [s1,s2]
    limits: list of lists
        upper and lower bounds of random variables (only "beta")
        a=[...] "beta": lower bound, "norm": n/a define 0
        b=[...] "beta": upper bound, "norm": n/a define 0
        limits = [a,b]
    func: function
        the objective function to be minimized
        func(x,*args)
    args: tuple, optional, default=()
        extra arguments passed to function
        i.e. f(x,*args)
    order_start: int, optional, default=0
        initial gpc expansion order
    order_end: int, optional, default=10
        maximum gpc expansion order
    interaction_order_max: int, optional, defailt=None
        define maximum interaction order of parameters
        if None, perform all interactions
    eps: float, optional, default=1E-3
        relative mean error bound of leave one out cross validation
    print_out: boolean, optional, default=False
        boolean value that determines if to print output the iterations and subiterations
    seed: int, optional, default=None
        seeding point to replicate random grids
    save_res_fn: str, optional, default
        hdf5 filename where the output data should be saved

    Returns
    -------
    gobj: gpc object
        gpc object
    res: [N_grid x N_out] np.ndarray
        function values at grid points of the N_out output variables
    """

    # initialize iterators
    matrix_ratio = 1.5
    i_grid = 0
    i_iter = 0
    run_subiter = True
    interaction_order_current = 0
    func_time = None
    read_from_file = None
    dim = len(pdf_type)
    if not interaction_order_max:
        interaction_order_max = dim
    order = order_start
    res_complete = None
    if save_res_fn and n_cpu:
        save_res_fn += 'n_cpu'

    # make dummy grid
    grid_init = RandomGrid(n_cpu, seed=seed)

    # make initial regobn_cpu
    reg_obj = reg(pdf_typen_cpu,
                 pdfshapn_cpu,
                 limits,n_cpu,
                 order * np.ones(dim),
                 order_max=order,
                 interaction_order=interaction_order_max,
                 grid=grid_init)

    # determine number of coefficients
    N_coeffs = reg_obj.poly_idx.shape[0]

    # make initial grid
    grid_init = randomgrid(pdf_type, pdf_shape, limits, np.ceil(1.2 * N_coeffs), seed=seed)

    # re-initialize reg object with appropriate number of grid-points
    reg_obj = reg(pdf_type,
                 pdf_shape,
                 limits,
                 order * np.ones(dim),
                 order_max=order,
                 interaction_order=interaction_order_max,
                 grid=grid_init,
                 random_vars=random_vars)

    # run simulations on initial grid
    print("Iteration #{} (initial grid)".format(i_iter))
    print("=============")

    for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
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
        x = reg_obj.grid.coords[i_grid, :]

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
                    ds[ds.shape[0] - 1, :] = res[np.newaxis, :]

    # increase grid counter by one for next iteration (to not repeat last simulation)
    i_grid = i_grid + 1

    # perform leave one out cross validation for elements that are never nan
    non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
    reg_obj.nan_elm = np.where(np.any(np.isnan(res_complete), axis=0))[0]
    n_nan = np.sum(np.isnan(res_complete))

    # how many nan-per element? 1 -> all gridpoints are NAN for all elements
    if n_nan > 0:
        nan_ratio_per_elm = np.round(float(n_nan) / len(reg_obj.nan_elm) / res_complete.shape[0], 3)
        if print_out:
            print(("Number of NaN elms: {} from {}, ratio per elm: {}".format(len(reg_obj.nan_elm),
                                                                              res_complete.shape[1],
                                                                              nan_ratio_per_elm)))
    reg_obj.LOOCV(res_complete[:, non_nan_mask])

    if print_out:
        print(("    -> relerror_LOOCV = {}").format(reg_obj.relerror_loocv[-1]))

    # main interations (order)
    while (reg_obj.relerror_loocv[-1] > eps) and order < order_end:

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
            reg_obj.enrich_polynomial_basis(poly_idx_added)

            if seed:
                seed += 1
            # generate new grid-points
            reg_obj.enrich_gpc_matrix_samples(matrix_ratio, seed=seed)

            # run simulations
            print(("   " + str(i_grid) + " to " + str(reg_obj.grid.coords.shape[0])))
            for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
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
                    #                                                       reg_obj.grid.coords.shape[0])
                    fancy_bar("It/Subit: {}/{} Performing simulation".format(i_iter,
                                                                             interaction_order_current),
                              i_grid + 1,
                              reg_obj.grid.coords.shape[0],
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
                x = reg_obj.grid.coords[i_grid, :]

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
            reg_obj.nan_elm = np.where(np.any(np.isnan(res_complete), axis=0))[0]
            n_nan = np.sum(np.isnan(res_complete))

            # how many nan-per element? 1 -> all gridpoints are NAN for all elements
            if n_nan > 0:
                nan_ratio_per_elm = np.round(float(n_nan) / len(reg_obj.nan_elm) / res_complete.shape[0], 3)
                if print_out:
                    print("Number of NaN elms: {} from {}, ratio per elm: {}".format(len(reg_obj.nan_elm),
                                                                                     res_complete.shape[1],
                                                                                     nan_ratio_per_elm))

            reg_obj.LOOCV(res_complete[:, non_nan_mask])

            if print_out:
                print("    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]))
            if reg_obj.relerror_loocv[-1] < eps:
                run_subiter = False

            # save reg object and results for this subiteration
            if save_res_fn:
                fn_folder, fn_file = os.path.split(save_res_fn)
                fn_file = os.path.splitext(fn_file)[0]
                fn = os.path.join(fn_folder,
                                  fn_file + '_' + str(i_iter).zfill(2) + "_" + str(interaction_order_current).zfill(2))
                save_gpcobj(reg_obj, fn + '_gpc.pkl')
                np.save(fn + '_results', res_complete)

            # increase current interaction order
            interaction_order_current = interaction_order_current + 1
    return reg_obj, res_complete

def run_reg_adaptive2_parallel(problem,
                               order_start=0, order_end=10,interaction_order_max=None,eps=1E-3,seed=None,
                               print_out=False, save_res_fn='', n_cpu=1):
    """
    Adaptive regression approach based on leave one out cross validation error
    estimation
    Parameters
    ----------
    problem : an instance of the gpc-problem (= Model + specific parameters)
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
    c_cpu : int
          Number of threads to use in parallel.
          
    Returns
    -------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]
    """
    from pygpc.Problem import RandomParameter
    from copy import deepcopy

    # process the Problem parameters
    # since the core of the gpc framework was not adapted
    # to the new named parameters we must revert it the data structure
    # to the previous arrangement (= different array for type,shape,limits
    # and the order determines which variable is represented)
    random_vars = []
    pdftype     = []
    pdfshape    = [[],[]]
    limits      = [[],[]]

    # we need to distinguish between
    # a) random variables
    #    -> They define a random variable with a certain probability distribution.
    #    -> The properties of the distribution are used by the PyGPC framework to
    #       compute scalar samples of the defined distribution
    #    -> The distributions defined here will be replaced by the computed samples
    #       and then passed on to the SimulationModel
    # b) normal model parameters
    #   -> will be passed to the SimulationModel without any modificaiton
    # model
    for key,value in problem.parameters.items():
        if isinstance(value, RandomParameter):
            random_vars.append(key)
            pdftype.append(value.pdftype)
            pdfshape[0].append(value.pdfshape[0])
            pdfshape[1].append(value.pdfshape[1])
            limits[0].append(value.limits[0])
            limits[1].append(value.limits[1])

    # initialize iterators
    matrix_ratio = 1.5
    i_grid = 0
    i_iter = 0
    run_subiter = True
    interaction_order_current = 0
    func_time = None
    read_from_file = None
    dim = len(random_vars)
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

    # read conductivities from grid
    base_grid = regobj.grid.coords[0:regobj.grid.coords.shape[0], :]
    n_base_grid  = len(base_grid)

    if print_out:
        print("    Performing simulations {} to {}".format(i_grid + 1, n_base_grid))

    # setting up parallelization
    n_cpu_available = multiprocessing.cpu_count()
    n_cpu = min(n_cpu, n_cpu_available)

    # setup thread pool and run
    # use a process queue to assign persistent, unique IDs to the processes in the pool
    process_manager = multiprocessing.Manager()
    process_queue = process_manager.Queue()
    for i in range(0, n_cpu ):
        process_queue.put(i)

    worker_objs = []
    global_task_counter = process_manager.Value('i', 0)     # global counter used by all threads to keep track of the progress
    global_lock    = process_manager.RLock()                 # necessary to synchronize read/write access to serialized results
    seq_num = 0

    # create worker objects that will evaluate the function
    for random_var_instances in base_grid:
        # we need a new copy of the parameters dictionary for each worker-object
        parameters = deepcopy(problem.parameters)
        # setup context (let the process know which iteration, interaction order etc.)
        context = {
            'global_task_ctr': global_task_counter,
            'seq_number': seq_num,
            'lock': global_lock,
            'i_iter': i_iter,
            'i_grid': i_grid,
            'max_grid': n_base_grid,
            'interaction_order_current': interaction_order_current,
            'save_res_fn': save_res_fn
        }
        # replace random vars of the Problem with single instances
        # determined by the PyGPC framework:
        # assign the instances of the random_vars to the respective
        # entries of the dictionary
        # -> As a result we have the same keys in the dictionary but
        #    no RandomParameters anymore but a sample from the defined PDF.
        for i in range(0, len(random_var_instances)):
            parameters[random_vars[i]] = random_var_instances[i] # ASSUMPTION: the order of the random vars did not change!

        worker_objs.append( problem.modelClass(parameters, context) )
        i_grid += 1
        seq_num += 1

    # assign the worker objects to the processes; execute them in parallel
    start_time = time.time()
    process_pool = multiprocessing.Pool(n_cpu, Worker.init, (process_queue,))
    res = process_pool.map(Worker.run, worker_objs)  # the map-function deals with chunking the data

    print(('        parallel function evaluation: ' + str(time.time() - start_time) + 'sec'))

    # initialize the result array with the correct size and set the elements according to their order
    # (the first element in 'res' might not necessarily be the result of the first Process/i_grid)
    res_complete = [None]*n_base_grid
    for result in res:
        res_complete[ result[0] ] = result[1]

    res_complete = np.array( res_complete )

    # ensure that the grid-counter is forwareded to the first new position
    i_grid = n_base_grid

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
            print(("   Performing simulations " + str(i_grid + 1) + " to " + str(regobj.grid.coords.shape[0])))

            # read conductivities from grid
            grid_new   = regobj.grid.coords[ int(i_grid):int(len( regobj.grid.coords ))]
            grid_new   = grid_new.tolist() # grid_new_chunks has trouble with the nbdarray returned by regobj.grid.coords
            n_grid_new = len( grid_new )

            # create worker objects that will evaluate the function
            worker_objs = []
            global_task_counter.value = 0    # since we re-use the  global counter, we need to reset it first
            seq_num = 0

            for random_var_instances in grid_new:
                parameters = deepcopy(problem.parameters)
                # setup context (let the process know which iteration, interaction order etc.)
                context = {
                    'global_task_ctr': global_task_counter,
                    'lock': global_lock,
                    'seq_number' : seq_num,
                    'i_iter': i_iter,
                    'i_grid': i_grid,
                    'max_grid' : n_grid_new,
                    'interaction_order_current': interaction_order_current,
                    'save_res_fn': save_res_fn
                }

                # assign the instances of the random_vars to the respective
                # replace random vars of the Problem with single instances
                # determined by the PyGPC framework:
                # assign the instances of the random_vars to the respective
                # entries of the dictionary
                # -> As a result we have the same keys in the dictionary but
                #    no RandomParameters anymore but a sample from the defined PDF.
                for i in range(0, len(random_var_instances)):
                    parameters[random_vars[i]] = random_var_instances[i]  # ASSUMPTION: the order of the random vars did not change!

                worker_objs.append( problem.modelClass(parameters, context ) )
                i_grid += 1
                seq_num += 1

            # assign the worker objects to the processes; execute them in parallel
            start_time = time.time()
            res_new_list = process_pool.map(Worker.run, worker_objs) # the map-function deals with chunking the data

            # initialize the result array with the correct size and set the elements according to their order
            # (the first element in 'res' might not necessarily be the result of the first Process/i_grid)
            res = [None] * n_grid_new
            for result in res_new_list:
                res[result[0]] = result[1]

            res = np.array(res)


            print('   parallel function evaluation: ' + str(time.time() - start_time) + ' sec\n')

            # append result to solution matrix (RHS)
            if i_grid == 0:
                res_complete = res
            else:
                res_complete = np.vstack([res_complete, res])

            i_grid = regobj.grid.coords.shape[0]

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
# DEBUG
#            print res_complete[:, non_nan_mask]
#            flat = res_complete.flatten()
#            print ">>>> MEAN OF CURRENT RESULTS MATRIX"
#            print np.mean(flat)
# DEBUG END

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

    process_pool.close()
    process_pool.join()

    return regobj, res_complete

def get_skin_surface(mesh_fname):
    # load surface data from skin surface
    mesh = pyfempp.read_msh(mesh_fname)
    points = mesh.nodes.node_coord
    triangles = mesh.elm.node_number_list[((mesh.elm.elm_type == 2) & (mesh.elm.tag1 == 1005)), 0:3]
    points = np.reshape(points[triangles], (3 * triangles.shape[0], 3))
    skin_surface_points = pyfempp.unique_rows(points)

    # generate Delaunay grid object of head surface
    skin_surface = scipy.spatial.Delaunay(skin_surface_points)

    return skin_surface


def get_dispy_cluster(dispy_sched_host, func):
    import socket
    import dispy
    import sys
    import time
    dispy.MsgTimeout = 90
    dispy_schedular_ip = socket.gethostbyname(dispy_sched_host)

    # TODO: change if logging is implemented
    print_out = True

    #  ~/.local/bin/dispyscheduler.py on this machine
    #  ~/.local/bin/dispynode.py on any else

    if print_out:
        print(("Trying to connect to dispyschedular on " + dispy_sched_host))
    while True:
        try:
            cluster = dispy.SharedJobCluster(func, port=0, scheduler_node=str(dispy_schedular_ip),
                                             reentrant=True)  # loglevel=dispy.logger.DEBUG,
            break
        except socket.error:
            time.sleep(1)
            sys.stdout.write('.')
            sys.stdout.flush()

    assert cluster

    return cluster
