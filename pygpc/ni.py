# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:21:45 2016

@author: Konstantin Weise
"""
import pyfempp
import os
import warnings

from _functools import partial
from .misc import *
from .io import *
from .reg import *
from .misc import *


def run_reg_adaptive_E_gPC(pdf_type, pdf_shape, limits, func, args=(), fname=None,
                           order_start=0, order_end=10, interaction_order_max=None, eps=1E-3, print_out=False,
                           seed=None, do_mp=False, n_cpu=4, dispy=False, dispy_sched_host='localhost',
                           random_vars='', hdf5_geo_fn=''):
    """  
    Adaptive regression approach based on leave one out cross validation error estimation
    
    Parameters
    ---------------------------
    
    # pdf object
    random_vars : list of str
        string labels of the random variables
    pdf_type : list
        Type of probability density functions of input parameters,
        i.e. ["beta", "norm",...]
    pdf_shape : list of lists
        Shape parameters of probability density functions
        s1=[...] "beta": p, "norm": mean
        s2=[...] "beta": q, "norm": std
        pdf_shape = [s1,s2]
    limits : list of lists
        Upper and lower bounds of random variables (only "beta")
        a=[...] "beta": lower bound, "norm": n/a define 0
        b=[...] "beta": upper bound, "norm": n/a define 0
        limits = [a,b]
    # pdf object

    # sys object
    func : callable func(x,*args)
        The objective function to be minimized.
    args : tuple, optional
        Extra arguments passed to func, i.e. f(x,*args)
    # sys object

    # new gpc object
    fname : String, None
        If fname exists, reg_obj will be created from it. If not exist, it is created.
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
    # new gpc object


    Returns
    ---------------------------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]

    """
    try:
        import setproctitle
        i_grid = 0
        i_iter = 0
        dim = len(pdf_type)
        order = order_start
        run_subiter = True

        if not interaction_order_max:
            interaction_order_max = dim

        # mesh_fn, tensor_fn, results_folder, coil_fn, positions_mean, v = args
        fn_cfg, subject, results_folder, _, _, _ = args

        with open(fn_cfg, 'r') as f:
            config = yaml.load(f)

        mesh_fn = subject.mesh[config['mesh_idx']]['fn_mesh_msh']

        save_res_fn = False
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
        # TODO: Work on code above

        if fname:
            # if .yaml does exist: load from .yaml file
            if os.path.exists(fname):
                print(results_folder + ": Loading reg_obj from file: " + fname)
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
                                warnings.warn(results_folder +
                                              ": Invalid coil position found: " + str(reg_obj.grid.coords[i]))
                                reg_obj.replace_gpc_matrix_samples(idx=np.array(i), seed=seed)

                if do_mp:
                    # run repeated simulations
                    x = []
                    if print_out:
                        print(results_folder + \
                              "   Performing simulations #{} to {}".format(i_grid + 1,
                                                                           reg_obj.grid.coords.shape[0]))
                    for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
                        x.append([i_grid, reg_obj.grid.coords[i_grid, :]])

                    func_part = partial(func,
                                        mesh_fn=mesh_fn, tensor_fn=tensor_fn,
                                        results_folder=results_folder,
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
                    print(results_folder + "    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]))

                if reg_obj.relerror_loocv[-1] < eps:
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
        save_gpcobj(reg_obj, fname)

        # save results of forward simulation
        np.save(os.path.splitext(fname)[0] + "_res", results)

    except Exception as e:
        if dispy:
            try:
                cluster.close()
            except UnboundLocalError:
                pass
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        quit()
    return  # reg_obj, RES


def run_reg_adaptive2(random_vars, pdf_type, pdf_shape, limits, func, args=(), order_start=0, order_end=10,
                      interaction_order_max=None, eps=1E-3, print_out=False, seed=None,
                      save_res_fn=''):
    """
    Adaptive regression approach based on leave one out cross validation error
    estimation

    Parameters
    ----------
    random_vars : list of str
        string labels of the random variables
    pdf_type : list
              Type of probability density functions of input parameters,
              i.e. ["beta", "norm",...]
    pdf_shape : list of lists
               Shape parameters of probability density functions
               s1=[...] "beta": p, "norm": mean
               s2=[...] "beta": q, "norm": std
               pdf_shape = [s1,s2]
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
    dim = len(pdf_type)
    if not interaction_order_max:
        interaction_order_max = dim
    order = order_start
    res_complete = None
    if save_res_fn and n_cpu:
        save_res_fn += 'n_cpu

    # make dummy grid
    grid_init = randomgrn_cpu, seed=seed)

    # make initial regobn_cpu
    reg_obj = reg(pdf_typen_cpu
                 pdfshapn_cpu
                 limits,n_cpu
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