# -*- coding: utf-8 -*-
from pygpc.Computation import *
from .MEGPC import *
from .Grid import *
from .misc import nrmsd
from .misc import get_cartesian_product
from .Visualization import *
from scipy.signal import savgol_filter
import os
import scipy.stats
import h5py
import warnings

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    warnings.warn("If you want to use plot functionality from pygpc, "
                  "please install matplotlib (pip install matplotlib).")
    pass


def validate_gpc_mc(gpc, coeffs, n_samples=1e4, output_idx=0, n_cpu=1, smooth_pdf=True, fn_out=None, plot=True):
    """
    Compares gPC approximation with original model function. Evaluates both at "n_samples" sampling points and
    evaluates the root mean square deviation. It also computes the pdf at the output quantity with output_idx
    and saves the plot as fn_pdf.png and fn_pdf.pdf.

    Parameters
    ----------
    gpc : GPC object instance
        GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out]
        GPC coefficients
    n_samples : int
        Number of samples to validate the gPC approximation
    output_idx : ndarray, optional, default=None [1 x n_out]
        Index of output quantities to consider (if output_idx=None, all output quantities are considered)
    n_cpu : int, optional, default=1
        Number of CPU cores to use (parallel function evaluations) to evaluate original model function
    smooth_pdf : bool, optional, default: True
        Smooth probability density functions using a Savgol filter
    fn_out : str or None
        Filename of validation results and pdf plot comparing original vs gPC model
    plot : boolean, optional, default: True
        Plots the pdfs of the original model function and the gPC approximation

    Returns
    -------
    nrmsd : ndarray of float [n_out]
        Normalized root mean square deviation for all output quantities between gPC and original model
    <file> : .hdf5 file
        Data file containing the sampling points, the results and the pdfs of the original and the gpc approximation
    <file> : .pdf file
        Plot showing the pdfs of the original and the gpc approximation
    """
    if type(gpc) is list:
        gpc = gpc[output_idx]
        coeffs = coeffs[output_idx]
        output_idx_gpc = 0
    else:
        output_idx_gpc = output_idx

    if isinstance(gpc, MEGPC):
        problem = gpc.problem
    else:
        if gpc.p_matrix is not None:
            problem = gpc.problem_original
        else:
            problem = gpc.problem

    if gpc.validation is None:
        # Create sampling points
        grid_mc = RandomGrid(parameters_random=problem.parameters_random,
                             options={"n_grid": n_samples, "seed": None})

        coords_norm = grid_mc.coords_norm
        coords = grid_mc.coords

        # Evaluate original model at grid points
        com = Computation(n_cpu=n_cpu, matlab_model=gpc.matlab_model)

        y_orig = com.run(model=gpc.problem.model,
                         problem=problem,
                         coords=coords,
                         coords_norm=coords_norm,
                         i_iter=None,
                         i_subiter=None,
                         fn_results=None,
                         print_func_time=False)[:, output_idx]

    else:
        y_orig = gpc.validation.results[:, output_idx]
        coords_norm = gpc.validation.grid.coords_norm
        coords = gpc.validation.grid.coords

    if y_orig.ndim == 1:
        y_orig = y_orig[:, np.newaxis]

    # Evaluate gPC approximation at grid points
    y_gpc = gpc.get_approximation(coeffs=coeffs, x=coords_norm, output_idx=None)[:, output_idx_gpc]

    if y_gpc.ndim == 1:
        y_gpc = y_gpc[:, np.newaxis]

    # Calculate normalized root mean square deviation
    relative_error_nrmsd = nrmsd(y_gpc, y_orig)

    pdf_y_gpc, tmp = np.histogram(y_gpc[:, output_idx_gpc].flatten(), bins=100, density=True)
    pdf_x_gpc = (tmp[1:] + tmp[0:-1]) / 2.

    pdf_y_orig, tmp = np.histogram(y_orig.flatten(), bins=100, density=True)
    pdf_x_orig = (tmp[1:] + tmp[0:-1]) / 2.

    if smooth_pdf:
        pdf_y_gpc = savgol_filter(pdf_y_gpc, 51, 5)
        pdf_y_orig = savgol_filter(pdf_y_orig, 51, 5)

    # plot pdfs
    if plot:
        matplotlib.rc('text', usetex=False)
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)

        fig1, ax1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(5.5, 5))

        ax1.plot(pdf_x_gpc, pdf_y_gpc, pdf_x_orig, pdf_y_orig)
        ax1.legend([r'gpc', r'original'], fontsize=14)
        ax1.grid()
        ax1.set_xlabel(r'$y$', fontsize=16)
        ax1.set_ylabel(r'$p(y)$', fontsize=16)
        ax1.text(0.05, 0.95, r'$error=%.2f$' % (relative_error_nrmsd[0],) + "%",
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if fn_out:
        plt.savefig(os.path.splitext(fn_out)[0] + '.pdf', facecolor='#ffffff')

        # save results in .hdf5 file
        with h5py.File(os.path.splitext(fn_out)[0] + '.hdf5', 'w') as f:
            f.create_dataset('results/original', data=y_orig)
            f.create_dataset('results/gpc', data=y_gpc)
            f.create_dataset('grid/coords', data=coords)
            f.create_dataset('grid/coords_norm', data=coords_norm)
            f.create_dataset('error/nrmsd', data=relative_error_nrmsd)
            f.create_dataset('pdf/original', data=np.vstack((pdf_x_orig, pdf_y_orig)).transpose())
            f.create_dataset('pdf/gpc', data=np.vstack((pdf_x_gpc, pdf_y_gpc)).transpose())

    return relative_error_nrmsd


def validate_gpc_plot(gpc, coeffs, random_vars, n_grid=None, coords=None, output_idx=0, data_original=None,
                      fn_out=None, n_cpu=1):
    """
    Compares gPC approximation with original model function. Evaluates both at n_grid (x n_grid) sampling points and
    calculate the difference between two solutions at the output quantity with output_idx and saves the plot as
    *_QOI_idx_<output_idx>.png/pdf. Also generates one .hdf5 results file with the evaluation results.

    Parameters
    ----------
    gpc : GPC object instance
        GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out] or list of ndarray of float [n_qoi][n_coeffs x n_out]
        GPC coefficients
    random_vars: str or list of str [2]
        Names of the random variables, the analysis is performed for one or max. two random variables
    n_grid : int or list of int [2], optional
        Number of samples in each dimension to compare the gPC approximation with the original model function.
        A cartesian grid is generated based on the limits of the specified random_vars
    coords : ndarray of float [n_coords x n_dim]
        Parameter combinations for the random_vars the comparison is conducted with
    output_idx : int, optional, default=0
        Indices of output quantity to consider
    data_original: ndarray of float [n_coords x n_out], optional, default: None
        If available, data of original model function at grid, containing all QOIs
    fn_out : str, optional, default: None
        Filename of plot comparing original vs gPC model (*_QOI_idx_<output_idx>.png is added)
        Filename of .hdf5 file containing the grid and the data from the original model and the gPC
    n_cpu : int, default=1
        Number of CPU cores to use to calculate results of original model on grid.

    Returns
    -------
    <file> : .hdf5 file
        Data file containing the grid points and the results of the original and the gpc approximation
    <file> : .png and .pdf file
        Plot comparing original vs gPC model
    """
    output_idx_gpc = output_idx

    if type(gpc) is list:
        gpc = gpc[output_idx]
        coeffs = coeffs[output_idx]
        output_idx_gpc = 0

    if isinstance(gpc, MEGPC):
        problem = gpc.problem
    else:
        if gpc.p_matrix is not None:
            problem = gpc.problem_original
        else:
            problem = gpc.problem

    if type(random_vars) is not list:
        random_vars = random_vars.tolist()

    if n_grid and type(n_grid) is not list:
        n_grid = n_grid.tolist()

    # Create grid such that it includes the mean values of other random variables
    if coords is None:
        grid = np.zeros((np.prod(n_grid), len(problem.parameters_random)))
    else:
        grid = np.zeros((coords.shape[0], len(problem.parameters_random)))

    idx = []

    # sort random_vars according to gpc.parameters
    for i_p, p in enumerate(problem.parameters_random.keys()):
        if p not in random_vars:
            grid[:, i_p] = problem.parameters_random[p].mean

        else:
            idx.append(random_vars.index(p))

    random_vars = [random_vars[i] for i in idx]
    x = []

    if coords is None:
        n_grid = [n_grid[i] for i in idx]

        for i_p, p in enumerate(random_vars):
            x.append(np.linspace(problem.parameters_random[p].pdf_limits[0],
                                 problem.parameters_random[p].pdf_limits[1],
                                 n_grid[i_p]))

        coords = get_cartesian_product(x)

    else:
        for i_p, p in enumerate(random_vars):
            x.append(np.unique(coords[:, i_p]))

    grid[:, (grid == 0).all(axis=0)] = coords

    # Normalize grid
    grid_norm = Grid(parameters_random=problem.parameters_random).get_normalized_coordinates(grid)

    # Evaluate gPC expansion on grid
    y_gpc = gpc.get_approximation(coeffs=coeffs, x=grid_norm, output_idx=output_idx_gpc)

    # Evaluate original model function on grid
    if data_original is None:
        com = Computation(n_cpu=n_cpu, matlab_model=gpc.options["matlab_model"])
        y_orig_all = com.run(model=gpc.problem.model,
                             problem=problem,
                             coords=grid,
                             coords_norm=grid_norm,
                             i_iter=None,
                             i_subiter=None,
                             fn_results=None,
                             print_func_time=False)
        y_orig = y_orig_all[:, output_idx]

    else:
        y_orig_all = data_original
        y_orig = data_original[:, output_idx]

    # add axes if necessary
    if y_gpc.ndim == 1:
        y_gpc = y_gpc[:, np.newaxis]

    if y_orig.ndim == 1:
        y_orig = y_orig[:, np.newaxis]

    # Evaluate difference between original and gPC approximation
    y_dif = y_orig - y_gpc

    # save results in .hdf5 file
    if fn_out is not None:
        with h5py.File(os.path.splitext(fn_out)[0] + '.hdf5', 'w') as f:
            f.create_dataset('model_evaluations/original', data=y_orig)
            f.create_dataset('model_evaluations/original_all_qoi', data=y_orig_all)
            f.create_dataset('model_evaluations/gpc', data=y_gpc)
            f.create_dataset('model_evaluations/difference', data=y_dif)
            f.create_dataset('grid/coords', data=grid)
            f.create_dataset('grid/coords_norm', data=grid_norm)

    # Plot results
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fs = 14

    # One random variable
    if len(random_vars) == 1:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(12, 5))

        ax1.plot(coords, y_orig)
        ax1.plot(coords, y_gpc)
        ax1.legend([r"original", r"gPC"], fontsize=fs)
        ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"y(%s)" % random_vars[0], fontsize=fs)
        ax1.grid()

        ax2.plot(coords, y_dif, '--k')
        ax2.legend([r"difference"], fontsize=fs)
        ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
        ax2.grid()

    # Two random variables
    elif len(random_vars) == 2:
        fig, (ax1, ax2, ax3) = matplotlib.pyplot.subplots(nrows=1, ncols=3,
                                                          sharex='all', sharey='all',
                                                          squeeze=True, figsize=(16, 5))

        x1_2d, x2_2d = np.meshgrid(x[0], x[1])

        min_all = np.min(np.array(np.min(y_orig), np.min(y_gpc)))
        max_all = np.max(np.array(np.max(y_orig), np.max(y_gpc)))

        # Original model function
        im1 = ax1.pcolor(x1_2d, x2_2d, np.reshape(y_orig, (x[1].size, x[0].size), order='f'),
                         cmap="jet",
                         vmin=min_all,
                         vmax=max_all)
        ax1.set_title(r'Original model', fontsize=fs)
        ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"%s" % random_vars[1], fontsize=fs)

        # gPC approximation
        # Original model function
        im2 = ax2.pcolor(x1_2d, x2_2d, np.reshape(y_gpc, (x[1].size, x[0].size), order='f'),
                         cmap="jet",
                         vmin=min_all,
                         vmax=max_all)
        ax2.set_title(r'gPC approximation', fontsize=fs)
        ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"%s" % random_vars[1], fontsize=fs)

        # Difference
        min_dif = np.min(y_dif)
        max_dif = np.max(y_dif)
        b2rcw_cmap = make_cmap(b2rcw(min_dif, max_dif))

        im3 = ax3.pcolor(x1_2d, x2_2d, np.reshape(y_dif, (x[1].size, x[0].size), order='f'),
                         cmap=b2rcw_cmap,
                         vmin=min_dif,
                         vmax=max_dif)
        ax3.set_title(r'Difference (Original vs gPC)', fontsize=fs)
        ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"%s" % random_vars[1], fontsize=fs)

        fig.colorbar(im1, ax=ax1, orientation='vertical')
        fig.colorbar(im2, ax=ax2, orientation='vertical')
        fig.colorbar(im3, ax=ax3, orientation='vertical')

        plt.tight_layout()

    if fn_out is not None:
        plt.savefig(os.path.splitext(fn_out)[0] + "_qoi_" + str(output_idx) + '.png', dpi=600)
        plt.savefig(os.path.splitext(fn_out)[0] + "_qoi_" + str(output_idx) + '.pdf')


def plot_gpc(gpc, coeffs, random_vars, n_grid=None, coords=None, output_idx=0, fn_out=None):
    """
    Compares gPC approximation with original model function. Evaluates both at n_grid (x n_grid) sampling points and
    calculate the difference between two solutions at the output quantity with output_idx and saves the plot as
    *_QOI_idx_<output_idx>.png/pdf. Also generates one .hdf5 results file with the evaluation results.

    Parameters
    ----------
    gpc : GPC object instance
        GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out]
        GPC coefficients
    random_vars: str or list of str [2]
        Names of the random variables, the analysis is performed for one or max. two random variables
    n_grid : int or list of int [2], optional
        Number of samples in each dimension to compare the gPC approximation with the original model function.
        A cartesian grid is generated based on the limits of the specified random_vars
    coords : ndarray of float [n_coords x n_dim]
        Parameter combinations for the random_vars the comparison is conducted with
    output_idx : int, optional, default=0
        Indices of output quantity to consider
    fn_out : str
        Filename of plot comparing original vs gPC model (*_QOI_idx_<output_idx>.png is added)

    Returns
    -------
    <file> : .hdf5 file
        Data file containing the grid points and the results of the original and the gpc approximation
    <file> : .png and .pdf file
        Plot comparing original vs gPC model
    """
    output_idx_gpc = output_idx

    if type(gpc) is list:
        gpc = gpc[output_idx]
        coeffs = coeffs[output_idx]
        output_idx_gpc = 0

    if isinstance(gpc, MEGPC):
        problem = gpc.problem
    else:
        if gpc.p_matrix is not None:
            problem = gpc.problem_original
        else:
            problem = gpc.problem

    if type(random_vars) is not list:
        random_vars = random_vars.tolist()

    if n_grid and type(n_grid) is not list:
        n_grid = n_grid.tolist()

    # Create grid such that it includes the mean values of other random variables
    if coords is None:
        grid = np.zeros((np.prod(n_grid), len(problem.parameters_random)))
    else:
        grid = np.zeros((coords.shape[0], len(problem.parameters_random)))

    idx = []

    # sort random_vars according to gpc.parameters
    for i_p, p in enumerate(problem.parameters_random.keys()):
        if p not in random_vars:
            grid[:, i_p] = problem.parameters_random[p].mean

        else:
            idx.append(random_vars.index(p))

    random_vars = [random_vars[i] for i in idx]
    x = []

    if coords is None:
        n_grid = [n_grid[i] for i in idx]

        for i_p, p in enumerate(random_vars):
            x.append(np.linspace(problem.parameters_random[p].pdf_limits[0],
                                 problem.parameters_random[p].pdf_limits[1],
                                 n_grid[i_p]))

        coords = get_cartesian_product(x)

    else:
        for i_p, p in enumerate(random_vars):
            x.append(np.unique(coords[:, i_p]))

    grid[:, (grid == 0).all(axis=0)] = coords

    # Normalize grid
    grid_norm = Grid(parameters_random=problem.parameters_random).get_normalized_coordinates(grid)

    # Evaluate gPC expansion on grid
    y_gpc = gpc.get_approximation(coeffs=coeffs, x=grid_norm, output_idx=output_idx_gpc)

    # add axes if necessary
    if y_gpc.ndim == 1:
        y_gpc = y_gpc[:, np.newaxis]

    # Plot results
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fs = 14

    # One random variable
    if len(random_vars) == 1:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(12, 5))

        ax1.plot(coords, y_gpc)
        ax1.legend([r"gPC"], fontsize=fs)
        ax1.set_xlabel(r"$%s$" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"$y(%s)$" % random_vars[0], fontsize=fs)
        ax1.grid()

    # Two random variables
    elif len(random_vars) == 2:
        fig, (ax1) = matplotlib.pyplot.subplots(nrows=1, ncols=1,
                                                sharex='all', sharey='all',
                                                squeeze=True, figsize=(4, 5))

        x1_2d, x2_2d = np.meshgrid(x[0], x[1])

        min_all = np.min(y_gpc)
        max_all = np.max(y_gpc)

        # gPC approximation
        im1 = ax1.pcolor(x1_2d, x2_2d, np.reshape(y_gpc, (x[1].size, x[0].size), order='f'),
                         cmap="jet",
                         vmin=min_all,
                         vmax=max_all)
        ax1.set_title(r'gPC approximation', fontsize=fs)
        ax1.set_xlabel(r"$%s$" % random_vars[0], fontsize=fs)
        ax1.set_ylabel(r"$%s$" % random_vars[1], fontsize=fs)

        fig.colorbar(im1, ax=ax1, orientation='vertical')

        plt.tight_layout()

    if fn_out:
        plt.savefig(os.path.splitext(fn_out)[0] + "_qoi_" + str(output_idx) + '.png', dpi=600)
        plt.savefig(os.path.splitext(fn_out)[0] + "_qoi_" + str(output_idx) + '.pdf')
