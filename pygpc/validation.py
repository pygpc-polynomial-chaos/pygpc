from Computation import *
from .Grid import *
from .misc import get_normalized_rms_deviation
from .misc import get_cartesian_product
from .Visualization import *
import matplotlib.pyplot as plt
import os
import scipy.stats
import matplotlib


def validate_gpc(gpc, coeffs, n_samples=1e4, output_idx=0, fn_pdf=None):
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
    fn_pdf : str
        Filename of pdf plot comparing original vs gPC model

    Returns
    -------
    nrmsd : ndarray of float [n_out]
        Normalized root mean square deviation for all output quantities between gPC and original model
    """

    # Create sampling points
    grid_mc = RandomGrid(problem=gpc.problem, parameters={"n_grid": n_samples, "seed": None})

    # Evaluate gPC approximation at grid points
    y_gpc = gpc.get_approximation(coeffs, grid_mc.coords_norm, output_idx=None)

    if y_gpc.ndim == 1:
        y_gpc = y_gpc[:, np.newaxis]

    # Evaluate original model at grid points
    com = Computation(n_cpu=gpc.n_cpu)
    y_orig = com.run(model=gpc.problem.model, problem=gpc.problem, coords=grid_mc.coords)

    if y_orig.ndim == 1:
        y_orig = y_orig[:, np.newaxis]

    # Calculate normalized root mean square deviation
    nrmsd = get_normalized_rms_deviation(y_gpc, y_orig)

    if fn_pdf:
        # Calculating output PDFs
        kde_gpc = scipy.stats.gaussian_kde(y_gpc[:, output_idx].flatten(), bw_method=0.15 / y_gpc.std(ddof=1))
        pdf_x_gpc = np.linspace(y_gpc.min(), y_gpc.max(), 100)
        pdf_y_gpc = kde_gpc(pdf_x_gpc)
        kde_orig = scipy.stats.gaussian_kde(y_orig[:, output_idx].flatten(), bw_method=0.15 / y_orig.std(ddof=1))
        pdf_x_orig = np.linspace(y_orig.min(), y_orig.max(), 100)
        pdf_y_orig = kde_orig(pdf_x_orig)

        # plot pdfs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(pdf_x_gpc, pdf_y_gpc, pdf_x_orig, pdf_y_orig)
        plt.legend(['gpc', 'original'])
        plt.grid()
        plt.title(os.path.split(os.path.splitext(fn_pdf)[0])[1], fontsize=10)
        plt.xlabel('y', fontsize=12)
        plt.ylabel('p(y)', fontsize=12)
        ax.text(0.05, 0.95, r'$error=%.2f$' % (nrmsd[0],) + "%",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.savefig(os.path.splitext(fn_pdf)[0] + '.pdf', facecolor='#ffffff')

    return nrmsd


def validate_gpc_2d(gpc, coeffs, random_vars, n_grid=None, grid=None, output_idx=0, data_original=None, fn_plot=None):
    """
    Compares gPC approximation with original model function. Evaluates both at n_grid (x n_grid) sampling points and
    calculate the difference between two solutions at the output quantity with output_idx and saves the plot as
    fn_plot.png and fn_plot.pdf.

    Parameters
    ----------
    gpc : GPC object instance
        GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out]
        GPC coefficients
    random_vars: str or list of str [2]
        Names of the random variables, the plot is generated for (one or max. two)
    n_grid : int or list of int [2], optional
        Number of samples in each dimension to compare the gPC approximation with the original model function.
        A cartesian grid is generated based on the limits of the specified random_vars
    grid : ndarray of float [n_grid x 2]
        Cartesian grid, the comparison is conducted with
    output_idx : int, optional, default=0
        Index of output quantity to consider
    data_original: ndarray of float, optional, default: None
        If available, data of original model function at grid
    fn_plot : str
        Filename of plot comparing original vs gPC model

    Returns
    -------
    <file> : .png and .pdf file
        Plot comparing original vs gPC model
    """

    # Generate grid points
    if grid is None:
        x = []
        for i, r in enumerate(random_vars):
            x.append(np.linspace(gpc.problem.parameters[r].pdf_limits[0],
                                 gpc.problem.parameters[r].pdf_limits[1],
                                 n_grid[i]))

        grid = get_cartesian_product(x)

    grid_norm = Grid(problem=gpc.problem).get_normalized_coordinates(grid)

    # Evaluate gpc expansion on grid
    y_gpc = gpc.get_approximation(coeffs=coeffs, x=grid_norm, output_idx=output_idx)

    # Evaluate original model function on grid
    if data_original is None:
        com = Computation(n_cpu=gpc.n_cpu)
        y_orig = com.run(model=gpc.problem.model, problem=gpc.problem, coords=grid)
    else:
        y_orig = data_original

    # Evaluate difference between original and gPC approximation
    y_dif = y_orig - y_gpc

    # Plot results
    if len(random_vars) == 2:
        from matplotlib import rc
        rc('text', usetex=True)
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)
        fig, (ax1, ax2, ax3) = matplotlib.pyplot.subplots(nrows=1, ncols=3,
                                                          sharex='all', sharey='all',
                                                          squeeze=True, figsize=(8, 18))
        fs = 13

        x1_2d, x2_2d = np.meshgrid(x[0], x[1])

        min_all = np.min(np.array(np.min(y_orig), np.min(y_gpc)))
        max_all = np.max(np.array(np.max(y_orig), np.max(y_gpc)))

        # Original model function
        im1 = ax1.pcolor(x1_2d, x2_2d, np.reshape(y_orig[:, output_idx], (np.unique(grid[:, 1]).shape[0],
                                                                          np.unique(grid[:, 0]).shape[0])),
                         cmap="jet",
                         vmin=min_all,
                         vmax=max_all)
        ax1.set_title('Original model', fontsize=fs)
        ax1.set_xlabel(random_vars[0], fontsize=fs)
        ax1.set_ylabel(random_vars[1], fontsize=fs)

        # gPC approximation
        # Original model function
        im2 = ax2.pcolor(x1_2d, x2_2d, np.reshape(y_gpc[:, output_idx], (np.unique(grid[:, 1]).shape[0],
                                                                         np.unique(grid[:, 0]).shape[0])),
                         cmap="jet",
                         vmin=min_all,
                         vmax=max_all)
        ax2.set_title('gPC approximation', fontsize=fs)
        ax2.set_xlabel(random_vars[0], fontsize=fs)
        ax2.set_ylabel(random_vars[1], fontsize=fs)

        # Difference
        min_dif = np.min(y_dif)
        max_dif = np.max(y_dif)
        b2rcw_cmap = make_cmap(b2rcw(min_dif, max_dif))

        im3 = ax3.pcolor(x1_2d, x2_2d, np.reshape(y_dif[:, output_idx], (np.unique(grid[:, 1]).shape[0],
                                                                         np.unique(grid[:, 0]).shape[0])),
                         cmap=b2rcw_cmap,
                         vmin=min_dif,
                         vmax=max_dif)
        ax3.set_title('Difference (Original vs gPC)', fontsize=fs)
        ax3.set_xlabel(random_vars[0], fontsize=fs)
        ax3.set_ylabel(random_vars[1], fontsize=fs)

        fig.colorbar(im1, ax=ax1, orientation='vertical')
        fig.colorbar(im2, ax=ax2, orientation='vertical')
        fig.colorbar(im3, ax=ax3, orientation='vertical')

        plt.tight_layout()
        plt.show()

        plt.savefig(os.path.splitext(fn_plot)[0] + '.png', dpi=600)
        plt.savefig(os.path.splitext(fn_plot)[0] + '.pdf')
