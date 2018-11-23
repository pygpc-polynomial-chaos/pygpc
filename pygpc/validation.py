from Computation import *
from .Grid import *
from .misc import get_normalized_rms_deviation
import matplotlib.pyplot as plt
import os
import scipy.stats


def validate_gpc(gpc, coeffs, n_samples=1e4, output_idx=0, fn_pdf=None):
    """
    Compared gPC approximation with original model function. Evaluates both at "n_samples" sampling points and
    evaluates the root mean square deviation. It also computes the pdf ot the output quantity with output_idx
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
