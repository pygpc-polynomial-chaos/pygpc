import h5py
import numpy as np
from .io import read_gpc_pkl


def get_sensitivities_hdf5(fn_gpc, output_idx=False, calc_sobol=True, calc_global_sens=False, calc_pdf=False):
    """
    Post-processes the gPC expansion and adds mean, standard deviation, relative standard deviation, variance, Sobol
    indices, global derivative based sensitivity coefficients and probability density functions of output quantities to
    .hdf5 file of gPC.

    Parameters:
    -----------
    fn_gpc : str
        Filename of gPC .pkl object and corresponding .hdf5 results file (without file extension)
        (e.g. .../foo/gpc)
    output_idx : nparray of int
        Indices of output quantities (QOIs) to consider in postprocessing (default: all)
    calc_sobol : bool
        Calculate Sobol indices (default: True)
    calc_global_sens : bool
        Calculate global derivative based sensitivities (default: False)
    calc_pdf : bool
        Calculate probability density functions of output quantities (default: False)

    Returns:
    --------
    <File> : .hdf5
        Adds datasets "sens/..." to the gPC .hdf5 file

    Example
    -------
    The content of .hdf5 files can be shown using the tool HDFView
    (https://support.hdfgroup.org/products/java/hdfview/)
    ::
        sens
        I---/mean               [n_qoi]                 Mean of QOIs
        I---/std                [n_qoi]                 Standard deviation of QOIs
        I---/rstd               [n_qoi]                 Relative standard deviation of QOIs
        I---/var                [n_qoi]                 Variance of QOIs
        I---/sobol              [n_sobol x n_qoi]       Sobol indices (all available orders)
        I---/sobol_idx_bool     [n_sobol x n_dim]       Corresponding parameter (combinations) of Sobol indices
        I---/global_sens        [n_dim x n_qoi]         Global derivative based sensitivity coefficients
        I---/pdf_x              [100 x n_qoi]           x-axis values of output PDFs
        I---/pdf_y              [100 x n_qoi]           y-axis values of output PDFs
    """
    print("> Loading gpc object: {}".format(fn_gpc + ".pkl"))
    gpc = read_gpc_pkl(fn_gpc + ".pkl")

    print("> Loading gpc coeffs: {}".format(fn_gpc + ".hdf5"))
    with h5py.File(fn_gpc + ".hdf5", 'r') as f:
        coeffs = np.array(f['coeffs'][:])

    if not output_idx:
        output_idx = np.arange(coeffs.shape[1])

    # determine mean
    mean = gpc.get_mean(coeffs[:, output_idx])

    # determine standard deviation
    std = gpc.get_standard_deviation(coeffs[:, output_idx])

    # determine relative standard deviation
    rstd = std / mean

    # determine variance
    var = std ** 2

    # determine Sobol indices
    if calc_sobol:
        sobol, sobol_idx, sobol_idx_bool = gpc.get_sobol_indices(coeffs[:, output_idx])

    # determine global derivative based sensitivity coefficients
    if calc_global_sens:
        global_sens = gpc.get_global_sens(coeffs[:, output_idx])

    # determine pdfs
    if calc_pdf:
        pdf_x, pdf_y = gpc.get_pdf(coeffs, n_samples=1E3, output_idx=output_idx)

    print("> Adding results to: {}".format(fn_gpc + ".hdf5"))
    # save results in .hdf5 file (overwrite existing quantities in sens/...)
    with h5py.File(fn_gpc + ".hdf5", 'a') as f:

        try:
            del f["sens"]
        except KeyError:
            pass

        f.create_dataset(data=mean, name="sens/mean")
        f.create_dataset(data=std, name="sens/std")
        f.create_dataset(data=rstd, name="sens/rstd")
        f.create_dataset(data=var, name="sens/var")

        if calc_sobol:
            f.create_dataset(data=sobol, name="sens/sobol")
            f.create_dataset(data=sobol_idx_bool, name="sens/sobol_idx_bool")

        if calc_global_sens:
            f.create_dataset(data=global_sens, name="sens/global_sens")

        if calc_pdf:
            f.create_dataset(data=pdf_x, name="sens/pdf_x")
            f.create_dataset(data=pdf_y, name="sens/pdf_y")
