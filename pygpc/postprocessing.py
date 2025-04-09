import h5py
import numpy as np

from .Grid import *
from .MEGPC import *
from .io import read_session

try:
    import matplotlib
except ImportError:
    pass

def get_sensitivities_hdf5(fn_gpc, output_idx=False, calc_sobol=True, calc_global_sens=False, calc_pdf=False,
                           algorithm="standard", n_samples=1e5):
    """
    Post-processes the gPC expansion from the gPC coefficients (standard) or by sampling. Adds mean,
    standard deviation, relative standard deviation, variance, Sobol indices, global derivative based
    sensitivity coefficients and probability density functions of output quantities to .hdf5 file of gPC.

    Parameters
    ----------
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
    algorithm : str, optional, default: "standard"
        Algorithm to determine the Sobol indices
        - "standard": Sobol indices are determined from the gPC coefficients
        - "sampling": Sobol indices are determined from sampling using Saltelli's Sobol sampling sequence [1, 2, 3]
    n_samples : int, optional, default: 1e5
        Number of samples to determine Sobol indices by sampling. The efficient number of samples
        increases to n_samples * (2*dim + 2) in Saltelli's Sobol sampling sequence.

    Returns
    -------
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
    sobol = None
    sobol_idx_bool = None
    global_sens = None
    pdf_x = None
    pdf_y = None
    grid = None
    res = None

    with h5py.File(fn_gpc + ".hdf5", 'r') as f:
        # filename of associated gPC .pkl files
        fn_session = os.path.join(os.path.split(fn_gpc)[0], f["misc/fn_session"][0].astype(str))
        fn_session_folder = f["misc/fn_session_folder"][0].astype(str)

    print("> Loading gpc session object: {}".format(fn_session))
    session = read_session(fname=fn_session, folder=fn_session_folder)

    with h5py.File(fn_gpc + ".hdf5", 'r') as f:
        # check if we have qoi specific gPCs here
        try:
            if "qoi" in list(f["coeffs"].keys())[0]:
                qoi_keys = list(f["coeffs"].keys())

        except AttributeError:
            pass

        # load coeffs depending on gpc type
        print("> Loading gpc coeffs: {}".format(fn_gpc + ".hdf5"))

        if not session.qoi_specific and not session.gpc_type == "megpc":
            coeffs = np.array(f['coeffs'][:])

            if not output_idx:
                output_idx = np.arange(coeffs.shape[1])

        elif not session.qoi_specific and session.gpc_type == "megpc":

            coeffs = [None for _ in range(len(list(f['coeffs'].keys())))]

            for i, d in enumerate(list(f['coeffs'].keys())):
                coeffs[i] = np.array(f['coeffs/' + str(d)])[:]

            if not output_idx:
                output_idx = np.arange(coeffs[0].shape[1])

        elif session.qoi_specific and not session.gpc_type == "megpc":

            algorithm = "sampling"
            coeffs = dict()

            for key in qoi_keys:
                coeffs[key] = np.array(f['coeffs/' + key])[:]

            if not output_idx:
                output_idx = np.arange(len(list(coeffs.keys())))

        elif session.qoi_specific and session.gpc_type == "megpc":

            algorithm = "sampling"
            coeffs = dict()

            for key in qoi_keys:
                coeffs[key] = [None for _ in range(len(list(f['coeffs/' + key].keys())))]

                for i, d in enumerate(list(f['coeffs/' + key].keys())):
                    coeffs[key][i] = np.array(f['coeffs/' + key + "/" + str(d)])[:]

            if not output_idx:
                output_idx = np.arange(len(list(coeffs.keys())))

        dim = f["grid/coords"][:].shape[1]

    # generate samples in case of sampling approach
    if algorithm == "sampling":
        grid = Random(parameters_random=session.parameters_random,
                      n_grid=n_samples,
                      options=None)

    # start prostprocessing depending on gPC type
    if not session.qoi_specific and not session.gpc_type == "megpc":

        coeffs = coeffs[:, output_idx]

        if algorithm == "standard":
            # determine mean
            mean = session.gpc[0].get_mean(coeffs=coeffs)

            # determine standard deviation
            std = session.gpc[0].get_std(coeffs=coeffs)

        elif algorithm == "sampling":
            # run model evaluations
            res = session.gpc[0].get_approximation(coeffs=coeffs, x=grid.coords_norm)

            # determine mean
            mean = session.gpc[0].get_mean(samples=res)

            # determine standard deviation
            std = session.gpc[0].get_std(samples=res)

        else:
            raise AssertionError("Please provide valid algorithm argument (""standard"" or ""sampling"")")

        # determine Sobol indices
        if calc_sobol:
            sobol, sobol_idx, sobol_idx_bool = session.gpc[0].get_sobol_indices(coeffs=coeffs,
                                                                                algorithm=algorithm,
                                                                                n_samples=n_samples)

        # determine global derivative based sensitivity coefficients
        if calc_global_sens:
            global_sens = session.gpc[0].get_global_sens(coeffs=coeffs,
                                                         algorithm=algorithm,
                                                         n_samples=n_samples)

        # determine pdfs
        if calc_pdf:
            pdf_x, pdf_y = session.gpc[0].get_pdf(coeffs, n_samples=n_samples, output_idx=output_idx)

    elif session.qoi_specific and not session.gpc_type == "megpc":

        gpc = []
        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        res = np.zeros((grid.coords_norm.shape[0], len(output_idx)))

        # loop over qoi (there may be different approximations and projections)
        for i, idx in enumerate(output_idx):

            # run model evaluations
            res[:, i] = session.gpc[idx].get_approximation(coeffs=coeffs["qoi_" + str(idx)],
                                                           x=grid.coords_norm).flatten()

            # determine Sobol indices
            if calc_sobol:
                sobol_qoi, sobol_idx_qoi, sobol_idx_bool_qoi = \
                    session.gpc[idx].get_sobol_indices(coeffs=coeffs["qoi_" + str(idx)],
                                                       algorithm=algorithm,
                                                       n_samples=n_samples)

            # rearrange sobol indices according to first qoi (reference)
            # (they are sorted w.r.t. highest contribution and this may change between qoi)
            if i == 0:
                sobol = copy.deepcopy(sobol_qoi)
                sobol_idx = copy.deepcopy(sobol_idx_qoi)
                sobol_idx_bool = copy.deepcopy(sobol_idx_bool_qoi)

            else:
                sobol = np.hstack((sobol, np.zeros(sobol.shape)))
                sobol_sort_idx = []
                for ii, s_idx in enumerate(sobol_idx):
                    for jj, s_idx_qoi in enumerate(sobol_idx_qoi):
                        if (s_idx == s_idx_qoi).all():
                            sobol_sort_idx.append(jj)

                for jj, s in enumerate(sobol_sort_idx):
                    sobol[jj, i] = sobol_qoi[s]

            # determine global derivative based sensitivity coefficients
            if calc_global_sens:
                global_sens[:, ] = session.gpc[idx].get_global_sens(coeffs=coeffs["qoi_" + str(idx)],
                                                                    algorithm=algorithm,
                                                                    n_samples=n_samples)

            # determine pdfs
            if calc_pdf:
                pdf_x[:, ], pdf_y[:, ] = session.gpc[idx].get_pdf(coeffs=coeffs["qoi_" + str(idx)],
                                                                  n_samples=n_samples,
                                                                  output_idx=0)

        # determine mean
        mean = gpc.get_mean(samples=res)

        # determine standard deviation
        std = gpc.get_std(samples=res)

    elif not session.qoi_specific and session.gpc_type == "megpc":

        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        if algorithm == "sampling":

            # run model evaluations
            res = session.gpc[0].get_approximation(coeffs=coeffs, x=grid.coords_norm)

            # determine Sobol indices
            if calc_sobol:
                sobol, sobol_idx, sobol_idx_bool = session.gpc[0].get_sobol_indices(coeffs=coeffs,
                                                                                    n_samples=n_samples)

            # determine global derivative based sensitivity coefficients
            if calc_global_sens:
                global_sens[:, ] = session.gpc[0].get_global_sens(coeffs=coeffs,
                                                                  n_samples=n_samples)

            # determine pdfs
            if calc_pdf:
                pdf_x[:, ], pdf_y[:, ] = session.gpc[0].get_pdf(coeffs=coeffs,
                                                                n_samples=n_samples,
                                                                output_idx=output_idx)

            # determine mean
            mean = session.gpc[0].get_mean(samples=res)

            # determine standard deviation
            std = session.gpc[0].get_std(samples=res)

        else:
            raise AssertionError("Please use ""sampling"" algorithm in case of multi-element gPC!")

    elif session.qoi_specific and session.gpc_type == "megpc":

        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        if algorithm == "sampling":

            res = np.zeros((grid.coords_norm.shape[0], len(output_idx)))

            # loop over qoi (there may be different approximations and projections)
            for i, idx in enumerate(output_idx):

                # run model evaluations
                res[:, i] = session.gpc[idx].get_approximation(coeffs=coeffs["qoi_" + str(idx)],
                                                               x=grid.coords_norm).flatten()

                # determine Sobol indices
                if calc_sobol:
                    sobol_qoi, sobol_idx_qoi, sobol_idx_bool_qoi = \
                        session.gpc[idx].get_sobol_indices(coeffs=coeffs["qoi_" + str(idx)],
                                                           n_samples=n_samples)

                # rearrange sobol indices according to first qoi (reference)
                # (they are sorted w.r.t. highest contribution and this may change between qoi)
                if i == 0:
                    sobol = copy.deepcopy(sobol_qoi)
                    sobol_idx = copy.deepcopy(sobol_idx_qoi)
                    sobol_idx_bool = copy.deepcopy(sobol_idx_bool_qoi)

                else:
                    sobol = np.hstack((sobol, np.zeros((sobol.shape[0], 1))))
                    sobol_sort_idx = []
                    for ii, s_idx in enumerate(sobol_idx):
                        for jj, s_idx_qoi in enumerate(sobol_idx_qoi):
                            if (s_idx == s_idx_qoi).all():
                                sobol_sort_idx.append(jj)

                    for jj, s in enumerate(sobol_sort_idx):
                        sobol[jj, i] = sobol_qoi[s]

                # determine global derivative based sensitivity coefficients
                if calc_global_sens:
                    global_sens[:, ] = session.gpc[idx].get_global_sens(coeffs=coeffs["qoi_" + str(idx)],
                                                                        n_samples=n_samples)

                # determine pdfs
                if calc_pdf:
                    pdf_x[:, ], pdf_y[:, ] = session.gpc[idx].get_pdf(coeffs=coeffs["qoi_" + str(idx)],
                                                                      n_samples=n_samples,
                                                                      output_idx=0)

            # determine mean
            mean = session.gpc[0].get_mean(samples=res)

            # determine standard deviation
            std = session.gpc[0].get_std(samples=res)

        else:
            raise AssertionError("Please use ""sampling"" algorithm in case of multi-element gPC!")

    # determine relative standard deviation
    rstd = std / mean

    # determine variance
    var = std ** 2

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

        if algorithm == "sampling":
            f.create_dataset(data=grid.coords_norm, name="sens/coords_norm")
            f.create_dataset(data=res, name="sens/res")

        if calc_sobol:
            f.create_dataset(data=sobol * var, name="sens/sobol")
            f.create_dataset(data=sobol, name="sens/sobol_norm")
            f.create_dataset(data=sobol_idx_bool, name="sens/sobol_idx_bool")

        if calc_global_sens:
            f.create_dataset(data=global_sens, name="sens/global_sens")

        if calc_pdf:
            f.create_dataset(data=pdf_x, name="sens/pdf_x")
            f.create_dataset(data=pdf_y, name="sens/pdf_y")


def get_extracted_sobol_order(sobol, sobol_idx_bool, order=1):
    """
    Extract Sobol indices with specified order from Sobol data.

    Parameters
    ----------
    sobol: ndarray of float [n_sobol x n_out]
        Sobol indices of n_out output quantities
    sobol_idx_bool: list of ndarray of bool
        Boolean mask which contains unique multi indices.
    order: int, optional, default=1
        Sobol index order to extract

    Returns
    -------
    sobol_n_order: ndarray of float [n_out]
        n-th order Sobol indices of n_out output quantities
    sobol_idx_n_order: ndarray of int
        Parameter label indices belonging to n-th order Sobol indices
    """

    sobol_idx = [np.argwhere(sobol_idx_bool[i, :]).flatten() for i in range(sobol_idx_bool.shape[0])]

    # make mask of nth order sobol indices
    mask = [index for index, sobol_element in enumerate(sobol_idx) if sobol_element.shape[0] == order]

    # extract from dataset
    sobol_n_order = sobol[mask, :]
    sobol_idx_n_order = np.array([sobol_idx[m] for m in mask])

    # sort sobol indices according to parameter indices in ascending order
    sort_idx = np.argsort(sobol_idx_n_order, axis=0)[:, 0]
    sobol_n_order = sobol_n_order[sort_idx, :]
    sobol_idx_n_order = sobol_idx_n_order[sort_idx, :]

    return sobol_n_order, sobol_idx_n_order


def get_sobol_composition(sobol, sobol_idx_bool, random_vars=None, verbose=False):
    """
    Determine average ratios of Sobol indices over all output quantities:
    (i) over all orders and (e.g. 1st: 90%, 2nd: 8%, 3rd: 2%)
    (ii) for the 1st order indices w.r.t. each random variable. (1st: x1: 50%, x2: 40%)


    Parameters
    ----------
    sobol : ndarray of float [n_sobol x n_out]
        Unnormalized sobol_indices
    sobol_idx_bool : list of ndarray of bool
        Boolean mask which contains unique multi indices.
    random_vars : list of str
        Names of random variables in the order as they appear in the OrderedDict from the Problem class
    verbose : boolean, optional, default=True
        Print output info

    Returns
    -------
    sobol_rel_order_mean: ndarray of float [n_out]
        Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,),
        (over all output quantities)
    sobol_rel_order_std: ndarray of float [n_out]
        Standard deviation of the proportion of the Sobol indices of the different order to the total variance
        (1st, 2nd, etc..,), (over all output quantities)
    sobol_rel_1st_order_mean: ndarray of float [n_out]
        Average proportion of the random variables of the 1st order Sobol indices to the total variance,
        (over all output quantities)
    sobol_rel_1st_order_std: ndarray of float [n_out]
        Standard deviation of the proportion of the random variables of the 1st order Sobol indices to the total
        variance
        (over all output quantities)
    sobol_rel_2nd_order_mean: ndarray of float [n_out]
        Average proportion of the random variables of the 2nd order Sobol indices to the total variance,
        (over all output quantities)
    sobol_rel_2nd_order_std: ndarray of float [n_out]
        Standard deviation of the proportion of the random variables of the 2nd order Sobol indices to the total
        variance
        (over all output quantities)
    """

    # sobol_idx = [np.argwhere(sobol_idx_bool[i, :]).flatten() for i in range(sobol_idx_bool.shape[0])]

    # get max order
    order_max = np.max(np.sum(sobol_idx_bool, axis=1))

    # total variance
    var = np.sum(sobol, axis=0).flatten()

    # get NaN values
    not_nan_mask = np.logical_not(np.isnan(var))

    sobol_rel_order_mean = []
    sobol_rel_order_std = []
    sobol_rel_1st_order_mean = []
    sobol_rel_1st_order_std = []
    sobol_rel_2nd_order_mean = []
    sobol_rel_2nd_order_std = []
    str_out = []

    # get maximum length of random_vars label
    if random_vars is not None:
        max_len = max([len(p) for p in random_vars])

    for i in range(order_max):
        # extract sobol coefficients of order i
        sobol_extracted, sobol_extracted_idx = get_extracted_sobol_order(sobol, sobol_idx_bool, i + 1)

        # determine average sobol index over all elements
        sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten()) /
                                    np.sum(var[not_nan_mask]))

        sobol_rel_order_std.append(np.std(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten() /
                                          var[not_nan_mask]))

        iprint("Ratio: Sobol indices order {} / total variance: {:.4f} +- {:.4f}"
               .format(i+1, sobol_rel_order_mean[i], sobol_rel_order_std[i]), tab=0, verbose=verbose)

        # for 1st order indices, determine ratios of all random variables
        if i == 0:
            sobol_extracted_idx_1st = sobol_extracted_idx[:]
            for j in range(sobol_extracted.shape[0]):
                sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                / np.sum(var[not_nan_mask]))
                sobol_rel_1st_order_std.append(0)

                if random_vars is not None:
                    str_out.append("\t{}{}: {:.4f}"
                                   .format((max_len - len(random_vars[sobol_extracted_idx_1st[j][0]])) * ' ',
                                           random_vars[sobol_extracted_idx_1st[j][0]],
                                           sobol_rel_1st_order_mean[j]))

        # for 2nd order indices, determine ratios of all random variables
        if i == 1:
            for j in range(sobol_extracted.shape[0]):
                sobol_rel_2nd_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                / np.sum(var[not_nan_mask]))
                sobol_rel_2nd_order_std.append(0)

    sobol_rel_order_mean = np.array(sobol_rel_order_mean)
    sobol_rel_1st_order_mean = np.array(sobol_rel_1st_order_mean)
    sobol_rel_2nd_order_mean = np.array(sobol_rel_2nd_order_mean)

    # print output of 1st order Sobol indices ratios of parameters
    if verbose:
        for j in range(len(str_out)):
            print(str_out[j])

    return sobol_rel_order_mean, sobol_rel_order_std, \
           sobol_rel_1st_order_mean, sobol_rel_1st_order_std, \
           sobol_rel_2nd_order_mean, sobol_rel_2nd_order_std


def get_sens_summary(fn_gpc, parameters_random, fn_out=None):
    """
    Print summary of Sobol indices and global derivative based sensitivity coefficients

    Parameters
    ----------
    fn_gpc : str
        Filename of gpc results file (without .hdf5 extension)
    parameters_random: OrderedDict containing the RandomParameter class instances
        Dictionary (ordered) containing the properties of the random parameters
    fn_out : str
        Filename of output .txt file containing the Sobol coefficient summary

    Returns
    -------
    sobol : OrderedDict
        OrderedDict containing the normalized Sobol indices by significance
    gsens : OrderedDict
        OrderedDict containing the global derivative based sensitivity coefficients
    """

    parameter_names = list(parameters_random.keys())

    with h5py.File(fn_gpc + ".hdf5", "r") as f:
        sobol_idx_bool = f["/sens/sobol_idx_bool"][:]
        sobol_norm = f["/sens/sobol_norm"][:]
        global_sens = f["/sens/global_sens"][:]

    global_sens_sort_idx = np.flip(np.argsort(global_sens[:, 0]))

    sobol_dict = OrderedDict()
    p_length = []
    params = []

    # Extract Sobol coefficients
    for i_s, s in enumerate(sobol_norm):
        params.append([p for i_p, p in enumerate(parameter_names) if sobol_idx_bool[i_s, i_p]])
        sobol_dict[str(params[-1])] = s
        p_length.append(len(str(params[-1])))

    len_max = np.max(p_length)

    # Extract global derivative sensitivity coefficients
    gsens_dict = OrderedDict()
    for i_p, p in enumerate(parameter_names):
        gsens_dict[p] = global_sens[i_p, :]

    # write output file
    if fn_out:
        # Sobol indices
        sep = " " * 4
        sobol_text = []
        sobol_text.append("Normalized Sobol indices:\n")
        sobol_text.append("=" * (len_max + 10) + "\n")

        for i_p, p in enumerate(params):
            len_diff = len_max - p_length[i_p]
            sobol_text.append(f"{str(p)}: " + " " * len_diff)
            for i_qoi in range(sobol_norm.shape[1]):
                sobol_text[-1] += sep + f"{sobol_norm[i_p, i_qoi]:.6f}"
            sobol_text.append("\n")

        len_max_sobol = np.max([len(line) for line in sobol_text])
        sobol_text[1] = "=" * (len_max_sobol) + "\n"

        # Derivative based sensitivity coefficients
        gsens_text = []
        gsens_text.append("Average derivatives:\n")
        gsens_text.append("=" * (len_max + 10) + "\n")

        for i_p, p in enumerate(parameter_names):
            len_diff = len_max - len(parameter_names[global_sens_sort_idx[i_p]]) - 4

            gsens_text.append(f"['{str(parameter_names[global_sens_sort_idx[i_p]])}']: " + " " * len_diff)
            for i_qoi in range(sobol_norm.shape[1]):
                if global_sens[global_sens_sort_idx[i_p]][0] < 0:
                    sep = " " * 3
                else:
                    sep = " " * 4
                gsens_text[-1] += sep + f"{global_sens[global_sens_sort_idx[i_p], i_qoi]:.2e}"
            gsens_text.append("\n")

        len_max_gsens = np.max([len(line) for line in gsens_text])
        gsens_text[1] = "=" * (len_max_gsens) + "\n"

        # write in file
        with open(fn_out, 'w') as f:
            for line in sobol_text:
                f.write(line)
            f.write("\n")
            for line in gsens_text:
                f.write(line)

    return sobol_dict, gsens_dict


def plot_sens_summary(sobol, gsens, session=None, coeffs=None, qois=None, mean=None, std=None, output_idx=None,
                      y_label="y", x_label="x", zlim=None, sobol_donut=True, plot_pdf_over_output_idx=False,
                      fn_plot=None):
    """
    Plot summary of Sobol indices and global derivative based sensitivity coefficients

    Parameters
    ----------
    session : GPC Session object instance
        GPC session object containing all information i.e., gPC, Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out] or list of ndarray of float [n_qoi][n_coeffs x n_out]
        GPC coefficients
    sobol : OrderedDict
        OrderedDict containing the normalized Sobol indices from get_sens_summary()
    gsens: OrderedDict
        OrderedDict containing the global derivative based sensitivity coefficients from get_sens_summary()
    sobol_donut : Boolean
        Option to plot the sobol indices as donut (pie) chart instead of bars, default is True
    multiple_qoi: Boolean
        Option to plot over a quantity of interest, needs an array of qoi values and results
    qois: numpy ndarray
        Axis of quantities of interest (x-axis, e.g. time)
    mean: numpy ndarray
        Mean from gpc session (determined with e.g.: pygpc.SGPC.get_mean(coeffs))
    std: numpy ndarray
        Std from gpc session (determined with e.g.: pygpc.SGPC.get_std(coeffs))
        (can be given and plotted when plot_pdf_over_output_idx=False)
    output_idx : int, str or None, optional, default=0
        Indices of output quantity to consider
    x_label : str
        Label of x-axis in case of multiple QOI plots
    y_label : str
        Label of y-axis in case of multiple QOI plots
    zlim : list of float, optional, default: None
        Limits of 3D plot (e.g. pdf) in z direction
    plot_pdf_over_output_idx : bool, optional, default: False
        Plots pdf as a surface plot over output index (e.g. a time axis)
    fn_plot : str, optional, default: None
        Filename of the plot to save (.png or .pdf)
    """
    import matplotlib.pyplot as plt

    if type(output_idx) is int:
        output_idx = [output_idx]

    if output_idx is None or output_idx == "all":
        if coeffs is not None:
            output_idx = np.arange(coeffs.shape[1])
        else:
            output_idx = np.array([0])

    glob_sens = np.array([gsens[key] for key in gsens.keys()])[:, output_idx].flatten()
    gsens_keys = list(gsens.keys())

    sobols = np.array([sobol[key] for key in sobol.keys()])[:, output_idx].flatten()
    sobol_keys = list(sobol.keys())

    # ignore very low Sobol indices
    mask = sobols >= 0.001

    # format keys for plot ticks
    sobol_labels = [(x[1:-1].replace("'", " ")).replace(" ,", ",") for x in sobol_keys]

    sobols = sobols[mask]
    sobol_labels = [s for i, s in enumerate(sobol_labels) if mask[i]]

    if len(output_idx) == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))
        if sobol_donut:
            wedgeprops = {"linewidth": 0.5, 'width': 0.5, "edgecolor": "k"}
            wedges, texts = ax1.pie(sobols, wedgeprops=wedgeprops, startangle=-40)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.72)
            kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

            last_label = False
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1

                if not last_label:
                    y = np.sin(np.deg2rad(ang))
                    x = np.cos(np.deg2rad(ang))
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                    kw["arrowprops"].update({"connectionstyle": connectionstyle})
                    ax1.annotate(sobol_labels[i] + f" ({sobols[i]*100:.1f}%)", xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                                horizontalalignment=horizontalalignment, **kw)
                    if ang > 310:
                        last_label = True

            # for i, p in enumerate(wedges):
            #     ang = (p.theta2 - p.theta1) / 2. + p.theta1
            #     y = np.sin(np.deg2rad(ang))
            #     x = np.cos(np.deg2rad(ang))
            #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
            #     ax1.annotate(sobol_labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
            #                 horizontalalignment=horizontalalignment, **kw)

            # ax1.legend(wedges, sobol_labels,
            #           title="Parameter",
            #           loc="center left",
            #           bbox_to_anchor=(1, 0, 0.5, 1))

            ax1.set_title("Normalized Sobol indices")
        else:
            ax1.bar(np.arange(len(sobol_keys)) + 1, sobols, width=0.8)
            ax1.set_xticklabels([" "] + sobol_labels)
            ax1.set_yscale('log')
            ax1.set_ylabel('Sobol indices', fontsize=14)
            ax1.set_xlabel('parameter', fontsize=14)
            ax1.set_xlim(0, len(sobol_keys) + 1)
            ax1.set_ylim(0., 1.0)
        ax2.bar(np.arange(len(gsens_keys)) + 1, glob_sens, color='orange')
        ax2.set_xticks(list(np.arange(len(gsens_keys)) + 1))
        ax2.set_xticklabels(list(gsens_keys))
        ax2.set_ylabel('global sensitivities', fontsize=14)
        ax2.set_xlabel('parameter', fontsize=14)
        ax2.axhline(y=0, color='k', alpha=0.5)
        plt.tight_layout()

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[8, 9])

        # Estimate output pdf
        if plot_pdf_over_output_idx:
            if session.qoi_specific:
                pdf_x = np.zeros((100, len(output_idx)))
                pdf_y = np.zeros((100, len(output_idx)))

                for i, o_idx in enumerate(output_idx):
                    pdf_x_tmp, pdf_y_tmp = session.gpc[o_idx].get_pdf(coeffs=coeffs[o_idx], n_samples=1e5, output_idx=0)
                    pdf_x[:, i] = pdf_x_tmp.flatten()
                    pdf_y[:, i] = pdf_y_tmp.flatten()
            else:
                pdf_x, pdf_y, _, y_gpc_samples = session.gpc[0].get_pdf(coeffs=coeffs,
                                                                        n_samples=1e5,
                                                                        output_idx=output_idx,
                                                                        return_samples=True)

            # interpolate pdf data on common grid
            x_interp = np.linspace(0, np.max(pdf_x), 1000)
            y_interp = np.zeros((len(x_interp), np.shape(pdf_x)[1]))

            for i in range(np.shape(pdf_x)[1]):
                y_interp[:, i] = np.interp(x_interp, pdf_x[:, i], pdf_y[:, i], left=0, right=0)

            if zlim is not None:
                vmin = zlim[0]
                vmax = zlim[1]
            else:
                vmin = np.min(y_interp)
                vmax = np.max(y_interp)

            if qois is not None:
                x_axis = qois
            else:
                x_axis = np.arange(0, len(output_idx))

            xx, yy = np.meshgrid(x_axis, x_interp)

            # plot pdf over output_idx
            ax1.pcolor(xx, yy, y_interp, cmap="bone_r", vmin=vmin, vmax=vmax)
            ax1.plot(x_axis, mean, "r", linewidth=1.5)
            legend_elements = [matplotlib.lines.Line2D([0], [0], color='r', lw=2, label='mean'),
                               matplotlib.patches.Patch(facecolor='grey', edgecolor='k', label='pdf')]
            ax1.legend(handles=legend_elements)

            ax1.grid()

            if x_label is not None:
                ax1.set_xlabel(x_label, fontsize=14)

            if y_label is not None:
                ax1.set_ylabel(y_label, fontsize=14)

        else:
            # mean and std
            ax1.plot(qois, mean)
            ax1.grid()
            ax1.set_ylabel(y_label, fontsize=14)
            ax1.set_xlabel(x_label, fontsize=14)
            ax1.legend(["mean of " + y_label], loc='upper left')
            ax1.set_xlim(qois[0], qois[-1] + (np.max(qois[-1]) * 1e-3))
            ax1.set_title("Mean and standard deviation of " + y_label, fontsize=14)
            # ax2.set_ylim(np.min(results) + np.max(std_results), np.max(results) + np.max(std_results))
            ax1.fill_between(qois, mean - std, mean + std, color="grey", alpha=0.5)

        # sobol
        for i in range(len(sobol.keys())):
            ax2.plot(qois, sobol[list(sobol.keys())[i]])
            ax2.set_title("Sobol indices of the parameters over the qois", fontsize=14)
            ax2.set_xlabel(x_label, fontsize=14)
            ax2.set_ylabel("Sobol index", fontsize=14)
            ax2.set_yscale('log')
        sobol_labels = [(x[1:-1].replace("'", " ")).replace(" ,", ",") for x in sobol_keys]
        ax2.legend(sobol_labels)
        # ax1.legend(sobol['sobol_norm (qoi 0)'].keys())
        ax2.set_xlim(qois[0], qois[-1] + (np.max(qois[-1]) * 1e-3))
        ax2.grid()

        # gsens
        for i in range(len(gsens.keys())):
            ax3.plot(qois, gsens[list(gsens.keys())[i]])
            ax3.set_title("Global derivatives of the parameters over the qois", fontsize=14)
            ax3.set_xlabel(x_label, fontsize=14)
            ax3.set_ylabel("Global sensitivity", fontsize=14)
            # ax3.set_yscale('log')
        gsens_labels = [x for x in gsens_keys]
        ax3.legend(gsens_labels)
        # ax1.legend(sobol['sobol_norm (qoi 0)'].keys())
        ax3.set_xlim(qois[0], qois[-1] + (np.max(qois[-1]) * 1e-3))
        ax3.grid()
        plt.tight_layout()

    if fn_plot is not None:
        plt.savefig(fn_plot, dpi=600)
        plt.close()


def plot_gpc(session, coeffs, random_vars=None, coords=None, results=None, n_grid=None, output_idx=0, fn_plot=None,
             camera_pos=None, zlim=None, plot_pdf_over_output_idx=False, qois=None, x_label=None, y_label=None):
    """
    Compares gPC approximation with original model function. Evaluates both at n_grid (x n_grid) sampling points and
    calculate the difference between two solutions at the output quantity with output_idx and saves the plot as
    *_QOI_idx_<output_idx>.png/pdf. Also generates one .hdf5 results file with the evaluation results.

    Parameters
    ----------
    session : GPC Session object instance
        GPC session object containing all information i.e., gPC, Problem, Model, Grid, Basis, RandomParameter instances
    coeffs : ndarray of float [n_coeffs x n_out] or list of ndarray of float [n_qoi][n_coeffs x n_out]
        GPC coefficients
    random_vars: str or list of str [2]
        Names of the random variables, the analysis is performed for one or max. two random variables
    n_grid : int or list of int [2], optional
        Number of samples in each dimension to compare the gPC approximation with the original model function.
        A cartesian grid is generated based on the limits of the specified random_vars
    coords : ndarray of float [n_coords x n_dim]
        Parameter combinations for the random_vars the comparison is conducted with
    output_idx : int, str or None, optional, default=0
        Indices of output quantity to consider
    results: ndarray of float [n_coords x n_out]
        If available, data of original model function at grid, containing all QOIs
    fn_plot : str, optional, default: None
        Filename of plot comparing original vs gPC model (*.png or *.pdf)
    camera_pos : list [2], optional, default: None
        Camera position of 3D surface plot (for 2 random variables only) [azimuth, elevation]
    zlim : list of float, optional, default: None
        Limits of 3D plot (e.g. pdf) in z direction
    plot_pdf_over_output_idx : bool, optional, default: False
        Plots pdf as a surface plot over output index (e.g. a time axis)
    qois: numpy ndarray
        Axis of quantities of interest (x-axis, e.g. time)
    x_label : str
        Label of x-axis in case of multiple QOI plots
    y_label : str
        Label of y-axis in case of multiple QOI plots

    Returns
    -------
    <file> : .hdf5 file
        Data file containing the grid points and the results of the original and the gpc approximation
    <file> : .png and .pdf file
        Plot comparing original vs gPC model
    """
    y_orig = None

    if type(output_idx) is int:
        output_idx = [output_idx]

    if output_idx is None or output_idx == "all":
        output_idx = np.arange(coeffs.shape[1])

    if random_vars is not None:
        if type(random_vars) is not list:
            random_vars = random_vars.tolist()
        assert len(random_vars) <= 2

    if n_grid is not None:
        if n_grid and type(n_grid) is not list:
            n_grid = n_grid.tolist()
    else:
        n_grid = [10, 10]

    if random_vars is not None:
        # Create grid such that it includes the mean values of other random variables
        grid = np.zeros((np.prod(n_grid), len(session.parameters_random)))

        idx = []
        idx_global = []

        # sort random_vars according to gpc.parameters
        for i_p, p in enumerate(session.parameters_random.keys()):
            if p not in random_vars:
                grid[:, i_p] = session.parameters_random[p].mean

            else:
                idx.append(random_vars.index(p))
                idx_global.append(i_p)

        random_vars = [random_vars[i] for i in idx]
        x = []

        for i_p, p in enumerate(random_vars):
            x.append(np.linspace(session.parameters_random[p].pdf_limits[0],
                                 session.parameters_random[p].pdf_limits[1],
                                 n_grid[i_p]))

        coords_gpc = get_cartesian_product(x)
        if len(random_vars) == 2:
            x1_2d, x2_2d = np.meshgrid(x[0], x[1])

        grid[:, idx_global] = coords_gpc

        # Normalize grid
        grid_norm = Grid(parameters_random=session.parameters_random).get_normalized_coordinates(grid)

    # Evaluate gPC expansion on grid and estimate output pdf
    if session.qoi_specific:
        y_gpc = np.zeros((grid_norm.shape[0], len(output_idx)))
        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))

        for i, o_idx in enumerate(output_idx):
            y_gpc[:, i] = session.gpc[o_idx].get_approximation(coeffs=coeffs[o_idx], x=grid_norm,
                                                               output_idx=0).flatten()

            pdf_x_tmp, pdf_y_tmp = session.gpc[o_idx].get_pdf(coeffs=coeffs[o_idx], n_samples=1e5, output_idx=0)
            pdf_x[:, i] = pdf_x_tmp.flatten()
            pdf_y[:, i] = pdf_y_tmp.flatten()
    else:
        if not plot_pdf_over_output_idx:
            y_gpc = session.gpc[0].get_approximation(coeffs=coeffs,
                                                     x=grid_norm,
                                                     output_idx=output_idx)

        pdf_x, pdf_y, _, y_gpc_samples = session.gpc[0].get_pdf(coeffs=coeffs,
                                                                n_samples=1e5,
                                                                output_idx=output_idx,
                                                                return_samples=True)

    if not plot_pdf_over_output_idx:
        if results is not None:
            y_orig = results[:, output_idx]

            if y_orig.ndim == 1:
                y_orig = y_orig[:, np.newaxis]

        # add axes if necessary
        if y_gpc.ndim == 1:
            y_gpc = y_gpc[:, np.newaxis]

    # Plot results
    matplotlib.rc('text', usetex=False)
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    fs = 14

    if plot_pdf_over_output_idx:
        # interpolate pdf data on common grid
        x_interp = np.linspace(0, np.max(pdf_x), 1000)
        y_interp = np.zeros((len(x_interp), np.shape(pdf_x)[1]))

        for i in range(np.shape(pdf_x)[1]):
            y_interp[:, i] = np.interp(x_interp, pdf_x[:, i], pdf_y[:, i], left=0, right=0)

        if zlim is not None:
            vmin = zlim[0]
            vmax = zlim[1]
        else:
            vmin = np.min(y_interp)
            vmax = np.max(y_interp)

        if qois is not None:
            x_axis = qois
        else:
            x_axis = np.arange(0, len(output_idx))

        xx, yy = np.meshgrid(x_axis, x_interp)

        # plot pdf over output_idx
        plt.figure(figsize=[10, 6])
        plt.pcolor(xx, yy, y_interp, cmap="bone_r", vmin=vmin, vmax=vmax)
        plt.plot(x_axis, np.mean(y_gpc_samples, axis=0), "r", linewidth=1.5)
        plt.grid()

        if x_label is not None:
            plt.xlabel(x_label, fontsize=14)

        if y_label is not None:
            plt.ylabel(y_label, fontsize=14)

        plt.tight_layout()

        if fn_plot is not None:
            plt.savefig(os.path.splitext(fn_plot)[0] + "_pdf_qoi.png", dpi=1200)
            plt.savefig(os.path.splitext(fn_plot)[0] + "_pdf_qoi.pdf")
            plt.close()

    else:
        for _i, i in enumerate(output_idx):
            fig = plt.figure(figsize=(12, 5))

            # One random variable
            if len(random_vars) == 1:
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.plot(coords_gpc, y_gpc[:, i])
                if y_orig is not None:
                    ax1.scatter(coords[:, idx_global[0]], y_orig[:, _i], s=7 * np.ones(len(y_orig[:, i])),
                                facecolor='w', edgecolors='k')
                    legend = [r"gPC", r"original"]
                else:
                    legend = [r"gPC"]
                ax1.legend(legend, fontsize=fs)
                ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
                ax1.set_ylabel(r"y(%s)" % random_vars[0], fontsize=fs)
                ax1.grid()

            # Two random variables
            elif len(random_vars) == 2:
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                im1 = ax1.plot_surface(x1_2d, x2_2d, np.reshape(y_gpc[:, _i], (x[1].size, x[0].size), order='f'),
                                       cmap="jet", alpha=0.75, linewidth=0, edgecolors=None)
                if y_orig is not None:
                    ax1.scatter(coords[:, idx_global[0]], coords[:, idx_global[1]], y_orig[:, _i],
                                'k', alpha=1, edgecolors='k', depthshade=False)
                ax1.set_title(r'gPC approximation', fontsize=fs)
                ax1.set_xlabel(r"%s" % random_vars[0], fontsize=fs)
                ax1.set_ylabel(r"%s" % random_vars[1], fontsize=fs)

                if camera_pos is not None:
                    ax1.view_init(elev=camera_pos[0], azim=camera_pos[1])

                fig.colorbar(im1, ax=ax1, orientation='vertical')

                if zlim is not None:
                    ax1.set_zlim(zlim)

            # plot histogram of output data and gPC estimated pdf
            ax2 = fig.add_subplot(1, 2, 2)
            if y_orig is not None:
                ax2.hist(y_orig[:, _i], density=True, bins=20, edgecolor='k')
            ax2.plot(pdf_x[:, _i], pdf_y[:, _i], 'r')
            ax2.grid()
            ax2.set_title("Probability density", fontsize=fs)
            ax2.set_xlabel(r'$y$', fontsize=16)
            ax2.set_ylabel(r'$p(y)$', fontsize=16)
            plt.tight_layout()

            if fn_plot is not None:
                plt.savefig(os.path.splitext(fn_plot)[0] + "_qoi_" + str(output_idx[i]) + '.png', dpi=1200)
                plt.savefig(os.path.splitext(fn_plot)[0] + "_qoi_" + str(output_idx[i]) + '.pdf')
                plt.close()
