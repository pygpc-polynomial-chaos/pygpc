import h5py
import numpy as np
import pandas as pd

from .Grid import *
from .MEGPC import *
from .io import read_session


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
    sobol : pandas DataFrame
        Pandas DataFrame containing the normalized Sobol indices by significance
    gsens : pandas DataFrame
        Pandas DataFrame containing the global derivative based sensitivity coefficients
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

    sobol = pd.DataFrame.from_dict(sobol_dict, orient="index", columns=[f"sobol_norm (qoi {i})"
                                                                        for i in range(sobol_norm.shape[1])])
    len_max = np.max(p_length)

    # Extract global derivative sensitivity coefficients
    gsens_dict = OrderedDict()
    for i_p, p in enumerate(parameter_names):
        gsens_dict[p] = global_sens[i_p, :]

    gsens = pd.DataFrame.from_dict(gsens_dict, orient="index", columns=[f"global_sens (qoi {i})"
                                                                        for i in range(sobol_norm.shape[1])])

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

    return sobol, gsens


def plot_sens_summary(sobol, gsens, multiple_qoi=False, qois=None, results=None,
                      y_label="y", x_label="x", sobol_donut=True):
    """
    Plot summary of Sobol indices and global derivative based sensitivity coefficients

    Parameters
    ----------
    sobol : pandas DataFrame
        Pandas DataFrame containing the normalized Sobol indices from get_sens_summary()
    gsens: pandas DataFrame
        Pandas DataFrame containing the global derivative based sensitivity coefficients from get_sens_summary()
    sobol_donut : Boolean
        Option to plot the sobol indices as donut (pie) chart instead of bars, default is True
    multiple_qoi: Boolean
        Option to plot over a quantity of interest, needs an array of qoi values and results
    qois: numpy ndarray
        Quantities of interest
    results: numpy ndarray
        Results from gpc session
    """
    import matplotlib.pyplot as plt

    glob_sens = gsens.values.flatten()
    gsens_keys = gsens["global_sens (qoi 0)"].keys()
    sobols = sobol.values.flatten()
    sobol_keys = sobol["sobol_norm (qoi 0)"].keys()

    # ignore very low Sobol indices
    mask = sobols >= 0.001

    # format keys for plot ticks
    sobol_labels = [(x[1:-1].replace("'", " ")).replace(" ,", ",") for x in sobol_keys]

    sobols = sobols[mask]
    sobol_labels = [s for i, s in enumerate(sobol_labels) if mask[i]]

    if multiple_qoi == False:
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
        plt.show()
    else:
        if not (type(qois) == np.ndarray and type(results) == np.ndarray):
            raise ValueError("Please specifiy qois and results as a numpy array of values!")
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[8, 9])

            # mean and std
            mean_results = np.mean(results, axis=0)
            std_results = np.std(results, axis=0)
            ax1.plot(qois, mean_results)
            ax1.grid()
            ax1.set_ylabel(y_label, fontsize=14)
            ax1.set_xlabel(x_label, fontsize=14)
            ax1.legend(["mean of " + y_label], loc='upper left')
            ax1.set_xlim(qois[0], qois[-1] + (np.max(qois[-1]) * 1e-3))
            ax1.set_title("Mean and standard deviation of " + y_label, fontsize=14)
            # ax2.set_ylim(np.min(results) + np.max(std_results), np.max(results) + np.max(std_results))
            ax1.fill_between(qois, mean_results - std_results, mean_results + std_results, color="grey", alpha=0.5)

            # sobol
            for i in range(sobol.values.shape[0]):
                ax2.plot(qois, sobol.values[i])
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
            for i in range(gsens.values.shape[0]):
                ax3.plot(qois, gsens.values[i])
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
            plt.show()
