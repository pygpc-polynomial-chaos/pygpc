import h5py
import numpy as np
from .io import read_session
from .Grid import *
from .MEGPC import *


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
