# -*- coding: utf-8 -*-
import h5py
import numpy as np
from .io import read_gpc_pkl
from .Grid import RandomGrid
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
    n_samples : int, optional, default: 1e4
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
    p_matrix = None
    p_matrix_norm = None
    sobol = None
    sobol_idx_bool = None
    global_sens = None
    pdf_x = None
    pdf_y = None
    grid = None
    res = None
    projection = False

    with h5py.File(fn_gpc + ".hdf5", 'r') as f:

        # list of filenames of gPC .pkl files
        fn_gpc_pkl = [fn.astype(str) for fn in list(f["misc/fn_gpc_pkl"][:].flatten())]

        print("> Loading gpc object: {}".format(fn_gpc_pkl[0]))

        gpc = read_gpc_pkl(os.path.join(os.path.split(fn_gpc)[0], fn_gpc_pkl[0]))

        # check if we have qoi specific gPCs here
        try:
            if "qoi" in list(f["coeffs"].keys())[0]:
                qoi_keys = list(f["coeffs"].keys())
                qoi_idx = [int(key.split("qoi_")[1]) for key in qoi_keys]
                qoi_specific = True
            else:
                qoi_specific = False

        except AttributeError:
            qoi_specific = False

        if isinstance(gpc, MEGPC):
            multi_element_gpc = True
        else:
            multi_element_gpc = False

        # load coeffs depending on gpc type
        print("> Loading gpc coeffs: {}".format(fn_gpc + ".hdf5"))

        if not qoi_specific and not multi_element_gpc:
            coeffs = np.array(f['coeffs'][:])

            if not output_idx:
                output_idx = np.arange(coeffs.shape[1])

        elif not qoi_specific and multi_element_gpc:

            coeffs = [None for _ in range(len(list(f['coeffs'].keys())))]

            for i, d in enumerate(list(f['coeffs'].keys())):
                coeffs[i] = np.array(f['coeffs/' + str(d)])[:]

            if not output_idx:
                output_idx = np.arange(coeffs[0].shape[1])

        elif qoi_specific and not multi_element_gpc:

            algorithm = "sampling"
            coeffs = dict()

            for key in qoi_keys:
                coeffs[key] = np.array(f['coeffs/' + key])[:]

            if not output_idx:
                output_idx = np.arange(len(list(coeffs.keys())))

        elif qoi_specific and multi_element_gpc:

            algorithm = "sampling"
            coeffs = dict()

            for key in qoi_keys:
                coeffs[key] = [None for _ in range(len(list(f['coeffs/' + key].keys())))]

                for i, d in enumerate(list(f['coeffs/' + key].keys())):
                    coeffs[key][i] = np.array(f['coeffs/' + key + "/" + str(d)])[:]

            if not output_idx:
                output_idx = np.arange(len(list(coeffs.keys())))

        if "p_matrix" in f.keys():
            projection = True

        dim = f["grid/coords"][:].shape[1]

    # generate samples in case of sampling approach
    if algorithm == "sampling":

        if projection and not multi_element_gpc:
            grid = RandomGrid(parameters_random=gpc.problem_original.parameters_random,
                              options={"n_grid": n_samples, "seed": None})

        else:
            grid = RandomGrid(parameters_random=gpc.problem.parameters_random,
                              options={"n_grid": n_samples, "seed": None})

    # start prostprocessing depending on gPC type
    if not qoi_specific and not multi_element_gpc:

        coeffs = coeffs[:, output_idx]

        if algorithm == "standard":
            # determine mean
            mean = gpc.get_mean(coeffs=coeffs)

            # determine standard deviation
            std = gpc.get_std(coeffs=coeffs)

        elif algorithm == "sampling":
            # run model evaluations
            res = gpc.get_approximation(coeffs=coeffs, x=grid.coords_norm)

            # determine mean
            mean = gpc.get_mean(samples=res)

            # determine standard deviation
            std = gpc.get_std(samples=res)

        # determine Sobol indices
        if calc_sobol:
            sobol, sobol_idx, sobol_idx_bool = gpc.get_sobol_indices(coeffs=coeffs,
                                                                     algorithm=algorithm,
                                                                     n_samples=n_samples)

        # determine global derivative based sensitivity coefficients
        if calc_global_sens:
            global_sens = gpc.get_global_sens(coeffs=coeffs,
                                              algorithm=algorithm,
                                              n_samples=n_samples)

        # determine pdfs
        if calc_pdf:
            pdf_x, pdf_y = gpc.get_pdf(coeffs, n_samples=n_samples, output_idx=output_idx)

        else:
            raise AssertionError("Please provide valid algorithm argument (""standard"" or ""sampling"")")

    elif qoi_specific and not multi_element_gpc:

        gpc = []
        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        res = np.zeros((grid.coords_norm.shape[0], len(output_idx)))

        # loop over qoi (there may be different approximations and projections)
        for i, idx in enumerate(output_idx):
            # load gpc object
            gpc = read_gpc_pkl(fn_gpc + "_qoi_" + str(idx) + ".pkl")

            # run model evaluations
            res[:, i] = gpc.get_approximation(coeffs=coeffs["qoi_" + str(idx)], x=grid.coords_norm).flatten()

            # determine Sobol indices
            if calc_sobol:
                sobol_qoi, sobol_idx_qoi, sobol_idx_bool_qoi = gpc.get_sobol_indices(coeffs=coeffs["qoi_" + str(idx)],
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
                global_sens[:, ] = gpc.get_global_sens(coeffs=coeffs["qoi_" + str(idx)],
                                                       algorithm=algorithm,
                                                       n_samples=n_samples)

            # determine pdfs
            if calc_pdf:
                pdf_x[:, ], pdf_y[:, ] = gpc.get_pdf(coeffs=coeffs["qoi_" + str(idx)],
                                                     n_samples=n_samples,
                                                     output_idx=0)

        # determine mean
        mean = gpc.get_mean(samples=res)

        # determine standard deviation
        std = gpc.get_std(samples=res)

    elif not qoi_specific and multi_element_gpc:

        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        if algorithm == "sampling":

            res = np.zeros((grid.coords_norm.shape[0], len(output_idx)))

            # load gpc object
            gpc = read_gpc_pkl(os.path.join(os.path.split(fn_gpc)[0], fn_gpc_pkl[0]))

            # run model evaluations
            res = gpc.get_approximation(coeffs=coeffs, x=grid.coords_norm)

            # determine Sobol indices
            if calc_sobol:
                sobol, sobol_idx, sobol_idx_bool = gpc.get_sobol_indices(coeffs=coeffs,
                                                                         algorithm=algorithm,
                                                                         n_samples=n_samples)

            # determine global derivative based sensitivity coefficients
            if calc_global_sens:
                global_sens[:, ] = gpc.get_global_sens(coeffs=coeffs,
                                                       algorithm=algorithm,
                                                       n_samples=n_samples)

            # determine pdfs
            if calc_pdf:
                pdf_x[:, ], pdf_y[:, ] = gpc.get_pdf(coeffs=coeffs,
                                                     n_samples=n_samples,
                                                     output_idx=output_idx)

            # determine mean
            mean = gpc.get_mean(samples=res)

            # determine standard deviation
            std = gpc.get_std(samples=res)

        else:
            raise AssertionError("Please use ""sampling"" algorithm in case of multi-element gPC!")

    elif qoi_specific and multi_element_gpc:

        pdf_x = np.zeros((100, len(output_idx)))
        pdf_y = np.zeros((100, len(output_idx)))
        global_sens = np.zeros((dim, len(output_idx)))

        if algorithm == "sampling":

            res = np.zeros((grid.coords_norm.shape[0], len(output_idx)))

            # loop over qoi (there may be different approximations and projections)
            for i, idx in enumerate(output_idx):

                # load gpc object
                gpc = read_gpc_pkl(os.path.join(os.path.split(fn_gpc)[0], fn_gpc_pkl[0]))

                # run model evaluations
                res[:, i] = gpc.get_approximation(coeffs=coeffs["qoi_" + str(idx)], x=grid.coords_norm).flatten()

                # determine Sobol indices
                if calc_sobol:
                    sobol_qoi, sobol_idx_qoi, sobol_idx_bool_qoi = gpc.get_sobol_indices(coeffs=coeffs["qoi_" + str(idx)],
                                                                                         algorithm=algorithm,
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
                    global_sens[:, ] = gpc.get_global_sens(coeffs=coeffs["qoi_" + str(idx)],
                                                           algorithm=algorithm,
                                                           n_samples=n_samples)

                # determine pdfs
                if calc_pdf:
                    pdf_x[:, ], pdf_y[:, ] = gpc.get_pdf(coeffs=coeffs["qoi_" + str(idx)],
                                                         n_samples=n_samples,
                                                         output_idx=0)

            # determine mean
            mean = gpc.get_mean(samples=res)

            # determine standard deviation
            std = gpc.get_std(samples=res)

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
            f.create_dataset(data=sobol, name="sens/sobol")
            f.create_dataset(data=sobol_idx_bool, name="sens/sobol_idx_bool")

        if calc_global_sens:
            f.create_dataset(data=global_sens, name="sens/global_sens")

        if calc_pdf:
            f.create_dataset(data=pdf_x, name="sens/pdf_x")
            f.create_dataset(data=pdf_y, name="sens/pdf_y")
