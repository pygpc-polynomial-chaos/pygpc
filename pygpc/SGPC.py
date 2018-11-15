import time
import random
import numpy as np
import scipy.stats
from .GPC import *
from .io import iprint, wprint
from .misc import display_fancy_bar
from .misc import get_array_unique_rows
from .Basis import *


class SGPC(GPC):
    """
    Sub-class for standard gPC (SGPC)

    Attributes
    ----------
    order: list of int [dim]
        Maximum individual expansion order [order_1, order_2, ..., order_dim].
        Generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        Maximum global expansion order (sum of all exponents).
        The maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        Number of random variables, which can interact with each other.
        All polynomials are ignored, which have an interaction order greater than the specified
    """

    def __init__(self, problem, order, order_max, interaction_order):
        """
        Constructor; Initializes the SGPC class

        Parameters
        ----------
        problem: Problem class instance
            GPC Problem under investigation
        order: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum global expansion order (sum of all exponents).
            The maximum expansion order considers the sum of the orders of combined polynomials only
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        """
        super(SGPC, self).__init__(problem)

        self.order = order
        self.order_max = order_max
        self.interaction_order = interaction_order

        self.basis = Basis()
        self.basis.init_basis_sgpc(problem=problem,
                                   order=order,
                                   order_max=order_max,
                                   interaction_order=interaction_order)

    @staticmethod
    def get_mean(coeffs):
        """
        Calculate the expected mean value.

        mean = SGPC.get_mean(coeffs)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients

        Returns
        -------
        mean: [1 x N_out] np.ndarray
            Expected mean value
        """

        mean = coeffs[0]
        # TODO: check if 1-dimensional array should be (N,) or (N,1)
        # mean = mean[np.newaxis, :]
        return mean

    @staticmethod
    def get_standard_deviation(coeffs):
        """
        Calculate the standard deviation.

        std = SGPC.get_standard_deviation(coeffs)

        Parameters
        ----------
        coeffs: np.array of float [N_coeffs x N_out]
            Gpc coefficients

        Returns
        -------
        std: [1 x N_out] np.ndarray
            Standard deviation
        """

        std = np.sqrt(np.sum(np.square(coeffs[1:]), axis=0))
        # TODO: check if 1-dimensional array should be (N,) or (N,1)
        # std = std[np.newaxis, :]
        return std

    def get_samples(self, n_samples, coeffs, output_idx=None):
        """
        Randomly sample gPC expansion.

        x, pce = SGPC.get_pdf_mc(N_samples, coeffs=None, output_idx=None)

        Parameters
        ----------
        n_samples: int
            Number of random samples drawn from the respective input pdfs.
        coeffs: [N_coeffs x N_out] np.ndarray, optional, default=None
            GPC coefficients
        output_idx: [1 x N_out] np.ndarray, optional, default=None
            Index of output quantities to consider.

        Returns
        -------
        x: [n_samples x dim] np.ndarray
            Generated samples in normalized coordinates.
        pce: [n_samples x n_out] np.ndarray
            GPC approximation at points x.
        """

        # seed the random numbers generator
        np.random.seed()

        # generate temporary grid with random samples for each random input variable [n_samples x dim]
        grid = RandomGrid(problem=self.problem, parameters={"n_grid": n_samples, "seed": None})

        # if output index list is not provided, sample all gpc outputs
        if output_idx is None:
            output_idx = np.arange(coeffs.shape[1])
            # output_idx = output_idx[np.newaxis, :]
        pce = self.get_approximation(coeffs=coeffs, x=grid.coords_norm, output_idx=output_idx)

        return grid.coords_norm, pce

    # noinspection PyTypeChecker
    def get_sobol_indices(self, coeffs):
        """
        Calculate the available sobol indices.

        sobol, sobol_idx = SGPC.get_sobol_indices(coeffs=None)

        Parameters
        ----------
        coeffs: [n_coeffs x n_out] np.ndarray
            GPC coefficients

        Returns
        -------
        sobol: ndarray of float [n_sobol x n_out]
            Unnormalized Sobol indices
        sobol_idx: list of ndarray of int [n_sobol x n_sobol_included]
            Parameter combinations in rows of sobol.
        sobol_idx_bool: list of ndarray of bool [n_sobol x dim]
            Boolean mask that determines which multi indices are unique.
        """

        iprint("Determining Sobol indices...")

        # handle (N,) arrays
        if len(coeffs.shape) == 1:
            n_out = 1
        else:
            n_out = coeffs.shape[1]

        n_coeffs = coeffs.shape[0]

        if n_coeffs == 1:
            raise Exception('Number of coefficients is 1 ... no sobol indices to calculate ...')

        # Generate boolean matrix of all basis functions where order > 0 = True
        # size: [n_basis x dim]
        multi_indices = np.array([map(lambda _b:_b.p["i"], b_row) for b_row in self.basis.b])
        sobol_mask = multi_indices != 0

        # look for unique combinations (i.e. available sobol combinations)
        # size: [N_sobol x dim]
        sobol_idx_bool = get_array_unique_rows(sobol_mask)

        # delete the first row where all polynomials are order 0 (no sensitivity)
        sobol_idx_bool = np.delete(sobol_idx_bool, [0], axis=0)
        n_sobol_available = sobol_idx_bool.shape[0]

        # check which basis functions contribute to which sobol coefficient set
        # True for specific coeffs if it contributes to sobol coefficient
        # size: [N_coeffs x N_sobol]
        sobol_poly_idx = np.zeros([n_coeffs, n_sobol_available])

        for i_sobol in range(n_sobol_available):
            sobol_poly_idx[:, i_sobol] = np.all(sobol_mask == sobol_idx_bool[i_sobol], axis=1)

        # calculate sobol coefficients matrix by summing up the individual
        # contributions to the respective sobol coefficients
        # size [N_sobol x N_points]
        sobol = np.zeros([n_sobol_available, n_out])

        for i_sobol in range(n_sobol_available):
            sobol[i_sobol] = np.sum(np.square(coeffs[sobol_poly_idx[:, i_sobol] == 1]), axis=0)

        # sort sobol coefficients in descending order (w.r.t. first output only ...)
        idx_sort_descend_1st = np.argsort(sobol[:, 0], axis=0)[::-1]
        sobol = sobol[idx_sort_descend_1st, :]
        sobol_idx_bool = sobol_idx_bool[idx_sort_descend_1st]

        # get list of sobol indices
        sobol_idx = [0 for _ in range(sobol_idx_bool.shape[0])]

        for i_sobol in range(sobol_idx_bool.shape[0]):
            sobol_idx[i_sobol] = np.array([i for i, x in enumerate(sobol_idx_bool[i_sobol, :]) if x])

        return sobol, sobol_idx, sobol_idx_bool

    def get_sobol_composition(self, sobol, sobol_idx, sobol_idx_bool):
        """
        Determine average ratios of Sobol indices over all output quantities:
        (i) over all orders and (e.g. 1st: 90%, 2nd: 8%, 3rd: 2%)
        (ii) for the 1st order indices w.r.t. each random variable. (1st: x1: 50%, x2: 40%)

        sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std, sobol_rel_1st_order_mean, sobol_rel_1st_order_std
        = SGPC.get_sobol_composition(coeffs=None, sobol=None, sobol_idx=None, sobol_idx_bool=None)

        Parameters
        ----------
        sobol: [N_sobol x N_out] np.ndarray
            Unnormalized sobol_indices
        sobol_idx: list of [N_sobol x dim] np.ndarray
            Parameter combinations in rows of sobol.
        sobol_idx_bool: list of np.ndarray of bool
            Boolean mask that determines which multi indices are unique.

        Returns
        -------
        sobol_rel_order_mean: np.ndarray
            Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,),
            (over all output quantities)
        sobol_rel_order_std: np.ndarray
            Standard deviation of the proportion of the Sobol indices of the different order to the total variance
            (1st, 2nd, etc..,), (over all output quantities)
        sobol_rel_1st_order_mean: np.ndarray
            Average proportion of the random variables of the 1st order Sobol indices to the total variance,
            (over all output quantities)
        sobol_rel_1st_order_std: np.ndarray
            Standard deviation of the proportion of the random variables of the 1st order Sobol indices to the total
            variance
            (over all output quantities)
        """

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
        str_out = []

        # get maximum length of random_vars label
        max_len = max([len(self.problem.random_vars[i]) for i in range(len(self.problem.random_vars))])

        for i in range(order_max):
            # extract sobol coefficients of order i
            sobol_extracted, sobol_extracted_idx = self.get_extracted_sobol_order(sobol, sobol_idx, i + 1)

            # determine average sobol index over all elements
            sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten()) /
                                        np.sum(var[not_nan_mask]))

            sobol_rel_order_std.append(np.std(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten() /
                                              var[not_nan_mask]))

            iprint("Ratio: Sobol indices order {} / total variance: {:.4f} +- {:.4f}"
                   .format(i+1, sobol_rel_order_mean[i], sobol_rel_order_std[i]), tab=1, verbose=self.verbose)

            # for first order indices, determine ratios of all random variables
            if i == 0:
                # deep copy
                sobol_extracted_idx_1st = sobol_extracted_idx[:]
                for j in range(sobol_extracted.shape[0]):
                    sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                    / np.sum(var[not_nan_mask]))
                    sobol_rel_1st_order_std.append(0)

                    str_out.append("\t{}{}: {:.4f}"
                                   .format((max_len - len(self.problem.random_vars[sobol_extracted_idx_1st[j]])) * ' ',
                                           self.problem.random_vars[sobol_extracted_idx_1st[j]],
                                           sobol_rel_1st_order_mean[j]))

        sobol_rel_order_mean = np.array(sobol_rel_order_mean)
        sobol_rel_1st_order_mean = np.array(sobol_rel_1st_order_mean)

        # print output of 1st order Sobol indices ratios of parameters
        if self.verbose:
            for j in range(len(str_out)):
                print(str_out[j])

        return sobol, sobol_idx, \
               sobol_rel_order_mean, sobol_rel_order_std, \
               sobol_rel_1st_order_mean, sobol_rel_1st_order_std

    @staticmethod
    def get_extracted_sobol_order(sobol, sobol_idx, order=1):
        """
        Extract Sobol indices with specified order from Sobol data.

        sobol_1st, sobol_idx_1st = SGPC.get_extracted_sobol_order(sobol, sobol_idx, order=1)

        Parameters
        ----------
        sobol: ndarray [N_sobol x N_out]
            Sobol indices of N_out output quantities
        sobol_idx: [N_sobol] list or ndarray of int
            Parameter label indices belonging to Sobol indices
        order: int, optional, default=1
            Sobol index order to extract

        Returns
        -------
        sobol_n_order: ndarray of float
            n-th order Sobol indices of N_out output quantities
        sobol_idx_n_order: ndarray of int
            Parameter label indices belonging to n-th order Sobol indices
        """

        # make mask of nth order sobol indices
        mask = [index for index, sobol_element in enumerate(sobol_idx) if sobol_element.shape[0] == order]

        # extract from dataset
        sobol_n_order = sobol[mask, :]
        sobol_idx_n_order = np.vstack(sobol_idx[mask])

        # sort sobol indices according to parameter indices in ascending order
        sort_idx = np.argsort(sobol_idx_n_order, axis=0)[:, 0]
        sobol_n_order = sobol_n_order[sort_idx, :]
        sobol_idx_n_order = sobol_idx_n_order[sort_idx, :]

        return sobol_n_order, sobol_idx_n_order

    # noinspection PyTypeChecker
    def get_global_sens(self, coeffs):
        """
        Determine the global derivative based sensitivity coefficients after Xiu (2009) [1].

        global_sens = SGPC.get_global_sens(coeffs)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients

        Returns
        -------
        global_sens: ndarray [dim x n_out]
            Global derivative based sensitivity coefficients

        Notes
        -----
        .. [1] D. Xiu, Fast Numerical Methods for Stochastic Computations: A Review,
           Commun. Comput. Phys., 5 (2009), pp. 242-272 eq. (3.14) page 255
        """

        b_int_global = np.zeros([self.problem.dim, self.basis.n_basis])

        # construct matrix with integral expressions [n_basis x dim]
        b_int = np.array([map(lambda _b: _b.fun_int, b_row) for b_row in self.basis.b])
        b_int_der = np.array([map(lambda _b: _b.fun_der_int, b_row) for b_row in self.basis.b])

        for i_sens in range(self.problem.dim):
            # replace column with integral expressions from derivative of parameter[i_dim]
            tmp = copy.deepcopy(b_int)
            tmp[:, i_sens] = b_int_der[:, i_sens]

            # determine global integral expression
            b_int_global[i_sens, :] = np.prod(tmp, axis=1)

        global_sens = np.dot(b_int_global, coeffs)
        # global_sens = np.dot(b_int_global, coeffs) / (2 ** self.problem.dim)

        return global_sens

    # noinspection PyTypeChecker
    def get_local_sens(self, coeffs, x):
        """
        Determine the local derivative based sensitivity coefficients in the point of interest x
        (normalized coordinates [-1, 1]).

        local_sens = SGPC.calc_localsens(coeffs, x)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        x: ndarray of float [n_basis x n_out]
            Point in variable space to evaluate local sensitivity in (normalized coordinates [-1, 1])

        Returns
        -------
        local_sens: ndarray [dim x n_out]
            Local sensitivity of output quantities in point x
        """

        b_x_global = np.zeros([self.problem.dim, self.basis.n_basis])

        # evaluate fun(x) and fun_der(x) at point of operation x [n_basis x dim]
        b_x = np.array([map(lambda _b: _b.fun(x), b_row) for b_row in self.basis.b])
        b_der_x = np.array([map(lambda _b: _b.fun_der(x), b_row) for b_row in self.basis.b])

        for i_sens in range(self.problem.dim):
            # replace column with integral expressions from derivative of parameter[i_dim]
            tmp = copy.deepcopy(b_x)
            tmp[:, i_sens] = b_der_x[:, i_sens]

            # determine global integral expression
            b_x_global[i_sens, :] = np.prod(tmp, axis=1)

        local_sens = np.dot(b_x_global, coeffs)

        return local_sens

    def get_pdf(self, coeffs, n_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = SGPC.get_pdf(coeffs, n_samples, output_idx=None)

        Parameters
        ----------
        coeffs: [n_coeffs x n_out] np.ndarray
            GPC coefficients
        n_samples: int
            Number of samples used to estimate output pdfs
        output_idx: ndarray, optional, default=None [1 x n_out]
            Index of output quantities to consider (if output_idx=None, all output quantities are considered)

        Returns
        -------
        pdf_x: [100 x n_out] np.ndarray
            x-coordinates of output pdfs of output quantities
        pdf_y: [100 x n_out] np.ndarray
            y-coordinates of output pdfs (probability density of output quantity)
        """

        # handle (N,) arrays
        if len(coeffs.shape) == 1:
            n_out = 1
        else:
            n_out = coeffs.shape[1]

        # if output index array is not provided, determine pdfs of all outputs
        if not output_idx:
            output_idx = np.linspace(0, n_out - 1, n_out)
            output_idx = output_idx[np.newaxis, :]

        # sample gPC expansion
        samples_in, samples_out = self.get_samples(n_samples=n_samples, coeffs=coeffs, output_idx=output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, n_out])
        pdf_y = np.zeros([100, n_out])

        for i_out in range(n_out):
            kde = scipy.stats.gaussian_kde(samples_out.transpose(), bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)
            pdf_y[:, i_out] = kde(pdf_x[:, i_out])

        return pdf_x, pdf_y


class Reg(SGPC):
    """
    Regression gPC subclass

    Reg(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None)

    Attributes
    ----------
    relative_error_loocv: list of float
        relative error of the leave-one-out-cross-validation
    """

    def __init__(self, problem, order, order_max, interaction_order):
        """
        Constructor; Initializes Regression SGPC class

        Parameters
        ----------
        problem: Problem class instance
            GPC Problem under investigation
        order: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum global expansion order (sum of all exponents).
            The maximum expansion order considers the sum of the orders of combined polynomials only
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        """
        super(Reg, self).__init__(problem, order, order_max, interaction_order)
        self.solver = 'Moore-Penrose'   # Default solver
        self.relative_error_loocv = []

    def loocv(self, sim_results):
        """
        Perform leave one out cross validation of gPC with maximal 100 points
        and add result to self.relative_error_loocv.

        relative_error_loocv = SGPC.loocv(sim_results)

        Parameters
        ----------
        sim_results: [n_grid x n_out] np.ndarray
            Results from n_grid simulations with n_out output quantities

        Returns
        -------
        relative_error_loocv: float
            Relative mean error of leave one out cross validation
        """

        # define number of performed cross validations (max 100)
        n_loocv_points = np.min((sim_results.shape[0], 100))

        # make list of indices, which are randomly sampled
        loocv_point_idx = random.sample(list(range(sim_results.shape[0])), n_loocv_points)

        start = time.time()
        relative_error = np.zeros(n_loocv_points)
        for i in range(n_loocv_points):
            # get mask of eliminated row
            mask = np.arange(sim_results.shape[0]) != loocv_point_idx[i]

            # invert reduced gpc matrix
            gpc_matrix_inv_loo = np.linalg.pinv(self.gpc_matrix[mask, :])

            # determine gpc coefficients (this takes a lot of time for large problems)
            coeffs_loo = np.dot(gpc_matrix_inv_loo, sim_results[mask, :])
            sim_results_temp = sim_results[loocv_point_idx[i], :]
            relative_error[i] = scipy.linalg.norm(sim_results_temp - np.dot(self.gpc_matrix[loocv_point_idx[i], :],
                                                                            coeffs_loo))\
                                / scipy.linalg.norm(sim_results_temp)
            display_fancy_bar("LOOCV", int(i + 1), int(n_loocv_points))

        # store result in relative_error_loocv
        self.relative_error_loocv.append(np.mean(relative_error))
        iprint(" (" + str(time.time() - start) + ")")

        return self.relative_error_loocv[-1]


class Quad(SGPC):
    """
    Quadrature SGPC sub-class

    Quad(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None)

    Attributes
    ----------

    """

    def __init__(self, problem, order, order_max, interaction_order):
        """
        Constructor; Initializes Quadrature SGPC sub-class

        Parameters
        ----------
        problem: Problem class instance
            GPC Problem under investigation
        order: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum global expansion order (sum of all exponents).
            The maximum expansion order considers the sum of the orders of combined polynomials only
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        """
        super(Quad, self).__init__(problem, order, order_max, interaction_order)
        self.solver = 'NumInt'  # Default solver



        # # handle (N,) arrays
        # if len(sim_results.shape) == 1:
        #     self.N_out = 1
        # else:
        #     self.N_out = sim_results.shape[1]




    # N_grid: int
    #     number of grid points
    # N_poly: int
    #     number of polynomials psi
    # N_samples: int
    #     number of samples xi
    #
    # order: [dim] list of int
    #     maximum individual expansion order
    #     generates individual polynomials also if maximum expansion order in order_max is exceeded
    # order_max: int
    #     maximum expansion order (sum of all exponents)
    #     the maximum expansion order considers the sum of the orders of combined polynomials only
    # interaction_order: int
    #     number of random variables, which can interact with each other
    #     all polynomials are ignored, which have an interaction order greater than the specified
    # grid: grid object
    #     grid object generated in grid.py including grid.coords and grid.coords_norm
    #
    # sobol: [N_sobol x N_out] np.ndarray
    #     Sobol indices of N_out output quantities
    # sobol_idx: [N_sobol] list of np.ndarray
    #     List of parameter label indices belonging to Sobol indices
    #
    # gpc_coeffs: [N_poly x N_out] np.ndarray
    #     coefficient matrix of independent regions of interest for every coefficient
    # poly: [dim x order_span] list of list of np.poly1d:
    #     polynomial objects containing the coefficients that are used to build the gpc matrix
    # poly_gpu: np.ndarray
    #     polynomial coefficients stored in a np.ndarray that can be processed on a graphic card
    # poly_idx: [N_poly x dim] np.ndarray
    #     multi indices to determine the degree of the used sub-polynomials
    # poly_idx_gpu [N_poly x dim] np.ndarray
    #     multi indices to determine the degree of the used sub-polynomials stored in a np.ndarray that can be processed
    #     on a graphic card
    # poly_der: [dim x order_span] list of list of np.poly1d:
    #     derivative of the polynomial objects containing the coefficients that are used to build the gpc matrix
    # poly_norm: [order_span x dim] np.ndarray
    #     normalizing scaling factors of the used sub-polynomials
    # poly_norm_basis: [N_poly] np.ndarray
    #     normalizing scaling factors of the polynomial basis functions
    # sobol_idx_bool: list of np.ndarray of bool
    #     boolean mask that determines which multi indices are unique


    # N_grid: int
    #     number of grid points
    # dim: int
    #     number of uncertain parameters to process
    # pdf_type: [dim] list of str
    #     type of pdf 'beta' or 'norm'
    # pdf_shape: list of list of float
    #     shape parameters of pdfs
    #     beta-dist:   [[alpha], [beta]    ]
    #     normal-dist: [[mean],  [variance]]
    # limits: list of list of float
    #     upper and lower bounds of random variables
    #     beta-dist:   [[a1 ...], [b1 ...]]
    #     normal-dist: [[0 ... ], [0 ... ]] (not used)
    # order: [dim] list of int
    #     maximum individual expansion order
    #     generates individual polynomials also if maximum expansion order in order_max is exceeded
    # order_max: int
    #     maximum expansion order (sum of all exponents)
    #     the maximum expansion order considers the sum of the orders of combined polynomials only
    # interaction_order: int
    #     number of random variables, which can interact with each other
    #     all polynomials are ignored, which have an interaction order greater than the specified
    # grid: grid object
    #     grid object generated in grid.py including grid.coords and grid.coords_norm
    # random_vars: [dim] list of str
    #     string labels of the random variables