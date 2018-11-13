import time
import random
import numpy as np
from builtins import range, int
from .GPC import *
from .io import iprint, wprint
from .misc import display_fancy_bar
from .Basis import *


class SGPC(GPC):
    """
    Class for standard gPC

    Attributes
    ----------
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    """

    def __init__(self, problem, order, order_max, interaction_order):
        """
        Constructor; Initializes the SGPC class


        Parameters
        ----------
        order: [dim] list of int
            Maximum individual expansion order
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum expansion order (sum of all exponents)
            The maximum expansion order considers the sum of the orders of combined polynomials only
        interaction_order: int
            Number of random variables, which can interact with each other
            All polynomials are ignored, which have an interaction order greater than specified
        """
        super(SGPC, self).__init__(problem)

        self.order = order
        self.order_max = order_max
        self.interaction_order = interaction_order

        self.basis = Basis()
        self.basis.init_basis_sgpc(order, order_max, interaction_order)

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
        Randomly sample the gPC expansion to determine output pdfs in specific points.

        xi = SGPC.get_pdf_mc(N_samples, coeffs=None, output_idx=None)

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
        xi: [N_samples x dim] np.ndarray
            Generated samples in normalized coordinates.
        pce: [N_samples x N_out] np.ndarray
            GPC approximation at points xi.
        """

        # handle input parameters
        if len(coeffs.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = coeffs.shape[1]

        # seed the random numbers generator
        np.random.seed()

        # generate random samples for each random input variable [N_samples x dim]
        xi = np.zeros([n_samples, self.dim])
        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == "beta":
                xi[:, i_dim] = (np.random.beta(self.pdf_shape[i_dim][0],
                                               self.pdf_shape[i_dim][1], [n_samples, 1]) * 2.0 - 1)[:, 0]
            if self.pdf_type[i_dim] == "norm" or self.pdf_type[i_dim] == "normal":
                xi[:, i_dim] = (np.random.normal(0, 1, [n_samples, 1]))[:, 0]

        # if output index list is not provided, sample all gpc outputs
        if output_idx is None:
            output_idx = np.arange(self.N_out)
            # output_idx = output_idx[np.newaxis, :]
        pce = self.get_approximation(coeffs=coeffs, xi=xi, output_idx=output_idx)

        return xi, pce

    def get_approximation(self, coeffs, xi, output_idx=None):
        """
        Calculates the gPC approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]).

        pce = SGPC.get_approximation(coeffs=None, xi=None, output_idx=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients
        xi: [1 x dim] np.ndarray
            Point in variable space to evaluate local sensitivity in normalized coordinates
        output_idx: [1 x N_out] np.ndarray, optional, default=None
            Index of output quantities to consider (Default: all outputs).

        Returns
        -------
        pce: [N_xi x N_out] np.ndarray
            Gpc approximation at normalized coordinates xi.

        Example
        -------
        pce = get_approximation([[xi_1_p1 ... xi_dim_p1] ,[xi_1_p2 ... xi_dim_p2]], np.array([[0,5,13]]))
        """

        def cpu(s):
            pce = np.zeros([xi.shape[0], s.N_out])
            for i_poly in range(s.N_poly):
                gpc_matrix_new_row = np.ones(xi.shape[0])
                for i_dim in range(s.dim):
                    gpc_matrix_new_row *= self.poly[s.poly_idx[i_poly][i_dim]][i_dim](xi[:, i_dim])
                pce += np.outer(gpc_matrix_new_row, coeffs[i_poly])
            return pce

        def gpu(s):
            # initialize matrices and parameters
            pce = np.zeros([xi.shape[0], coeffs.shape[1]])
            number_of_variables = len(s.poly[0])
            highest_degree = len(s.poly)

            # handle pointer
            polynomial_coeffs_pointer = s.poly_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            polynomial_index_pointer = s.poly_idx_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            xi_pointer = xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sim_result_pointer = pce.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sim_coeffs_pointer = coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            number_of_xi_size_t = ctypes.c_size_t(xi.shape[0])
            number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
            number_of_psi_size_t = ctypes.c_size_t(coeffs.shape[0])
            highest_degree_size_t = ctypes.c_size_t(highest_degree)
            number_of_result_vectors_size_t = ctypes.c_size_t(coeffs.shape[1])

            # handle shared object
            dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'pce.so'), mode=ctypes.RTLD_GLOBAL)
            cuda_pce = dll.polynomial_chaos_matrix
            cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
                                [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_size_t] * 5

            # evaluate CUDA implementation
            cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, sim_result_pointer,
                     sim_coeffs_pointer, number_of_psi_size_t, number_of_result_vectors_size_t,
                     number_of_variables_size_t,
                     highest_degree_size_t, number_of_xi_size_t)
            return pce

        # handle input parameters
        if len(coeffs.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = coeffs.shape[1]

        self.N_poly = self.poly_idx.shape[0]

        if len(xi.shape) == 1:
            xi = xi[:, np.newaxis]
        if np.any(output_idx):
            output_idx = np.array(output_idx)
            if len(output_idx.shape):
                output_idx = output_idx[np.newaxis, :]
            coeffs = coeffs[:, output_idx]

        if self.cpu:
            return cpu(self)
        else:
            return gpu(self)

    # noinspection PyTypeChecker
    def get_sobol_indices(self, coeffs):
        """
        Calculate the available sobol indices.

        sobol, sobol_idx = SGPC.get_sobol_indices(coeffs=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients

        Returns
        -------
        sobol: [N_sobol x N_out] np.ndarray
            Unnormalized sobol_indices
        sobol_idx: list of [N_sobol x dim] np.ndarray
            Parameter combinations in rows of sobol.
        sobol_idx_bool: list of np.ndarray of bool
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
        # size: [N_coeffs x dim]
        sobol_mask = self.poly_idx != 0

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
        max_len = max([len(self.random_vars[i]) for i in range(len(self.random_vars))])

        for i in range(order_max):
            # extract sobol coefficients of order i
            sobol_extracted, sobol_extracted_idx = get_extracted_sobol_order(sobol, sobol_idx, i + 1)

            sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten())
                                        / np.sum(var[not_nan_mask]))

            # TODO: @Konstantin: Implement the STD of the relative averaged Sobol coefficients here
            sobol_rel_order_std.append(0)

            iprint("Ratio: Sobol indices order {} / total variance: {:.4f}".format(i+1, sobol_rel_order_mean[i]), tab=1)

            # for first order indices, determine ratios of all random variables
            if i == 0:
                # deep copy
                sobol_extracted_idx_1st = sobol_extracted_idx[:]
                for j in range(sobol_extracted.shape[0]):
                    sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                    / np.sum(var[not_nan_mask]))
                    sobol_rel_1st_order_std.append(0)

                    str_out.append("\t{}{}: {:.4f}".format((max_len -
                                                            len(self.random_vars[
                                                                    sobol_extracted_idx_1st[j]])) * ' ',
                                                           self.random_vars[sobol_extracted_idx_1st[j]],
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

    # noinspection PyTypeChecker
    def get_global_sens(self, coeffs):
        """
        Determine the global derivative based sensitivity coefficients after Xiu (2009) [1].

        global_sens = SGPC.get_global_sens(coeffs)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients

        Returns
        -------
        global_sens: [dim x N_out] np.ndarray
            Global derivative based sensitivity coefficients

        Notes
        -----
        .. [1] D. Xiu, Fast Numerical Methods for Stochastic Computations: A Review,
           Commun. Comput. Phys., 5 (2009), pp. 242-272 eq. (3.14) page 255
        """

        n_max = int(len(self.poly))

        self.poly_der = [[0 for _ in range(self.dim)] for _ in range(n_max)]
        poly_der_int = [[0 for _ in range(self.dim)] for _ in range(n_max)]
        poly_int = [[0 for _ in range(self.dim)] for _ in range(n_max)]
        knots_list_1d = [0 for _ in range(self.dim)]
        weights_list_1d = [0 for _ in range(self.dim)]

        # generate quadrature points for numerical integration for each random
        # variable separately (2*N_max points for high accuracy)

        for i_dim in range(self.dim):
            # Jacobi polynomials
            if self.pdf_type[i_dim] == 'beta':
                knots_list_1d[i_dim], weights_list_1d[i_dim] = get_quadrature_jacobi_1d(2 * n_max,
                                                                                        self.pdf_shape[0][i_dim] - 1,
                                                                                        self.pdf_shape[1][i_dim] - 1)
            # Hermite polynomials
            if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == "normal":
                knots_list_1d[i_dim], weights_list_1d[i_dim] = get_quadrature_hermite_1d(2 * n_max)

        # pre-process polynomials
        for i_dim in range(self.dim):
            for i_order in range(n_max):
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_dim] = np.polyder(self.poly[i_order][i_dim])

                # evaluate poly and poly_der at quadrature points and integrate w.r.t.
                # pdf (multiply with weights and sum up)
                # saved like self.poly [N_order x dim]
                poly_int[i_order][i_dim] = np.sum(
                    np.dot(self.poly[i_order][i_dim](knots_list_1d[i_dim]), weights_list_1d[i_dim]))
                poly_der_int[i_order][i_dim] = np.sum(
                    np.dot(self.poly_der[i_order][i_dim](knots_list_1d[i_dim]), weights_list_1d[i_dim]))

        n_poly = self.poly_idx.shape[0]
        poly_der_int_multi = np.zeros([self.dim, n_poly])

        for i_sens in range(self.dim):
            for i_poly in range(n_poly):
                poly_der_int_single = 1

                # evaluate complete integral (joint basis function)
                for i_dim in range(self.dim):
                    if i_dim == i_sens:
                        poly_der_int_single *= poly_der_int[self.poly_idx[i_poly][i_dim]][i_dim]
                    else:
                        poly_der_int_single *= poly_int[self.poly_idx[i_poly][i_dim]][i_dim]

                poly_der_int_multi[i_sens, i_poly] = poly_der_int_single

        # sum up over all coefficients
        # [dim x N_points]  = [dim x N_poly] * [N_poly x N_points]
        return np.dot(poly_der_int_multi, coeffs) / (2 ** self.dim)

    # noinspection PyTypeChecker
    def get_local_sens(self, coeffs, xi):
        """
        Determine the local derivative based sensitivity coefficients in the point of interest xi
        (normalized coordinates [-1, 1]).

        local_sens = SGPC.calc_localsens(coeffs, xi)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients
        xi: [N_coeffs x N_out] np.ndarray
            Point in variable space to evaluate local sensitivity in (normalized coordinates!)

        Returns
        -------
        local_sens: [dim x N_out] np.ndarray
            Local sensitivity
        """

        n_max = len(self.poly)

        self.poly_der = [[0 for _ in range(self.dim)] for _ in range(n_max + 1)]
        poly_der_xi = [[0 for _ in range(self.dim)] for _ in range(n_max + 1)]
        poly_opvals = [[0 for _ in range(self.dim)] for _ in range(n_max + 1)]

        # pre-process polynomials
        for i_dim in range(self.dim):
            for i_order in range(n_max + 1):
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_dim] = np.polyder(self.poly[i_order][i_dim])

                # evaluate poly and poly_der at point of operation
                poly_opvals[i_order][i_dim] = self.poly[i_order][i_dim](xi[1, i_dim])
                poly_der_xi[i_order][i_dim] = self.poly_der[i_order][i_dim](xi[1, i_dim])

        n_vals = 1
        poly_sens = np.zeros([self.dim, self.N_poly])

        for i_sens in range(self.dim):
            for i_poly in range(self.N_poly):
                poly_sens_single = np.ones(n_vals)

                # construct polynomial basis according to partial derivatives
                for i_dim in range(self.dim):
                    if i_dim == i_sens:
                        poly_sens_single *= poly_der_xi[self.poly_idx[i_poly][i_dim]][i_dim]
                    else:
                        poly_sens_single *= poly_opvals[self.poly_idx[i_poly][i_dim]][i_dim]
                poly_sens[i_sens, i_poly] = poly_sens_single

        # sum up over all coefficients
        # [dim x N_points]  = [dim x N_poly]  *   [N_poly x N_points]
        return np.dot(poly_sens, coeffs)

    def get_pdf(self, coeffs, n_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = SGPC.get_pdf(coeffs, N_samples, output_idx=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients
        n_samples: int
            Number of samples used to estimate output pdf
        output_idx: [1 x N_out] np.ndarray, optional, default=None
            Index of output quantities to consider.
            If output_idx=None, all output quantities are considered

        Returns
        -------
        pdf_x: [100 x N_out] np.ndarray
            x-coordinates of output pdf (output quantity),
        pdf_y: [100 x N_out] np.ndarray
            y-coordinates of output pdf (probability density of output quantity)
        """

        # handle (N,) arrays
        if len(coeffs.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = coeffs.shape[1]

        # if output index array is not provided, determine pdfs of all outputs
        if not output_idx:
            output_idx = np.linspace(0, self.N_out - 1, self.N_out)
            output_idx = output_idx[np.newaxis, :]

        # sample gPC expansion
        samples_in, samples_out = self.get_samples(n_samples, coeffs=coeffs, output_idx=output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, self.N_out])
        pdf_y = np.zeros([100, self.N_out])

        for i_out in range(self.N_out):
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
    N_grid: int
        number of grid points
    dim: int
        number of uncertain parameters to process
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str
        string labels of the random variables
    relative_error_loocv: list of float
        relative error of the leave-one-out-cross-validation
    nan_elm: list of float
        which elements were dropped due to NaN

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str, optional, default=None
        string labels of the random variables
    """

    def __init__(self, problem, order, order_max, interaction_order):
        """
        Constructor; Initializes Regression SGPC class

        Attributes
        ----------
        relative_error_loocv: float
            Relative mean error of leave one out cross validation (loocv)

        """
        super(Reg, self).__init__(problem, order, order_max, interaction_order)

        self.relative_error_loocv = []

    def get_coeffs_expand(self, sim_results):
        """
        Determine the gPC coefficients by the regression method.

        coeffs = get_coeffs_expand(sim_results)

        Parameters
        ----------
        sim_results: [N_grid x N_out] np.ndarray of float
            results from simulations with N_out output quantities,

        Returns
        -------
        coeffs: [N_coeffs x N_out] np.ndarray of float
            gPC coefficients
        """

        iprint('Determine gPC coefficients...')

        # handle (N,) arrays
        if len(sim_results.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = sim_results.shape[1]

        try:
            return np.dot(self.gpc_matrix_inv, sim_results.T)
        except ValueError:
            wprint("Please check format of parameter sim_results: [N_grid x N_out] np.ndarray.")
            raise

    def get_loocv(self, sim_results):
        """
        Perform leave one out cross validation of gPC with maximal 100 points
        and add result to self.relative_error_loocv.

        relative_error_loocv = GPC.get_loocv(sim_results)

        Parameters
        ----------
        sim_results: [n_grid x n_out] np.ndarray
            Results from n_grid simulations with n_out output quantities

        Returns
        -------
        relative_error_loocv: float
            relative mean error of leave one out cross validation
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
    Quadrature gPC subclass

    Quad(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None)

    Attributes
    ----------
    N_grid: int
        number of grid points
    dim: int
        number of uncertain parameters to process
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str
        string labels of the random variables

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str, optional, default=None
        string labels of the random variables
    """

    def __init__(self, pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None):
        super(Reg, self).__init__(problem, order, order_max, interaction_order)

    def get_coeffs_expand(self, sim_results):
        """
        Determine the gPC coefficients by the quadrature method

        coeffs = get_coeffs_expand(self, sim_results)

        Parameters
        ----------
        sim_results: [N_grid x N_out] np.ndarray of float
            results from simulations with N_out output quantities

        Returns
        -------
        coeffs: [N_coeffs x N_out] np.ndarray of float
            gPC coefficients
        """

        iprint('Determine gPC coefficients...')

        # handle (N,) arrays
        if len(sim_results.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = sim_results.shape[1]

        # check if quadrature rule (grid) fits to the probability density distribution (pdf)
        grid_pdf_fit = True
        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == 'beta':
                if not (self.grid.grid_type[i_dim] == 'jacobi'):
                    grid_pdf_fit = False
                    break
            elif (self.pdf_type[i_dim] == 'norm') or (self.pdf_type[i_dim] == 'normal'):
                if not (self.grid.gridtype[i_dim] == 'hermite'):
                    grid_pdf_fit = False
                    break

        # if not, calculate joint pdf
        if not grid_pdf_fit:
            joint_pdf = np.ones(self.grid.coords_norm.shape)

            for i_dim in range(self.dim):
                if self.pdf_type[i_dim] == 'beta':
                    joint_pdf[:, i_dim] = get_pdf_beta(self.grid.coords_norm[:, i_dim],
                                                       self.pdf_shape[0][i_dim],
                                                       self.pdf_shape[1][i_dim], -1, 1)

                if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == 'normal':
                    joint_pdf[:, i_dim] = scipy.stats.norm.pdf(self.grid.coords_norm[:, i_dim])

            joint_pdf = np.array([np.prod(joint_pdf, axis=1)]).transpose()

            # weight sim_results with the joint pdf
            sim_results = sim_results * joint_pdf * 2 ** self.dim

        # scale rows of gpc matrix with quadrature weights
        gpc_matrix_weighted = np.dot(np.diag(self.grid.weights), self.gpc_matrix)

        # determine gpc coefficients [N_coeffs x N_output]
        return np.dot(sim_results.transpose(), gpc_matrix_weighted).transpose()
