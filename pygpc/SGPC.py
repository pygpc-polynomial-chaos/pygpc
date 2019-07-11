# -*- coding: utf-8 -*-
import time
import random
import numpy as np
import scipy.stats
from .GPC import *
from .io import iprint, wprint
from .misc import display_fancy_bar
from .misc import get_array_unique_rows
from .Basis import *
from scipy.special import binom
from .sobol_saltelli import get_sobol_indices_saltelli
from .sobol_saltelli import saltelli_sampling


class SGPC(GPC):
    """
    Sub-class for standard gPC (SGPC)

    Attributes
    ----------
    order: list of int [dim]
        Maximum individual expansion order [order_1, order_2, ..., order_dim].
        Generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        Maximum global expansion order.
        The maximum expansion order considers the sum of the orders of combined polynomials together with the
        chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
        monomial orders.
    order_max_norm: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
    interaction_order: int
        Number of random variables, which can interact with each other.
        All polynomials are ignored, which have an interaction order greater than the specified
    interaction_order_current: int
        Number of random variables currently interacting with respect to the highest order.
        (interaction_order_current <= interaction_order)
        The parameters for lower orders are all interacting with "interaction order".
    options : dict
        Options of gPC
    validation: ValidationSet object (optional)
        Object containing a set of validation points and corresponding solutions. Can be used
        to validate gpc approximation setting options["error_type"]="nrmsd".
        - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
        - results: ndarray [n_grid x n_out] results
    """

    def __init__(self, problem, order, order_max, order_max_norm, interaction_order, options,
                 interaction_order_current=None, validation=None):
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
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        order_max_norm: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        interaction_order_current: int, optional, default: interaction_order
            Number of random variables currently interacting with respect to the highest order.
            (interaction_order_current <= interaction_order)
            The parameters for lower orders are all interacting with "interaction order".
        options : dict
            Options of gPC
        validation: ValidationSet object (optional)
            Object containing a set of validation points and corresponding solutions. Can be used
            to validate gpc approximation setting options["error_type"]="nrmsd".
            - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
            - results: ndarray [n_grid x n_out] results

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle 
           regression. Journal of Computational Physics, 230(6), 2345-2367.
        """
        super(SGPC, self).__init__(problem, options, validation)

        self.order = order
        self.order_max = order_max
        self.order_max_norm = order_max_norm
        self.interaction_order = interaction_order

        if interaction_order_current is None:
            self.interaction_order_current = interaction_order
        else:
            self.interaction_order_current = interaction_order_current

        self.basis = Basis()
        self.basis.init_basis_sgpc(problem=problem,
                                   order=order,
                                   order_max=order_max,
                                   order_max_norm=order_max_norm,
                                   interaction_order=interaction_order,
                                   interaction_order_current=interaction_order_current)

    @staticmethod
    def get_mean(coeffs=None, samples=None):
        """
        Calculate the expected mean value. Provide either gPC coeffs or a certain number of samples.

        mean = SGPC.get_mean(coeffs)

        Parameters
        ----------
        coeffs : ndarray of float [n_basis x n_out], optional, default: None
            GPC coefficients
        samples : ndarray of float [n_samples x n_out], optional, default: None
            Model evaluations from gPC approximation

        Returns
        -------
        mean: ndarray of float [1 x n_out]
            Expected value of output quantities
        """
        if coeffs is not None:
            mean = coeffs[0, ]

        elif samples is not None:
            mean = np.mean(samples, axis=0)

        else:
            raise AssertionError("Provide either ""coeffs"" or ""samples"" to determine mean!")

        mean = mean[np.newaxis, :]

        return mean

    @staticmethod
    def get_std(coeffs=None, samples=None):
        """
        Calculate the standard deviation. Provide either gPC coeffs or a certain number of samples.

        std = SGPC.get_std(coeffs)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out], optional, default: None
            GPC coefficients
        samples : ndarray of float [n_samples x n_out], optional, default: None
            Model evaluations from gPC approximation

        Returns
        -------
        std: ndarray of float [1 x n_out]
            Standard deviation of output quantities
        """
        if coeffs is not None:
            std = np.sqrt(np.sum(np.square(coeffs[1:]), axis=0))

        elif samples is not None:
            std = np.std(samples, axis=0)

        else:
            raise AssertionError("Provide either ""coeffs"" or ""samples"" to determine standard deviation!")

        std = std[np.newaxis, :]

        return std

    # noinspection PyTypeChecker
    def get_sobol_indices(self, coeffs, algorithm="standard", n_samples=1e4):
        """
        Calculate the available sobol indices from the gPC coefficients (standard) or by sampling.
        In case of sampling, the Sobol indices are calculated up to second order.

        sobol, sobol_idx, sobol_idx_bool = SGPC.get_sobol_indices(coeffs, algorithm="standard", n_samples=1e4)

        Parameters
        ----------
        coeffs:  ndarray of float [n_basis x n_out]
            GPC coefficients
        algorithm : str, optional, default: "standard"
            Algorithm to determine the Sobol indices
            - "standard": Sobol indices are determined from the gPC coefficients
            - "sampling": Sobol indices are determined from sampling using Saltelli's Sobol sampling sequence [1, 2, 3]
        n_samples : int, optional, default: 1e4
            Number of samples to determine Sobol indices by sampling. The efficient number of samples
            increases to n_samples * (2*dim + 2) in Saltelli's Sobol sampling sequence.

        Returns
        -------
        sobol: ndarray of float [n_sobol x n_out]
            Normalized Sobol indices w.r.t. total variance
        sobol_idx: list of ndarray of int [n_sobol x (n_sobol_included)]
            Parameter combinations in rows of sobol.
        sobol_idx_bool: ndarray of bool [n_sobol x dim]
            Boolean mask which contains unique multi indices.

        Notes
        -----
        .. [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
               mathematical models and their Monte Carlo estimates."  Mathematics
               and Computers in Simulation, 55(1-3):271-280,
               doi:10.1016/S0378-4754(00)00270-6.
        .. [2] Saltelli, A. (2002).  "Making best use of model evaluations to
               compute sensitivity indices."  Computer Physics Communications,
               145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.
        .. [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
               S. Tarantola (2010).  "Variance based sensitivity analysis of model
               output.  Design and estimator for the total sensitivity index."
               Computer Physics Communications, 181(2):259-270,
               doi:10.1016/j.cpc.2009.09.018.
        """

        # iprint("Determining Sobol indices...", tab=0)

        if algorithm == "standard":
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
            multi_indices = np.array([list(map(lambda _b:_b.p["i"], b_row)) for b_row in self.basis.b])
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

            var = self.get_std(coeffs=coeffs) ** 2

            sobol = sobol / var

        elif algorithm == "sampling":

            if self.p_matrix is None:
                dim = self.problem.dim
            else:
                dim = self.problem_original.dim

            # generate sobol sequence (original parameter space, scaled to [-1, 1])
            coords_norm = 2 * saltelli_sampling(n_samples=n_samples, dim=dim, calc_second_order=True) - 1

            # run model evaluations
            res = self.get_approximation(coeffs=coeffs, x=coords_norm)

            # determine sobol indices
            sobol, sobol_idx, sobol_idx_bool = get_sobol_indices_saltelli(y=res,
                                                                          dim=dim,
                                                                          calc_second_order=True,
                                                                          num_resamples=100,
                                                                          conf_level=0.95)

            # sort
            idx = np.flip(np.argsort(sobol[:, 0], axis=0))
            sobol = sobol[idx, :]
            sobol_idx = [sobol_idx[i] for i in idx]
            sobol_idx_bool = sobol_idx_bool[idx, :]

        else:
            raise AssertionError("Please provide valid algorithm argument (""standard"" or ""sampling"")")

        return sobol, sobol_idx, sobol_idx_bool

    def get_sobol_composition(self, sobol, sobol_idx_bool):
        """
        Determine average ratios of Sobol indices over all output quantities:
        (i) over all orders and (e.g. 1st: 90%, 2nd: 8%, 3rd: 2%)
        (ii) for the 1st order indices w.r.t. each random variable. (1st: x1: 50%, x2: 40%)

        sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std, sobol_rel_1st_order_mean, sobol_rel_1st_order_std
        = SGPC.get_sobol_composition(coeffs, sobol, sobol_idx, sobol_idx_bool)

        Parameters
        ----------
        sobol: ndarray of float [n_sobol x n_out]
            Unnormalized sobol_indices
        sobol_idx_bool: list of ndarray of bool
            Boolean mask which contains unique multi indices.

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

        sobol_idx = [np.argwhere(sobol_idx_bool[i, :]).flatten() for i in range(sobol_idx_bool.shape[0])]

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
        max_len = max([len(p) for p in self.problem.parameters_random])

        for i in range(order_max):
            # extract sobol coefficients of order i
            sobol_extracted, sobol_extracted_idx = self.get_extracted_sobol_order(sobol, sobol_idx_bool, i + 1)

            # determine average sobol index over all elements
            sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten()) /
                                        np.sum(var[not_nan_mask]))

            sobol_rel_order_std.append(np.std(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten() /
                                              var[not_nan_mask]))

            iprint("Ratio: Sobol indices order {} / total variance: {:.4f} +- {:.4f}"
                   .format(i+1, sobol_rel_order_mean[i], sobol_rel_order_std[i]), tab=0, verbose=self.verbose)

            # for 1st order indices, determine ratios of all random variables
            if i == 0:
                sobol_extracted_idx_1st = sobol_extracted_idx[:]
                for j in range(sobol_extracted.shape[0]):
                    sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                    / np.sum(var[not_nan_mask]))
                    sobol_rel_1st_order_std.append(0)

                    str_out.append("\t{}{}: {:.4f}"
                                   .format((max_len - len(self.problem.parameters_random.keys()[sobol_extracted_idx_1st[j][0]])) * ' ',
                                           self.problem.parameters_random.keys()[sobol_extracted_idx_1st[j][0]],
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
        if self.verbose:
            for j in range(len(str_out)):
                print(str_out[j])

        return sobol_rel_order_mean, sobol_rel_order_std, \
               sobol_rel_1st_order_mean, sobol_rel_1st_order_std, \
               sobol_rel_2nd_order_mean, sobol_rel_2nd_order_std

    @staticmethod
    def get_extracted_sobol_order(sobol, sobol_idx_bool, order=1):
        """
        Extract Sobol indices with specified order from Sobol data.

        sobol_1st, sobol_idx_1st = SGPC.get_extracted_sobol_order(sobol, sobol_idx, order=1)

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

    # noinspection PyTypeChecker
    def get_global_sens(self, coeffs, algorithm="standard", n_samples=1e5):
        """
        Determine the global derivative based sensitivity coefficients after Xiu (2009) [1]
        from the gPC coefficients (standard) or by sampling.

        global_sens = SGPC.get_global_sens(coeffs, algorithm="standard", n_samples=1e5)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out], optional, default: None
            GPC coefficients
        algorithm : str, optional, default: "standard"
            Algorithm to determine the Sobol indices
            - "standard": Sobol indices are determined from the gPC coefficients
            - "sampling": Sobol indices are determined from sampling using Saltelli's Sobol sampling sequence [1, 2, 3]
        n_samples : int, optional, default: 1e4
            Number of samples

        Returns
        -------
        global_sens: ndarray [dim x n_out]
            Global derivative based sensitivity coefficients

        Notes
        -----
        .. [1] D. Xiu, Fast Numerical Methods for Stochastic Computations: A Review,
           Commun. Comput. Phys., 5 (2009), pp. 242-272 eq. (3.14) page 255
        """

        if algorithm == "standard":
            b_int_global = np.zeros([self.problem.dim, self.basis.n_basis])

            # construct matrix with integral expressions [n_basis x dim]
            b_int = np.array([list(map(lambda _b: _b.fun_int, b_row)) for b_row in self.basis.b])
            b_int_der = np.array([list(map(lambda _b: _b.fun_der_int, b_row)) for b_row in self.basis.b])

            for i_sens in range(self.problem.dim):
                # replace column with integral expressions from derivative of parameter[i_dim]
                tmp = copy.deepcopy(b_int)
                tmp[:, i_sens] = b_int_der[:, i_sens]

                # determine global integral expression
                b_int_global[i_sens, :] = np.prod(tmp, axis=1)

            global_sens = np.dot(b_int_global, coeffs) / (2 ** self.problem.dim)
            # global_sens = np.dot(b_int_global, coeffs)

        elif algorithm == "sampling":
            # generate sample coordinates (original parameter space)
            if self.p_matrix is not None:
                grid = RandomGrid(parameters_random=self.problem_original.parameters_random,
                                  options={"n_grid": n_samples, "seed": None})
            else:
                grid = RandomGrid(parameters_random=self.problem.parameters_random,
                                  options={"n_grid": n_samples, "seed": None})

            local_sens = self.get_local_sens(coeffs, grid.coords_norm)

            # # transform the coordinates to the reduced parameter space
            # coords_norm = np.dot(coords_norm, self.p_matrix.transpose() / self.p_matrix_norm[np.newaxis, :])
            #
            # # construct gPC gradient matrix [n_samples x n_basis x dim_red]
            # gpc_matrix_gradient = self.calc_gpc_matrix(b=self.basis.b, x=coords_norm, gradient=True)
            #
            # # determine gradient in each sampling point [n_samples x n_out x dim_red]
            # grad_samples_projected = np.dot(gpc_matrix_gradient.transpose(2, 0, 1), coeffs).transpose(1, 2, 0)
            #
            # # project the gradient back to the original parameter space if necessary [n_samples x n_out x dim]
            # grad_samples = np.dot(grad_samples_projected, self.p_matrix / self.p_matrix_norm[:, np.newaxis])

            # average the results and reshape [dim x n_out]
            global_sens = np.mean(local_sens, axis=0).transpose()

        else:
            raise AssertionError("Please provide valid algorithm argument (""standard"" or ""sampling"")")

        return global_sens

    # noinspection PyTypeChecker
    def get_local_sens(self, coeffs, x=None):
        """
        Determine the local derivative based sensitivity coefficients in the point of interest x
        (normalized coordinates [-1, 1]).

        local_sens = SGPC.calc_localsens(coeffs, x)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        x: ndarray of float [dim], optional, default: center of parameter space
            Point in variable space to evaluate local sensitivity in (normalized coordinates [-1, 1])

        Returns
        -------
        local_sens: ndarray [dim x n_out]
            Local sensitivity of output quantities in point x
        """

        if x is None:
            x = np.zeros(self.problem.dim)[np.newaxis, :]

        # project coordinate to reduced parameter space if necessary
        if self.p_matrix is not None:
            x = np.dot(x, self.p_matrix.transpose() / self.p_matrix_norm[np.newaxis, :])

        # construct gPC gradient matrix [n_samples x n_basis x dim(_red)]
        gpc_matrix_gradient = self.calc_gpc_matrix(b=self.basis.b, x=x, gradient=True)

        local_sens = np.dot(gpc_matrix_gradient.transpose(2, 0, 1), coeffs).transpose(1, 2, 0)

        # project the gradient back to the original space if necessary
        if self.p_matrix is not None:
            local_sens = np.dot(local_sens, self.p_matrix / self.p_matrix_norm[:, np.newaxis])

        return local_sens


class Reg(SGPC):
    """
    Regression gPC subclass

    Reg(problem, order, order_max, interaction_order, fn_results=None)

    Attributes
    ----------
    solver: str
        Solver to determine the gPC coefficients
        - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
        - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
    settings: dict
        Solver settings
        - 'Moore-Penrose' ... None
        - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
        - 'NumInt' ... None
    """

    def __init__(self, problem, order, order_max, order_max_norm, interaction_order, options,
                 interaction_order_current=None, validation=None):
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
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        order_max_norm: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        interaction_order_current: int, optional, default: interaction_order
            Number of random variables currently interacting with respect to the highest order.
            (interaction_order_current <= interaction_order)
            The parameters for lower orders are all interacting with "interaction order".
        options : dict
            Options of gPC
        validation: ValidationSet object (optional)
            Object containing a set of validation points and corresponding solutions. Can be used
            to validate gpc approximation setting options["error_type"]="nrmsd".
            - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
            - results: ndarray [n_grid x n_out] results

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle 
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> gpc = pygpc.Reg(problem=problem,
        >>>                 order=[7, 6],
        >>>                 order_max=5,
        >>>                 order_max_norm=1,
        >>>                 interaction_order=2,
        >>>                 interaction_order_current=1
        >>>                 fn_results="/tmp/my_results")
        """

        if interaction_order_current is None:
            self.interaction_order_current = interaction_order
        else:
            self.interaction_order_current = interaction_order_current

        super(Reg, self).__init__(problem=problem,
                                  order=order,
                                  order_max=order_max,
                                  order_max_norm=order_max_norm,
                                  interaction_order=interaction_order,
                                  interaction_order_current=interaction_order_current,
                                  options=options,
                                  validation=validation)

        self.solver = 'Moore-Penrose'   # Default solver
        self.settings = None            # Default Solver settings


class Quad(SGPC):
    """
    Quadrature SGPC sub-class
    """

    def __init__(self, problem, order, order_max, order_max_norm, interaction_order, options,
                 interaction_order_current=None, validation=None):
        """
        Constructor; Initializes Quadrature SGPC sub-class

        Quad(problem, order, order_max, order_max_norm, interaction_order)

        Parameters
        ----------
        problem: Problem class instance
            GPC Problem under investigation
        order: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        order_max_norm: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        interaction_order: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        interaction_order_current: int, optional, default: interaction_order
            Number of random variables currently interacting with respect to the highest order.
            (interaction_order_current <= interaction_order)
            The parameters for lower orders are all interacting with "interaction order".
        options : dict
            Options of gPC
        validation: ValidationSet object (optional)
            Object containing a set of validation points and corresponding solutions. Can be used
            to validate gpc approximation setting options["error_type"]="nrmsd".
            - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
            - results: ndarray [n_grid x n_out] results

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle 
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> gpc = pygpc.Quad(problem=problem,
        >>>                  order=[7, 6],
        >>>                  order_max=5,
        >>>                  order_max_norm=1,
        >>>                  interaction_order=2,
        >>>                  interaction_order_current=1,
        >>>                  fn_results="/tmp/my_results")
        """

        if interaction_order_current is None:
            self.interaction_order_current = interaction_order
        else:
            self.interaction_order_current = interaction_order_current

        super(Quad, self).__init__(problem=problem,
                                   order=order,
                                   order_max=order_max,
                                   order_max_norm=order_max_norm,
                                   interaction_order=interaction_order,
                                   interaction_order_current=interaction_order_current,
                                   options=options,
                                   validation=validation)

        self.solver = 'NumInt'  # Default solver
        self.settings = None    # Default solver settings
