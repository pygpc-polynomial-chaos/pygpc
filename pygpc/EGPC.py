import numpy as np
from .GPC import *


class EGPC(GPC):
    """

    """
    def __init__(self):
        """

        """
        super(EGPC, self).__init__()

    @staticmethod
    def get_mean(coeffs):
        """
        Calculate the expected mean value.

        mean = EGPC.get_mean(coeffs)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray
            Gpc coefficients

        Returns
        -------
        mean: [1 x N_out] np.ndarray
            Expected mean value
        """

        mean = 1

        return mean

    @staticmethod
    def get_standard_deviation(coeffs):
        """
        Calculate the standard deviation.

        std = EGPC.get_standard_deviation(coeffs)

        Parameters
        ----------
        coeffs: np.array of float [N_coeffs x N_out]
            Gpc coefficients

        Returns
        -------
        std: [1 x N_out] np.ndarray
            Standard deviation
        """

        std = 1

        return std

    @staticmethod
    def get_samples(self, coeffs=None, n_samples=100, output_idx=None):
        """
        Randomly sample the gPC expansion to determine output pdfs in specific points.

        xi = EGPC.get_pdf_mc(N_samples, coeffs=None, output_idx=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray, optional, default=None
            gPC coefficients
        n_samples: int
            Number of random samples drawn from the respective input pdfs.
        output_idx: [1 x N_out] np.ndarray, optional, default=None
            Index of output quantities to consider.

        Returns
        -------
        xi: [N_samples x dim] np.ndarray
            Generated samples in normalized coordinates.
        pce: [N_samples x N_out] np.ndarray
            GPC approximation at points xi.
        """

        xi = 1

        pce = 1

        return xi, pce

    def get_approximation(self, coeffs=None, xi=None, output_idx=None):
        """
        Calculates the gPC approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]).

        pce = EGPC.get_approximation(coeffs=None, xi=None, output_idx=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray, optional, default=None
            Gpc coefficients
        xi: [1 x dim] np.ndarray, optional, default=None
            Point in variable space to evaluate local sensitivity in normalized coordinates
        output_idx: [1 x N_out] np.ndarray, optional, default=None
            Index of output quantities to consider (Default: all outputs).

        Returns
        -------
        pce: [N_xi x N_out] np.ndarray
            Gpc approximation at normalized coordinates xi.
        """

        def cpu(s):
            pce = 1
            return pce

        def gpu(s):
            pce = 1
            return pce

        if self.cpu:
            return cpu(self)
        else:
            return gpu(self)

    # noinspection PyTypeChecker
    @staticmethod
    def get_sobol_indices(self, coeffs=None):
        """
        Calculate the available sobol indices.

        sobol, sobol_idx = EGPC.get_sobol_indices(coeffs=None)

        Parameters
        ----------
        coeffs: [N_coeffs x N_out] np.ndarray, optional, default=None
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

        sobol = 1
        sobol_idx = 1
        sobol_idx_bool = 1

        return sobol, sobol_idx, sobol_idx_bool

    def get_sobol_composition(self, sobol=None, sobol_idx=None, sobol_idx_bool=None):
        """
        Determine average ratios of Sobol indices over all output quantities:
        (i) over all orders and (e.g. 1st: 90%, 2nd: 8%, 3rd: 2%)
        (ii) for the 1st order indices w.r.t. each random variable. (1st: x1: 50%, x2: 40%)

        sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std, sobol_rel_1st_order_mean, sobol_rel_1st_order_std
        = EGPC.get_sobol_composition(coeffs=None, sobol=None, sobol_idx=None, sobol_idx_bool=None)

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
        sobol = 1
        sobol_idx = 1
        sobol_rel_order_mean = 1
        sobol_rel_order_std = 1
        sobol_rel_1st_order_mean = 1
        sobol_rel_1st_order_std = 1

        return sobol, sobol_idx, \
               sobol_rel_order_mean, sobol_rel_order_std, \
               sobol_rel_1st_order_mean, sobol_rel_1st_order_std

    # noinspection PyTypeChecker
    def get_global_sens(self, coeffs):
        """
        Determine the global derivative based sensitivity coefficients after Xiu (2009) [1].

        global_sens = EGPC.get_global_sens(coeffs)

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

        global_sens = 1

        return global_sens

    # noinspection PyTypeChecker
    def get_local_sens(self, coeffs, xi):
        """
        Determine the local derivative based sensitivity coefficients in the point of interest xi
        (normalized coordinates [-1, 1]).

        local_sens = EGPC.calc_localsens(coeffs, xi)

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

        local_sens = 1

        return local_sens

    def get_pdf(self, coeffs, n_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = EGPC.get_pdf(coeffs, N_samples, output_idx=None)

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
        samples_in, samples_out = self.get_samples(n_samples=n_samples, coeffs=coeffs, output_idx=output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, self.N_out])
        pdf_y = np.zeros([100, self.N_out])

        for i_out in range(self.N_out):
            kde = scipy.stats.gaussian_kde(samples_out.transpose(), bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)
            pdf_y[:, i_out] = kde(pdf_x[:, i_out])

        return pdf_x, pdf_y
