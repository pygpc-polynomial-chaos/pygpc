# -*- coding: utf-8 -*-
import copy
import h5py
import time
import random
from .Grid import *
from .SGPC import *
from .misc import get_cartesian_product
from .misc import display_fancy_bar
from .misc import nrmsd
from .misc import mat2ten
from .misc import ten2mat
from .misc import increment_basis
from .ValidationSet import *
from .Computation import *
from .Classifier import *
import numpy as np
import fastmat as fm
import scipy.stats
from sklearn import linear_model


class MEGPC(object):
    """
    General Multi-Element gPC base class

    Attributes
    ----------
    problem: Problem class instance
        GPC Problem under investigation
    grid: Grid class instance
        Grid of the derived gPC approximation
    validation: ValidationSet object (optional)
        Object containing a set of validation points and corresponding solutions. Can be used
        to validate gpc approximation setting options["error_type"]="nrmsd".
        - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
        - results: ndarray [n_grid x n_out] results
    n_grid: int or list of int
        Number of grid points (for iterative solvers, this is a list of its history)
    solver: str
        Default solver to determine the gPC coefficients (can be chosen during GPC.solve)
        - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
        - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        - 'LarsLasso' ... {"alpha": float 0...1} Regularization parameter
        - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
    gpu: bool
        Flag to execute the calculation on the gpu
    verbose: bool
        boolean value to determine if to print out the progress into the standard output
    fn_results : string, optional, default=None
        If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
    options : dict
        Options of gPC algorithm
    """

    def __init__(self, problem, options, validation=None):
        """
        Constructor; Initializes MEGPC class

        Parameters
        ----------
        problem: Problem class instance
            GPC Problem under investigation
        options : dict
            Options of gPC algorithm
        validation: ValidationSet object (optional)
            Object containing a set of validation points and corresponding solutions. Can be used
            to validate gpc approximation setting options["error_type"]="nrmsd".
            - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
            - results: ndarray [n_grid x n_out] results
        """

        # objects
        self.problem = problem
        # self.sub_problems = None
        self.grid = None
        self.validation = validation
        self.gpc = None
        self.classifier = None
        self.domains = None

        # arrays
        self.n_grid = []
        self.n_out = []
        self.n_gpc = None
        self.relative_error_loocv = []
        self.relative_error_nrmsd = []
        self.error = []
        self.mask_res_domains = None
        self.mask_grad_res_domains = None

        # options
        self.gradient = options["gradient_enhanced"]
        self.solver = None
        self.settings = None
        self.gpu = None
        self.verbose = True
        if "fn_results" not in options.keys():
            options["fn_results"] = None
        self.fn_results = options["fn_results"]
        self.options = options
        self.matlab_model = options["matlab_model"]

    def init_classifier(self, coords, results, algorithm, options):
        """
        Initializes Classifier object in MEGPC class

        Parameters
        ----------
        coords : ndarray of float [n_grid, n_dim]
            Set of n_grid parameter combinations
        results : ndarray [n_grid x n_out]
            Results of the model evaluation
        algorithm : str, optional, default: "learning"
            Algorithm to classify grid points
            - "learning" ... 2-step procedure with unsupervised and supervised learning
            - ...
        options : dict, optional, default=None
            Classifier options
        """
        self.classifier = Classifier(coords=coords,
                                     results=results,
                                     algorithm=algorithm,
                                     options=options)

        self.domains = self.classifier.domains
        self.n_gpc = len(np.unique(self.domains))

        if self.gradient:
            self.mask_res_domains = np.hstack((self.domains,
                                               self.domains.repeat(self.problem.dim)))
            self.mask_grad_res_domains = np.hstack((np.array([np.nan]).repeat(len(self.domains)),
                                                    self.domains.repeat(self.problem.dim)))
        else:
            self.mask_res_domains = self.domains

    def update_classifier(self, coords, results):
        """
        Updates self.classifier and keeps the existing class labels

        Parameters
        ----------
        coords : ndarray of float [n_grid, n_dim]
            Set of n_grid parameter combinations
        results : ndarray [n_grid x n_out]
            Results of the model evaluation
        """
        self.classifier.update(coords=coords, results=results)
        self.domains = self.classifier.domains
        self.n_gpc = len(np.unique(self.domains))

        if self.gradient:
            self.mask_res_domains = np.hstack((self.domains,
                                               self.domains.repeat(self.problem.dim)))
            self.mask_grad_res_domains = np.hstack((np.array([np.nan]).repeat(len(self.domains)),
                                                    self.domains.repeat(self.problem.dim)))
        else:
            self.mask_res_domains = self.domains

    def add_sub_gpc(self, problem, order, order_max, order_max_norm, interaction_order,
                    interaction_order_current, options, domain, validation=None):
        """
        Add sub-gPC
        """
        if self.gpc is None:
            if self.n_gpc is not None:
                self.gpc = [None for _ in range(self.n_gpc)]
            else:
                self.gpc = [0 for _ in range(domain)]
        elif len(self.gpc) < domain:
            self.gpc = self.gpc + [None for _ in range(domain - len(self.gpc))]

        # create sub-gpc objects
        self.gpc[domain] = Reg(problem=problem,
                               order=order,
                               order_max=order_max,
                               order_max_norm=order_max_norm,
                               interaction_order=interaction_order,
                               interaction_order_current=interaction_order_current,
                               options=options,
                               validation=validation)

    def init_gpc_matrices(self):
        """
        Sets self.gpc_matrix with given self.basis and self.grid
        """

        for gpc in self.gpc:
            gpc.init_gpc_matrix()

    def assign_grids(self):
        """
        Assign sub-grids to sub-gPCs (including transformation in case of projection)
        """
        # update domain indices if grid points were added
        if len(self.domains) != self.grid.coords_norm.shape[0]:
            self.domains = self.classifier.predict(self.grid.coords_norm)

        for d in np.unique(self.domains):
            coords = self.grid.coords[self.domains == d, :]
            coords_norm = self.grid.coords_norm[self.domains == d, :]
            coords_id = np.array(self.grid.coords_id)[self.domains == d].tolist()

            # transform variables of original grid to reduced parameter space
            if self.gpc[d].p_matrix is not None:
                coords = np.dot(coords, self.gpc[d].p_matrix.transpose())
                coords_norm = np.dot(coords_norm, self.gpc[d].p_matrix.transpose() /
                                     self.gpc[d].p_matrix_norm[np.newaxis, :])

            if self.grid.coords_gradient is not None:
                coords_gradient = self.grid.coords_gradient[self.domains == d, :, :]
                coords_gradient_norm = self.grid.coords_gradient_norm[self.domains == d, :, :]
                coords_gradient_id = np.array(self.grid.coords_gradient_id)[self.domains == d].tolist()
            else:
                coords_gradient = None
                coords_gradient_norm = None
                coords_gradient_id = None

            self.gpc[d].grid = Grid(parameters_random=self.problem.parameters_random,
                                    coords=coords,
                                    coords_norm=coords_norm,
                                    coords_gradient=coords_gradient,
                                    coords_gradient_norm=coords_gradient_norm,
                                    coords_id=coords_id,
                                    coords_gradient_id=coords_gradient_id)

    def loocv(self, results, error_norm="relative", domain=None):
        """
        Perform leave-one-out cross validation of gPC approximation and add error value to self.relative_error_loocv.
        The loocv error is calculated analytically after eq. (35) in [1] but omitting the "1 - " term, i.e. it
        corresponds to 1 - Q^2.

        relative_error_loocv = GPC.loocv(sim_results, coeffs)

        .. math::
           \\epsilon_{LOOCV} = \\frac{\\frac{1}{N}\sum_{i=1}^N \\left( \\frac{y(\\xi_i) - \hat{y}(\\xi_i)}{1-h_i}
           \\right)^2}{\\frac{1}{N-1}\sum_{i=1}^N \\left( y(\\xi_i) - \\bar{y} \\right)^2}

        with

        .. math::
           \\mathbf{h} = \mathrm{diag}(\\mathbf{\\Psi} (\\mathbf{\\Psi}^T \\mathbf{\\Psi})^{-1} \\mathbf{\\Psi}^T)

        Parameters
        ----------
        results : ndarray of float [n_grid x n_out]
            Results from n_grid simulations with n_out output quantities
        error_norm : str, optional, default="relative"
            Decide if error is determined "relative" or "absolute"
        domain : int, optional, default: None
            Determine error in specified domain only. Default: None (all domains)

        Returns
        -------
        relative_error_loocv : float
            Relative mean error of leave one out cross validation

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2010). An adaptive algorithm to build up sparse polynomial chaos expansions
           for stochastic finite element analysis. Probabilistic Engineering Mechanics, 25(2), 183-197.
        """

        n_loocv = 25

        if domain is not None:
            results_domain = copy.deepcopy(results[self.domains == domain, :])
            domain_idx = copy.deepcopy(domain)
        else:
            results_domain = copy.deepcopy(results)

        # define number of performed cross validations (max 25)
        n_loocv_points = np.min((results_domain.shape[0], n_loocv))

        # make list of indices, which are randomly sampled (this index is w.r.t. to all points if domain is None)
        loocv_point_idx = random.sample(list(range(results_domain.shape[0])), n_loocv_points)

        start = time.time()
        relative_error = np.zeros(n_loocv_points)

        for i in range(n_loocv_points):

            if domain is None:
                # determine domain of loocv point
                domain_idx = int(self.classifier.predict(self.grid.coords_norm[loocv_point_idx[i]][np.newaxis, :]))
                results_domain = results[self.domains == domain_idx, ]

            # determine row in sub-gPC matrix of loocv point
            loocv_point_idx_domain = np.sum(np.array(self.domains == domain_idx)[0:loocv_point_idx[i]])

            # get mask of eliminated row
            mask = np.arange(results_domain.shape[0]) != loocv_point_idx_domain

            # select right gpc matrix
            matrix = self.gpc[domain_idx].gpc_matrix

            # determine gpc coefficients (this takes a lot of time for large problems)
            coeffs_loo = self.gpc[domain_idx].solve(results=results_domain[mask, :],
                                                    solver=self.options["solver"],
                                                    matrix=matrix[mask, :],
                                                    settings=self.options["settings"],
                                                    verbose=False)

            sim_results_temp = results_domain[loocv_point_idx_domain, :]

            if error_norm == "relative":
                norm = scipy.linalg.norm(sim_results_temp)
            else:
                norm = 1.

            # determine error
            relative_error[i] = scipy.linalg.norm(sim_results_temp - np.dot(matrix[loocv_point_idx_domain, :],
                                                                            coeffs_loo)) \
                                / norm
            display_fancy_bar("LOOCV", int(i + 1), int(n_loocv_points))

        # store result in relative_error_loocv
        relative_error_loocv = np.mean(relative_error)
        iprint("LOOCV computation time: {} sec".format(time.time() - start), tab=0, verbose=True)

        return relative_error_loocv

    def validate(self, coeffs, results=None, gradient_results=None, domain=None, output_idx=None):
        """
        Validate gPC approximation using the ValidationSet object contained in the Problem object.
        Determines the normalized root mean square deviation between the gpc approximation and the
        original model. Skips this step if no validation set is present

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_coeffs x n_out]
            GPC coefficients
        results: ndarray of float [n_grid x n_out]
            Results from n_grid simulations with n_out output quantities
        gradient_results : ndarray of float [n_grid x n_out x dim], optional, default: None
            Gradient of results in original parameter space (tensor)
        domain : int, optional, default: None
            Determine error in specified domain only. Default: None (all domains)
        output_idx : int or list of int
            Index of the QOI the provided coeffs and results are referring to. The correct QOI will be
            selected from the validation set in case of nrmsd error.

        Returns
        -------
        error: float
            Estimated difference between gPC approximation and original model
        """
        if output_idx is None:
            output_idx = np.arange(results.shape[1])
        if type(output_idx) is list:
            output_idx = np.array(output_idx)
        if type(output_idx) is not np.ndarray:
            output_idx = np.array([output_idx])

        if domain is None:
            domain_idx = np.arange(len(coeffs))
        else:
            domain_idx = domain

        # Determine QOIs with NaN in results and exclude them from validation
        non_nan_mask = np.where(np.all(~np.isnan(results), axis=0))[0]
        n_nan = results.shape[1] - non_nan_mask.size

        if n_nan > 0:
            iprint("In {}/{} output quantities NaN's were found.".format(n_nan, results.shape[1]),
                   tab=0, verbose=self.options["verbose"])

        results = results[:, non_nan_mask]

        # always determine nrmsd if a validation set is present
        if isinstance(self.validation, ValidationSet):

            if domain is None:
                mask_domain = np.ones(self.validation.grid.coords_norm.shape[0]).astype(bool)
                gpc_results = self.get_approximation(coeffs, self.validation.grid.coords_norm, output_idx=None)
            else:
                mask_domain = self.classifier.predict(self.validation.grid.coords_norm) == domain
                coords_domain = self.validation.grid.coords_norm[mask_domain, ]
                gpc_results = self.gpc[domain].get_approximation(coeffs[domain],
                                                                 coords_domain,
                                                                 output_idx=None)

            if gpc_results.ndim == 1:
                gpc_results = gpc_results[:, np.newaxis]

            if len(output_idx) == 1:
                validation_results_passed = self.validation.results[np.argwhere(mask_domain), output_idx][:, np.newaxis]
            else:
                validation_results_passed = self.validation.results[np.argwhere(mask_domain), output_idx]

            error_nrmsd = float(np.mean(nrmsd(gpc_results,
                                              validation_results_passed,
                                              error_norm=self.options["error_norm"],
                                              x_axis=False)))

            if domain is None:
                self.relative_error_nrmsd.append(error_nrmsd)

        if self.options["error_type"] == "nrmsd":
            if domain is None:
                self.error.append(self.relative_error_nrmsd[-1])

        elif self.options["error_type"] == "loocv":
            error_loocv = self.loocv(results=results,
                                     error_norm=self.options["error_norm"],
                                     domain=domain)

            if domain is None:
                self.relative_error_loocv.append(error_loocv)
                self.error.append(self.relative_error_loocv[-1])

        if domain is None:
            return self.error[-1]
        else:
            if self.options["error_type"] == "nrmsd":
                return error_nrmsd
            elif self.options["error_type"] == "loocv":
                return error_loocv

    def get_pdf(self, coeffs, n_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = MEGPC.get_pdf(coeffs, n_samples, output_idx=None)

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_coeffs x n_out]
            GPC coefficients
        n_samples: int
            Number of samples used to estimate output pdfs
        output_idx: ndarray, optional, default=None [1 x n_out]
            Index of output quantities to consider (if output_idx=None, all output quantities are considered)

        Returns
        -------
        pdf_x: ndarray of float [100 x n_out]
            x-coordinates of output pdfs of output quantities
        pdf_y: ndarray of float [100 x n_out]
            y-coordinates of output pdfs (probability density of output quantity)
        """

        # handle (N,) arrays
        if len(coeffs[0].shape) == 1:
            n_out = 1
        else:
            n_out = coeffs[0].shape[1]

        # if output index array is not provided, determine pdfs of all outputs
        if output_idx is None:
            output_idx = np.linspace(0, n_out - 1, n_out)
            output_idx = output_idx[np.newaxis, :]

        # sample gPC expansion
        samples_in, samples_out = self.get_samples(n_samples=n_samples, coeffs=coeffs, output_idx=output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, n_out])
        pdf_y = np.zeros([100, n_out])

        for i_out in range(n_out):
            pdf_y[:, i_out], tmp = np.histogram(samples_out, bins=100, density=True)
            pdf_x[:, i_out] = (tmp[1:] + tmp[0:-1])/2.

            # kde = scipy.stats.gaussian_kde(samples_out[:, i_out], bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            # pdf_y[:, i_out] = kde(pdf_x[:, i_out])
            # pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)

        return pdf_x, pdf_y

    def get_samples(self, coeffs, n_samples, output_idx=None):
        """
        Randomly sample gPC expansion.

        x, pce = SGPC.get_pdf_mc(n_samples, coeffs, output_idx=None)

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients for each sub-domain
        n_samples: int
            Number of random samples drawn from the respective input pdfs.
        output_idx: ndarray of int [1 x n_out] optional, default=None
            Index of output quantities to consider.

        Returns
        -------
        x: ndarray of float [n_samples x dim]
            Generated samples in normalized coordinates [-1, 1]. (original parameter space)
        pce: ndarray of float [n_samples x n_out]
            GPC approximation at points x.
        """

        # seed the random numbers generator
        np.random.seed()

        # generate temporary grid with random samples for each random input variable [n_samples x dim]
        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": n_samples, "seed": None})

        # if output index list is not provided, sample all gpc outputs
        if output_idx is None:
            n_out = 1 if coeffs[0].ndim == 1 else coeffs[0].shape[1]
            output_idx = np.arange(n_out)

        pce = self.get_approximation(coeffs=coeffs, x=grid.coords_norm, output_idx=output_idx)

        return grid.coords_norm, pce

    def get_approximation(self, coeffs, x, output_idx=None):
        """
        Calculates the gPC approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]).

        pce = MEGPC.get_approximation(coeffs, x, output_idx=None)

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients for each output variable of each sub-domain
        x: ndarray of float [n_x x n_dim]
            Normalized coordinates, where the gPC approximation is calculated (original parameter space)
        output_idx: ndarray of int, optional, default=None [n_out]
            Indices of output quantities to consider (Default: all).

        Returns
        -------
        pce: ndarray of float [n_x x n_out]
            GPC approximation at normalized coordinates x.
        """
        if type(output_idx) != np.ndarray and output_idx is not None:
            output_idx = np.array([output_idx])
        else:
            if type(coeffs) is list:
                output_idx = np.arange(coeffs[0].shape[1])
            else:
                output_idx = np.arange(coeffs.shape[1])

        pce = np.zeros((x.shape[0], len(output_idx)))

        # get classes of grid-points
        domains = self.classifier.predict(x)

        # determine gPC approximation for sub-domains
        for d in np.unique(domains):
            pce[domains == d, :] = self.gpc[d].get_approximation(coeffs=coeffs[d],
                                                                 x=x[domains == d, :],
                                                                 output_idx=output_idx)

        return pce

    def update_gpc_matrices(self, gradient=False):
        """
        Update gPC matrix according to existing self.grid and self.basis.

        Call this method when self.gpc_matrix does not fit to self.grid and self.basis objects anymore
        The old gPC matrix with their self.gpc_matrix_b_id and self.gpc_matrix_coords_id is compared
        to self.basis.b_id and self.grid.coords_id. New rows and columns are computed when differences are found.
        """
        for i, gpc in enumerate(self.gpc):
            gpc.update_gpc_matrix(gradient=gradient)

    def save_gpc_matrices_hdf5(self):
        """
        Save gPC matrix and gPC gradient matrix in .hdf5 file <"fn_results" + ".hdf5"> under the key "gpc_matrix/dom_x"
        and "gpc_matrix_gradient/dom_x". If matrices are already present, check for equality and save only appended
        rows and columns.
        """
        for i, gpc in enumerate(self.gpc):
            gpc.save_gpc_matrix_hdf5(hdf5_path_gpc_matrix="gpc_matrix/dom_" + str(i),
                                     hdf5_path_gpc_matrix_gradient="gpc_matrix_gradient/dom_" + str(i))

    def solve(self, results, gradient_results=None, solver=None, settings=None, verbose=False):
        """
        Determines gPC coefficients of sub-gPCs

        Parameters
        ----------
        results : ndarray of float [n_grid x n_out]
            Results from simulations with n_out output quantities
        gradient_results : ndarray of float [n_grid x dim x n_out x dim], optional, default: None
            Gradient of results in original parameter space (tensor)
        solver : str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
            - 'LarsLasso' ... Least-Angle Regression using Lasso model (SGPC.Reg, EGPC)
            - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
        settings : dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0 or "sparsity": float 0...1
            - 'LarsLasso' ... {"alpha": float 0...1} Regularization parameter
            - 'NumInt' ... None
        verbose : bool
            boolean value to determine if to print out the progress into the standard output

        Returns
        -------
        coeffs: list of ndarray of float [n_gpc][n_coeffs x n_out]
            gPC coefficients
        """
        # use default solver if not specified
        if solver is None:
            solver = self.solver

        # use default solver settings if not specified
        if solver is None:
            settings = self.settings

        coeffs = [0 for _ in range(self.n_gpc)]

        # determine coeffs of sub-gPCs
        for d in np.unique(self.domains):
            if gradient_results is not None:
                gradient_results_passed = gradient_results[self.domains == d, :, :]
            else:
                gradient_results_passed = None

            coeffs[d] = self.gpc[d].solve(results=results[self.domains == d, :],
                                          gradient_results=gradient_results_passed,
                                          solver=solver,
                                          settings=settings,
                                          verbose=verbose)

        return coeffs

    # def extract_domain(self, data, domain):
    #     """
    #     Extract data from dataset of specified domain
    #
    #     Parameters
    #     ----------
    #     data : ndarray of float [n_data x m]
    #         Dataset
    #     domain : int
    #         Domain index to extract
    #     """
    #     mask_results = self.domains == domain
    #
    #     # determine mask
    #     if self.gpc[domain].gpc_matrix_gradient is not None:
    #         mask_gradient = np.zeros((self.grid.coords_norm.shape[0], 1, self.problem.dim)).astype(bool)
    #         mask_gradient[mask_results, :, :] = True
    #         mask = np.vstack((mask_results[:, np.newaxis], ten2mat(mask_gradient)))
    #
    #     else:
    #         mask = np.zeros((data.shape[0], 1)).astype(bool)
    #         mask[mask_results, :] = True
    #
    #     return data[mask.flatten(), :]

    def create_validation_set(self, n_samples, n_cpu=1, gradient=False):
        """
        Creates a ValidationSet instance (calls the model)

        Parameters
        ----------
        n_samples : int
            Number of sampling points contained in the validation set
        n_cpu : int
            Number of parallel function evaluations to evaluate validation set (n_cpu=0 assumes that the
            model is capable to evaluate all grid points in parallel)
        gradient : bool, optional, default: False
            Determine gradient of results in each grid points
        """
        # create set of validation points
        n_samples = n_samples

        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": n_samples, "seed": None})

        # Evaluate original model at grid points
        com = Computation(n_cpu=n_cpu, matlab_model=self.matlab_model)
        results = com.run(model=self.problem.model, problem=self.problem, coords=grid.coords)

        if results.ndim == 1:
            results = results[:, np.newaxis]

        # Determine gradient of results at grid points
        if gradient:
            gradient_results = self.get_gradient(grid=grid, results=results)
        else:
            gradient_results = None

        self.validation = ValidationSet(grid=grid, results=results, gradient_results=gradient_results)

    @staticmethod
    def get_mean(samples):
        """
        Calculate the expected value.

        mean = MEGPC.get_mean(samples)

        Parameters
        ----------
        samples : ndarray of float [n_x x n_out], optional, default: None
            Model evaluations from MEGPC approximation

        Returns
        -------
        mean: ndarray of float [1 x n_out]
            Expected value of output quantities
        """
        mean = np.mean(samples, axis=0)
        mean = mean[np.newaxis, :]

        return mean

    @staticmethod
    def get_std(samples=None):
        """
        Calculate the standard deviation.

        std = MEGPC.get_std(samples)

        Parameters
        ----------
        samples : ndarray of float [n_samples x n_out], optional, default: None
            Model evaluations from MEGPC approximation

        Returns
        -------
        std: ndarray of float [1 x n_out]
            Standard deviation of output quantities
        """
        std = np.std(samples, axis=0)
        std = std[np.newaxis, :]

        return std

    # noinspection PyTypeChecker
    def get_sobol_indices(self, coeffs, algorithm="sampling", n_samples=1e4):
        """
        Calculate the available sobol indices from the gPC coefficients by sampling up to second order.

        sobol, sobol_idx, sobol_idx_bool = MEGPC.get_sobol_indices(coeffs, n_samples=1e4)

        Parameters
        ----------
        coeffs:  list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients
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
        dim = self.problem.dim

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

        return sobol, sobol_idx, sobol_idx_bool

    def get_sobol_composition(self, sobol, sobol_idx_bool):
        """
        Determine average ratios of Sobol indices over all output quantities:
        (i) over all orders and (e.g. 1st: 90%, 2nd: 8%, 3rd: 2%)
        (ii) for the 1st order indices w.r.t. each random variable. (1st: x1: 50%, x2: 40%)

        sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std, sobol_rel_1st_order_mean, sobol_rel_1st_order_std
        = MEGPC.get_sobol_composition(coeffs, sobol, sobol_idx, sobol_idx_bool)

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

        sobol_1st, sobol_idx_1st = MEGPC.get_extracted_sobol_order(sobol, sobol_idx, order=1)

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
    def get_global_sens(self, coeffs, algorithm="sampling", n_samples=1e5):
        """
        Determine the global derivative based sensitivity coefficients after Xiu (2009) [1].

        global_sens = MEGPC.get_global_sens(coeffs, n_samples=1e5)

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out], optional, default: None
            GPC coefficients
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

        # generate sample coordinates (original parameter space)
        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": n_samples, "seed": None})

        local_sens = self.get_local_sens(coeffs, grid.coords_norm)

        # average the results and reshape [dim x n_out]
        global_sens = np.mean(local_sens, axis=0).transpose()

        return global_sens

    # noinspection PyTypeChecker
    def get_local_sens(self, coeffs, x=None):
        """
        Determine the local derivative based sensitivity coefficients in the point of interest x
        (normalized coordinates [-1, 1]).

        local_sens = MEGPC.calc_localsens(coeffs, x)

        Parameters
        ----------
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients
        x: ndarray of float [n_points x dim], optional, default: center of parameter space
            Points in variable space to evaluate local sensitivity in (normalized coordinates [-1, 1])
            (original parameter space)

        Returns
        -------
        local_sens: ndarray [n_points x n_out x dim]
            Local sensitivity of output quantities in point x
        """

        if x is None:
            x = np.zeros(self.problem.dim)[np.newaxis, :]

        # classify coordinates
        domains = self.classifier.predict(x)

        local_sens = np.zeros((x.shape[0], coeffs[0].shape[1], self.problem.dim))

        for d in np.unique(domains):
            # project coordinate to reduced parameter space if necessary
            if self.gpc[d].p_matrix is not None:
                x_passed = np.dot(x[domains == d, :], self.gpc[d].p_matrix.transpose() /
                                  self.gpc[d].p_matrix_norm[np.newaxis, :])
            else:
                x_passed = x[domains == d, :]

            # construct gPC gradient matrix [n_samples x n_basis x dim(_red)]
            gpc_matrix_gradient = self.gpc[d].calc_gpc_matrix(b=self.gpc[d].basis.b,
                                                              x=x_passed,
                                                              gradient=True)

            local_sens_domain = np.dot(gpc_matrix_gradient.transpose(2, 0, 1), coeffs[d]).transpose(1, 2, 0)

            # project the gradient back to the original space if necessary
            if self.gpc[d].p_matrix is not None:
                local_sens_domain = np.dot(local_sens_domain, self.gpc[d].p_matrix /
                                           self.gpc[d].p_matrix_norm[:, np.newaxis])

            local_sens[domains == d, :, :] = local_sens_domain

        return local_sens

