import numpy as np
import scipy.stats
import copy
import h5py
import time
import random
import sys
from sklearn import linear_model
from scipy.signal import savgol_filter
from .misc import get_cartesian_product
from .misc import display_fancy_bar
from .misc import nrmsd
from .misc import mat2ten
from .misc import ten2mat
from .ValidationSet import *
from .Computation import *
from .Grid import *


class GPC(object):
    """
    General gPC base class

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

    Attributes
    ----------
    problem: Problem class instance
        GPC Problem under investigation
    basis: Basis class instance
        Basis of the gPC including BasisFunctions
    grid: Grid class instance
        Grid of the derived gPC approximation
    validation: ValidationSet object (optional)
        Object containing a set of validation points and corresponding solutions. Can be used
        to validate gpc approximation setting options["error_type"]="nrmsd".
        - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
        - results: ndarray [n_grid x n_out] results
    gpc_matrix: [N_samples x N_poly] ndarray of float
        Generalized polynomial chaos matrix
    gpc_matrix_gradient: [N_samples * dim x N_poly] ndarray of float
        Derivative of generalized polynomial chaos matrix
    matrix_inv: [N_poly (+ N_gradient) x N_samples] ndarray of float
        Pseudo inverse of the generalized polynomial chaos matrix (with or without gradient)
    p_matrix: [dim_red x dim] ndarray of float
        Projection matrix to reduce number of efficient dimensions (\\eta = p_matrix * \\xi)
    p_matrix_norm: [dim_red] ndarray of float
        Maximal possible length of new axis in \\eta space. Since the projected variables are modelled in
        the normalized space between [-1, 1], the transformed coordinates need to be scaled.
    nan_elm: ndarray of int
        Indices of NaN elements of model output
    gpc_matrix_coords_id: list of UUID4()
        UUID4() IDs of grid points the gPC matrix derived with
    gpc_matrix_b_id: list of UUID4()
        UUID4() IDs of basis functions the gPC matrix derived with
    n_basis: int or list of int
        Number of basis functions (for iterative solvers, this is a list of its history)
    n_grid: int or list of int
        Number of grid points (for iterative solvers, this is a list of its history)
    solver: str
        Default solver to determine the gPC coefficients (can be chosen during GPC.solve)
        - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
        - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        - 'LarsLasso' ... {"alpha": float 0...1} Regularization parameter
        - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
    verbose: bool
        Boolean value to determine if to print out the progress into the standard output
    fn_results : string, optional, default=None
        If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
    relative_error_loocv: list of float
        Relative error of the leave-one-out-cross-validation
    relative_error_nrmsd: list of float
        Normalized root mean square deviation between model and gpc approximation
    options : dict
        Options of gPC algorithm
    """

    def __init__(self, problem, options, validation=None):
        """
        Constructor; Initializes GPC class
        """
        # objects
        self.problem = problem
        self.problem_original = None
        self.basis = None
        self.grid = None
        self.validation = validation

        # arrays
        self.gpc_matrix = None
        self.gpc_matrix_gradient = None
        self.matrix_inv = None
        self.p_matrix = None
        self.p_matrix_norm = None
        self.nan_elm = []
        self.gpc_matrix_coords_id = None
        self.gpc_matrix_b_id = None
        self.gpc_matrix_gradient_coords_id = None
        self.gpc_matrix_gradient_b_id = None
        self.n_basis = []
        self.n_grid = []
        self.relative_error_nrmsd = []
        self.relative_error_loocv = []
        self.relative_error_kcv = []
        self.error = []
        self.n_out = []
        self.gradient_idx = None

        # options
        if options is not None:
            if "gradient_enhanced" not in options.keys():
                options["gradient_enhanced"] = False

            if "fn_results" not in options.keys():
                options["fn_results"] = None

            if "matlab_model" not in options.keys():
                options["matlab_model"] = False

            if "backend" not in options.keys():
                options["backend"] = "omp"

            self.gradient = options["gradient_enhanced"]
            self.fn_results = options["fn_results"]
            self.matlab_model = options["matlab_model"]
            self.backend = options["backend"]

        else:
            self.gradient = None
            self.fn_results = None
            self.matlab_model = False
            self.backend = "omp"

        self.solver = None
        self.settings = None
        self.verbose = True

        self.options = options

    def init_gpc_matrix(self, gradient_idx=None):
        """
        Sets self.gpc_matrix and self.gpc_matrix_gradient with given self.basis and self.grid

        Parameters
        ----------
        gradient_idx : ndarray of int [gradient_results.shape[0]]
            Indices of grid points where the gradient in gradient_results is provided
        """

        if self.gradient_idx is None or gradient_idx is not None:
            self.gradient_idx = gradient_idx

        self.gpc_matrix = self.create_gpc_matrix(b=self.basis.b,
                                                 x=self.grid.coords_norm,
                                                 gradient=False)
        self.n_grid.append(self.gpc_matrix.shape[0])
        self.n_basis.append(self.gpc_matrix.shape[1])
        self.gpc_matrix_coords_id = copy.deepcopy(self.grid.coords_id)
        self.gpc_matrix_b_id = copy.deepcopy(self.basis.b_id)

        if self.gradient and self.gradient_idx is not None:
            self.gpc_matrix_gradient = self.create_gpc_matrix(b=self.basis.b,
                                                              x=self.grid.coords_norm,
                                                              gradient=True)
            self.gpc_matrix_gradient = ten2mat(self.gpc_matrix_gradient)
            self.gpc_matrix_gradient_coords_id = copy.deepcopy(self.grid.coords_id)
            self.gpc_matrix_gradient_b_id = copy.deepcopy(self.basis.b_id)

    def create_gpc_matrix(self, b, x, gradient=False, gradient_idx=None, weighted=False, verbose=False):
        """
        Construct the gPC matrix or its derivative.

        Parameters
        ----------
        b : list of BasisFunction object instances [n_basis][n_dim]
            Parameter wise basis function objects used in gPC (Basis.b)
            Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.
        x : ndarray of float [n_x x n_dim]
            Coordinates of x = (x1, x2, ..., x_dim) where the rows of the gPC matrix are evaluated (normalized [-1, 1])
        gradient : bool, optional, default: False
            Determine gradient gPC matrix.
        gradient_idx : ndarray of int [gradient_results.shape[0]]
            Indices of grid points where the gradient in gradient_results is provided
        weighted : bool, optional, default: False
            Weight gPC matrix with (row 2-norm)^-1
        verbose : bool, optional, default: False
            boolean value to determine if to print out the progress into the standard output

        Returns
        -------
        gpc_matrix: ndarray of float [n_x x n_basis (x dim)]
            GPC matrix where the columns correspond to the basis functions and the rows the to the sample coordinates.
            If gradient_idx!=None, the gradient is returned at the specified sample coordinates point by point.
        """
        # overwrite self.gradient_idx if it is provided
        if gradient_idx is not None:
            self.gradient_idx = gradient_idx

        iprint('Constructing gPC matrix...', verbose=verbose, tab=0)

        if self.backend not in ["cpu", "omp", "cuda", "python"]:
            raise NotImplementedError(f"Backend {self.backend} is not implemented")

        if self.backend == "cpu":
            # CPU backend (CPU single core)
            try:
                from .pygpc_extensions import create_gpc_matrix_cpu
                if not gradient:
                    # the third dimension is important and should not be removed
                    # otherwise the code could produce undefined behaviour
                    gpc_matrix = np.empty([x.shape[0], len(b), 1])
                    create_gpc_matrix_cpu(x, self.basis.b_array, gpc_matrix)
                    gpc_matrix = np.squeeze(gpc_matrix)
                else:
                    gpc_matrix = np.empty([len(self.gradient_idx), len(b), self.problem.dim])
                    create_gpc_matrix_cpu(x[self.gradient_idx, :], self.basis.b_array_grad, gpc_matrix)
            except (ImportError):
                print("The CPU-extension is not installed. Fall back to pure Python as backend.")
                self.backend = "python"
        elif self.backend == "omp":
            # OpenMP backend (CPU multi core)
            try:
                from .pygpc_extensions import create_gpc_matrix_omp
                if not gradient:
                    # the third dimension is important and should not be removed
                    # otherwise the code could produce undefined behaviour
                    gpc_matrix = np.empty([x.shape[0], len(b), 1])
                    create_gpc_matrix_omp(x, self.basis.b_array, gpc_matrix)
                    gpc_matrix = np.squeeze(gpc_matrix)
                else:
                    gpc_matrix = np.empty([len(self.gradient_idx), len(b), self.problem.dim])
                    create_gpc_matrix_omp(x[self.gradient_idx, :], self.basis.b_array_grad, gpc_matrix)
            except (ImportError):
                print("The OMP-extension is not installed. Fall back to pure Python as backend.")
                self.backend = "python"
        elif self.backend == "cuda":
            # CUDA backend (GPU multi core)
            try:
                from .pygpc_extensions_cuda import create_gpc_matrix_cuda
                if not gradient:
                    # the third dimension is important and should not be removed
                    # otherwise the code could produce undefined behaviour
                    gpc_matrix = np.empty([x.shape[0], len(b), 1])
                    create_gpc_matrix_cuda(x, self.basis.b_array, gpc_matrix)
                    gpc_matrix = np.squeeze(gpc_matrix)
                else:
                    gpc_matrix = np.empty([len(self.gradient_idx), len(b), self.problem.dim])
                    create_gpc_matrix_cuda(x[self.gradient_idx, :], self.basis.b_array_grad, gpc_matrix)
            except (ImportError):
                print("The CUDA-extension is not installed. Fall back to pure Python as backend.")
                self.backend = "python"
                
        if self.backend == "python":
            # Python backend
            if not gradient:
                gpc_matrix = np.ones([x.shape[0], len(b)])
                for i_basis in range(len(b)):
                    for i_dim in range(self.problem.dim):
                        gpc_matrix[:, i_basis] *= b[i_basis][i_dim](x[:, i_dim])
            else:
                gpc_matrix = np.ones([len(self.gradient_idx), len(b), self.problem.dim])
                for i_dim_gradient in range(self.problem.dim):
                    for i_basis in range(len(b)):
                        for i_dim in range(self.problem.dim):
                            if i_dim == i_dim_gradient:
                                derivative = True
                            else:
                                derivative = False
                            gpc_matrix[:, i_basis, i_dim_gradient] *= b[i_basis][i_dim](x[self.gradient_idx, i_dim],
                                                                                        derivative=derivative)

        if gpc_matrix.ndim == 1 and x.shape[0] == 1:
            gpc_matrix = gpc_matrix[np.newaxis, :]
        elif gpc_matrix.ndim == 1 and self.basis.n_basis == 1:
            gpc_matrix = gpc_matrix[:, np.newaxis]

        if weighted:
            w = np.diag(1/np.linalg.norm(gpc_matrix, axis=1))
            gpc_matrix = np.matmul(w, gpc_matrix)

        return gpc_matrix

    def get_kcv(self, results,  error_norm="relative",k=10):
        """
        Perfrom k-fold cross validation of gPC approximation and add error value to self.relative_error_kcv.
        Parameters
        ----------
        results: ndarray of float [n_grid x n_out]
            Results from n_grid simulations with n_out output quantities
        error_norm: str, optional, default="relative"
            Decide if error is determined "relative" or "absolute"
        k: int, optional, default=10
            Number of folds for k-fold cross-validation
        Returns:
        -------
        relative_error_kcv: float
            Relative mean error of k-fold cross validation
        """
        # prepare necessary data
        # if results.ndim == 1:
        #     results = results[:, None]
        matrix = self.gpc_matrix
        results_complete = results
        # define number of performed cross validations
        n_simulations = results_complete.shape[0]
        if n_simulations <= k:
            k = n_simulations

        start = time.time()
        relative_error = np.zeros(k)
        # random assign results to k groups for cross-validation
        shuffle = np.arange(n_simulations, dtype=int)
        np.random.shuffle(shuffle)
        groups = np.zeros((k, n_simulations), dtype=int)
        group_sizes = float(n_simulations) / k
        for i in range(k):
            f = int(i * group_sizes)
            t = int((i + 1) * group_sizes)
            if i == k - 1:
                t = n_simulations
            groups[i, shuffle[f:t]] = True
        eps = 0.0
        i = 1
        # error estimation
        for g in groups:
            # determine regression coefficients
            coeffs_g = self.solve(results=results_complete[~g,:],
                                solver=self.options["solver"],
                                matrix=matrix[~g, :],
                                settings=self.options["settings"],
                                verbose=False)
            sim_results_temp = results_complete[~g,:]
            if error_norm == "relative":
                norm = scipy.linalg.norm(sim_results_temp)
            else:
                norm = 1.

            eps += scipy.linalg.norm(sim_results_temp - np.matmul(matrix[~g,:], coeffs_g)) / norm
            display_fancy_bar("Cross Validation", int(i), int(k))
            i += 1
        eps /= k
        iprint("Cross-Validation time: {} sec". format(time.time() - start), tab=0, verbose=True)

        return eps    
    
    def get_loocv(self, coeffs, results, gradient_results=None, error_norm="relative"):
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
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        results: ndarray of float [n_grid x n_out]
            Results from n_grid simulations with n_out output quantities
        error_norm: str, optional, default="relative"
            Decide if error is determined "relative" or "absolute"
        gradient_results : ndarray of float [n_grid x n_out x dim], optional, default: None
            Gradient of results in original parameter space (tensor)

        Returns
        -------
        relative_error_loocv: float
            Relative mean error of leave one out cross validation

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2010). An adaptive algorithm to build up sparse polynomial chaos expansions
           for stochastic finite element analysis. Probabilistic Engineering Mechanics, 25(2), 183-197.
        """
        # if self.options["gradient_enhanced"]:
        #     matrix = np.vstack((self.gpc_matrix, self.gpc_matrix_gradient))
        #
        #     # transform gradient of results in case of projection
        #     if self.p_matrix is not None:
        #         gradient_results = np.matmul(gradient_results,
        #                                   self.p_matrix.transpose() * self.p_matrix_norm[np.newaxis, :])
        #
        #     results_complete = np.vstack((results, ten2mat(gradient_results)))
        # else:
        #     matrix = self.gpc_matrix
        #     results_complete = results

        # matrix = self.gpc_matrix
        # results_complete = results

        # Analytical error estimation in case of overdetermined systems
        # if matrix.shape[0] > 2*matrix.shape[1]:
            # determine Psi (Psi^T Psi)^-1 Psi^T
            # h = np.matmul(np.matmul(matrix, np.linalg.inv(np.matmul(matrix.transpose(), matrix))), matrix.transpose())
            #
            # # determine loocv error
            # err = np.mean(((results_complete - np.matmul(matrix, coeffs)) /
            #                (1 - np.diag(h))[:, np.newaxis]) ** 2, axis=0)
            #
            # if error_norm == "relative":
            #     norm = np.var(results_complete, axis=0, ddof=1)
            # else:
            #     norm = 1.
            #
            # # normalize
            # relative_error_loocv = np.mean(err / norm)

        # else:
        # perform manual loocv without gradient
        matrix = self.gpc_matrix
        results_complete = results

        n_loocv = 25

        # define number of performed cross validations (max 100)
        n_loocv_points = np.min((results_complete.shape[0], n_loocv))

        # make list of indices, which are randomly sampled
        loocv_point_idx = random.sample(list(range(results_complete.shape[0])), n_loocv_points)

        start = time.time()
        relative_error = np.zeros(n_loocv_points)
        for i in range(n_loocv_points):
            # get mask of eliminated row
            mask = np.arange(results_complete.shape[0]) != loocv_point_idx[i]

            # determine gpc coefficients (this takes a lot of time for large problems)
            coeffs_loo = self.solve(results=results_complete[mask, :],
                                    solver=self.options["solver"],
                                    matrix=matrix[mask, :],
                                    settings=self.options["settings"],
                                    verbose=False)

            sim_results_temp = results_complete[loocv_point_idx[i], :]

            if error_norm == "relative":
                norm = scipy.linalg.norm(sim_results_temp)
            else:
                norm = 1.

            relative_error[i] = scipy.linalg.norm(sim_results_temp - np.matmul(matrix[loocv_point_idx[i], :],
                                                                            coeffs_loo))\
                                / norm
            display_fancy_bar("LOOCV", int(i + 1), int(n_loocv_points))

        # store result in relative_error_loocv
        relative_error_loocv = np.mean(relative_error)
        iprint("LOOCV computation time: {} sec".format(time.time() - start), tab=0, verbose=True)

        return relative_error_loocv

    def validate(self, coeffs, results=None, gradient_results=None, qoi_idx=None):
        """
        Validate gPC approximation using the ValidationSet object contained in the Problem object.
        Determines the normalized root mean square deviation between the gpc approximation and the
        original model. Skips this step if no validation set is present

        Parameters
        ----------
        coeffs: ndarray of float [n_coeffs x n_out]
            GPC coefficients
        results: ndarray of float [n_grid x n_out]
            Results from n_grid simulations with n_out output quantities
        gradient_results : ndarray of float [n_grid x n_out x dim], optional, default: None
            Gradient of results in original parameter space (tensor)
        qoi_idx : int, optional, default: None
            Index of QOI to validate (if None, all QOI are considered)

        Returns
        -------
        error: float
            Estimated difference between gPC approximation and original model
        """
        if qoi_idx is None:
            qoi_idx = np.arange(0, results.shape[1])

        # Determine QOIs with NaN in results and exclude them from validation
        non_nan_mask = np.where(np.all(~np.isnan(results), axis=0))[0]
        n_nan = results.shape[1] - non_nan_mask.size

        if n_nan > 0:
            iprint("In {}/{} output quantities NaN's were found.".format(n_nan, results.shape[1]),
                   tab=0, verbose=self.options["verbose"])

        results = results[:, non_nan_mask]

        if gradient_results is not None:
            gradient_results = gradient_results[:, non_nan_mask, :]

        # always determine nrmsd if a validation set is present
        if isinstance(self.validation, ValidationSet):

            gpc_results = self.get_approximation(coeffs, self.validation.grid.coords_norm, output_idx=None)

            if gpc_results.ndim == 1:
                gpc_results = gpc_results[:, np.newaxis]

            if self.validation.results[:, qoi_idx].ndim == 1:
                validation_results = self.validation.results[:, qoi_idx][:, np.newaxis]
            else:
                validation_results = self.validation.results[:, qoi_idx]

            self.relative_error_nrmsd.append(float(np.mean(nrmsd(gpc_results,
                                                                 validation_results,
                                                                 error_norm=self.options["error_norm"],
                                                                 x_axis=False))))

        if self.options["error_type"] == "nrmsd":
            self.error.append(self.relative_error_nrmsd[-1])

        elif self.options["error_type"] == "loocv":
            self.relative_error_loocv.append(self.get_loocv(coeffs=coeffs,
                                             results=results,
                                             gradient_results=gradient_results,
                                             error_norm=self.options["error_norm"]))
            self.error.append(self.relative_error_loocv[-1])

        elif self.options["error_type"] is None:
            self.error.append(None)

        return self.error[-1]

    def get_pdf(self, coeffs, n_samples, output_idx=None, filter=True, return_samples=False):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = SGPC.get_pdf(coeffs, n_samples, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_coeffs x n_out]
            GPC coefficients
        n_samples: int
            Number of samples used to estimate output pdfs
        output_idx: ndarray, optional, default=None [1 x n_out]
            Index of output quantities to consider (if output_idx=None, all output quantities are considered)
        filter : bool, optional, default: True
            Use savgol_filter to smooth probability density
        return_samples : bool, optional, default: False
            Additionally returns in and output samples with which the pdfs were computed

        Returns
        -------
        pdf_x: ndarray of float [100 x n_out]
            x-coordinates of output pdfs of output quantities
        pdf_y: ndarray of float [100 x n_out]
            y-coordinates of output pdfs (probability density of output quantity)
        samples_in : ndarray of float [n_samples x dim] (optional)
            Input samples (if return_samples=True)
        samples_out : ndarray of float [n_samples x n_out] (optional)
            Output samples (if return_samples=True)
        """

        # handle (N,) arrays
        if len(coeffs.shape) == 1:
            coeffs = coeffs[:, np.newaxis]

        # if output index array is not provided, determine pdfs of all outputs
        if output_idx is None:
            output_idx = np.arange(0, coeffs.shape[1])
            output_idx = output_idx[np.newaxis, :]

        n_out = len(output_idx)

        # sample gPC expansion
        samples_in, samples_out = self.get_samples(n_samples=n_samples, coeffs=coeffs, output_idx=output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, n_out])
        pdf_y = np.zeros([100, n_out])

        for i_out in range(n_out):

            pdf_y[:, i_out], tmp = np.histogram(samples_out[:, i_out], bins=100, density=True)
            pdf_x[:, i_out] = (tmp[1:] + tmp[0:-1]) / 2.

            if filter:
                pdf_y[:, i_out] = savgol_filter(pdf_y[:, i_out], 51, 5)

            # kde = scipy.stats.gaussian_kde(samples_out[:, i_out], bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            # pdf_y[:, i_out] = kde(pdf_x[:, i_out])
            # pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)

        if return_samples:
            return pdf_x, pdf_y, samples_in, samples_out
        else:
            return pdf_x, pdf_y

    def get_samples(self, coeffs, n_samples, output_idx=None):
        """
        Randomly sample gPC expansion.

        x, pce = SGPC.get_pdf_mc(n_samples, coeffs, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        n_samples: int
            Number of random samples drawn from the respective input pdfs.
        output_idx: ndarray of int [1 x n_out] optional, default=None
            Index of output quantities to consider.

        Returns
        -------
        x: ndarray of float [n_samples x dim]
            Generated samples in normalized coordinates [-1, 1].
        pce: ndarray of float [n_samples x n_out]
            GPC approximation at points x.
        """

        # seed the random numbers generator
        np.random.seed()

        if self.p_matrix is not None:
            problem = self.problem_original
        else:
            problem = self.problem

        # generate temporary grid with random samples for each random input variable [n_samples x dim]
        grid = Random(parameters_random=problem.parameters_random,
                      n_grid=n_samples)

        # if output index list is not provided, sample all gpc outputs
        if output_idx is None:
            n_out = 1 if coeffs.ndim == 1 else coeffs.shape[1]
            output_idx = np.arange(n_out)
            # output_idx = output_idx[np.newaxis, :]

        pce = self.get_approximation(coeffs=coeffs, x=grid.coords_norm, output_idx=output_idx)

        return grid.coords_norm, pce

    def get_approximation(self, coeffs, x, output_idx=None):
        """
        Calculates the gPC approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]).

        pce = GPC.get_approximation(coeffs, x, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients for each output variable
        x: ndarray of float [n_x x n_dim]
            Coordinates of x = (x1, x2, ..., x_dim) where the rows of the gPC matrix are evaluated (normalized [-1, 1]).
            The coordinates will be transformed in case of projected gPC.
        output_idx: ndarray of int, optional, default=None [n_out]
            Indices of output quantities to consider (Default: all).

        Returns
        -------
        pce: ndarray of float [n_x x n_out]
            GPC approximation at normalized coordinates x.
        """

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        # crop coordinates to gPC boundaries (values outside do not yield meaningful values)
        for i_dim, key in enumerate(list(self.problem.parameters_random.keys())):
            xmin = self.problem.parameters_random[key].pdf_limits_norm[0]
            xmax = self.problem.parameters_random[key].pdf_limits_norm[1]
            x[x[:, i_dim] < xmin, i_dim] = xmin
            x[x[:, i_dim] > xmax, i_dim] = xmax

        if output_idx is not None:
            # convert to 1d array
            output_idx = np.asarray(output_idx).flatten().astype(int)

            # crop coeffs array if output index is specified
            coeffs = coeffs[:, output_idx]

        if coeffs.ndim == 1:
            coeffs = coeffs[:, np.newaxis]

        # transform variables from xi to eta space if gpc model is reduced
        if self.p_matrix is not None:
            x = np.matmul(x, self.p_matrix.transpose() / self.p_matrix_norm[np.newaxis, :])

        if self.backend == "cuda":
            try:
                from .pygpc_extensions_cuda import get_approximation_cuda
                pce = np.empty([x.shape[0], coeffs.shape[1]])
                get_approximation_cuda(x, self.basis.b_array, coeffs, pce)
            except ImportError:
                print("The CUDA-extension is not installed. Fall back to pure Python as backend.")
                self.backend = "python"

        if self.backend == 'python' or self.backend == 'cpu' or self.backend == 'omp':
            # determine gPC matrix at coordinates x
            gpc_matrix = self.create_gpc_matrix(self.basis.b, x, gradient=False)

            # multiply with gPC coeffs
            pce = np.matmul(gpc_matrix, coeffs)

        else:
            raise NotImplementedError

        return pce

    def replace_gpc_matrix_samples(self, idx, seed=None):
        """
        Replace distinct sample points from the gPC matrix with new ones.

        GPC.replace_gpc_matrix_samples(idx, seed=None)

        Parameters
        ----------
        idx: ndarray of int [n_samples]
            Array of grid indices of grid.coords[idx, :] which are going to be replaced
            (rows of gPC matrix will be replaced by new ones)
        seed: float, optional, default=None
            Random seeding point
        """

        # Generate new grid points
        new_grid_points = Random(parameters_random=self.problem.parameters_random,
                                 n_grid=idx.size,
                                 seed=seed)

        # replace old grid points
        self.grid.coords[idx, :] = new_grid_points.coords
        self.grid.coords_norm[idx, :] = new_grid_points.coords_norm

        # replace old IDs of grid points with new ones
        for i in idx:
            self.grid.coords_id[i] = uuid.uuid4()
            self.gpc_matrix_coords_id[i] = copy.deepcopy(self.grid.coords_id[i])

        # determine new rows of gpc matrix and overwrite rows of gpc matrix
        self.gpc_matrix[idx, :] = self.create_gpc_matrix(b=self.basis.b,
                                                         x=new_grid_points.coords_norm,
                                                         gradient=False)

    def update_gpc_matrix(self):
        """
        Update gPC matrix and gPC matrix gradient according to existing self.grid and self.basis.

        Call this method when self.gpc_matrix does not fit to self.grid and self.basis objects anymore
        The old gPC matrix with their self.gpc_matrix_b_id and self.gpc_matrix_coords_id is compared
        to self.basis.b_id and self.grid.coords_id. New rows and columns are computed if differences are found.
        """
        self._update_gpc_matrix(gradient=False)

        if self.gradient:
            self._update_gpc_matrix(gradient=True)

    def _update_gpc_matrix(self, gradient=False):
        """
        Update gPC matrix or gPC gradient matrix
        """
        if self.backend == "python":
            # initialize updated matrix and variables
            if gradient:
                # reshape gpc gradient matrix from 2D to 3D representation [n_grid x n_basis x n_dim]
                matrix = mat2ten(mat=self.gpc_matrix_gradient, incr=self.problem.dim)
                matrix_updated = np.zeros((len(self.gradient_idx), len(self.basis.b_id), self.problem.dim))
                coords_id = self.gpc_matrix_gradient_coords_id[self.gradient_idx]
                coords_id_ref = self.grid.coords_gradient_id[self.gradient_idx]
                b_id = self.gpc_matrix_gradient_b_id
                b_id_ref = self.basis.b_id
                coords_norm = self.grid.coords_norm[self.gradient_idx]
                ge_str = "(gradient)"

            else:
                matrix = self.gpc_matrix
                matrix_updated = np.zeros((len(self.grid.coords_id), len(self.basis.b_id)))
                coords_id = self.gpc_matrix_coords_id
                coords_id_ref = self.grid.coords_id
                b_id = self.gpc_matrix_b_id
                b_id_ref = self.basis.b_id
                coords_norm = self.grid.coords_norm
                ge_str = ""

            # # determine indices of new basis functions and grid_points
            # idx_coords_new = [i for i, _id in enumerate(self.grid.coords_id) if _id not in self.gpc_matrix_coords_id]
            # idx_basis_new = [i for i, _id in enumerate(self.basis.b_id) if _id not in self.gpc_matrix_b_id]

            # determine indices of old grid points in updated gpc matrix
            idx_coords_old = np.empty(len(coords_id)) * np.nan
            for i, coords_id_old in enumerate(coords_id):
                for j, coords_id_new in enumerate(coords_id_ref):
                    if coords_id_old == coords_id_new:
                        idx_coords_old[i] = j
                        break

            # determine indices of old basis functions in updated gpc matrix
            idx_b_old = np.empty(len(b_id))*np.nan
            for i, b_id_old in enumerate(b_id):
                for j, b_id_new in enumerate(b_id_ref):
                    if b_id_old == b_id_new:
                        idx_b_old[i] = j
                        break

            # filter out non-existent rows and columns
            matrix = matrix[~np.isnan(idx_coords_old), :, ]
            matrix = matrix[:, ~np.isnan(idx_b_old), ]

            idx_coords_old = idx_coords_old[~np.isnan(idx_coords_old)].astype(int)
            idx_b_old = idx_b_old[~np.isnan(idx_b_old)].astype(int)

            # indices of new coords and basis in updated gpc matrix (values have to be computed there)
            idx_coords_new = np.array(list(set(np.arange(len(coords_id_ref))) - set(idx_coords_old))).astype(int)
            idx_b_new = np.array(list(set(np.arange(len(b_id_ref))) - set(idx_b_old))).astype(int)

            # write old results at correct location in updated gpc matrix
            idx = get_cartesian_product([idx_coords_old, idx_b_old]).astype(int)
            idx_row = np.reshape(idx[:, 0], matrix.shape[:2]).astype(int)
            idx_col = np.reshape(idx[:, 1], matrix.shape[:2]).astype(int)

            matrix_updated[idx_row, idx_col, ] = matrix

            # determine new columns (new basis functions) with old grid
            idx = get_cartesian_product([idx_coords_old, idx_b_new]).astype(int)
            if idx.any():
                iprint('Adding {} columns to gPC matrix {}...'.format(idx_b_new.size, ge_str), tab=0, verbose=True)

                idx_row = np.reshape(idx[:, 0], (idx_coords_old.size, idx_b_new.size)).astype(int)
                idx_col = np.reshape(idx[:, 1], (idx_coords_old.size, idx_b_new.size)).astype(int)

                matrix_updated[idx_row, idx_col, ] = self.create_gpc_matrix(b=[self.basis.b[i] for i in idx_b_new],
                                                                            x=coords_norm[idx_coords_old, :],
                                                                            gradient=gradient,
                                                                            verbose=False)

            # determine new rows (new grid points) with all basis functions
            idx = get_cartesian_product([idx_coords_new, np.arange(len(self.basis.b))]).astype(int)
            if idx.any():
                iprint('Adding {} rows to gPC matrix {}...'.format(idx_coords_new.size, ge_str), tab=0, verbose=True)

                idx_row = np.reshape(idx[:, 0], (idx_coords_new.size, len(self.basis.b))).astype(int)
                idx_col = np.reshape(idx[:, 1], (idx_coords_new.size, len(self.basis.b))).astype(int)

                matrix_updated[idx_row, idx_col, ] = self.create_gpc_matrix(b=self.basis.b,
                                                                            x=coords_norm[idx_coords_new, :],
                                                                            gradient=gradient,
                                                                            verbose=False)

            # overwrite old attributes and append new sizes
            if gradient:
                # reshape from 3D to 2D
                self.gpc_matrix_gradient = ten2mat(matrix_updated)
                self.gpc_matrix_gradient_coords_id = copy.deepcopy(self.grid.coords_id)
                self.gpc_matrix_gradient_b_id = copy.deepcopy(self.basis.b_id)
            else:
                self.gpc_matrix = matrix_updated
                self.gpc_matrix_coords_id = copy.deepcopy(self.grid.coords_id)
                self.gpc_matrix_b_id = copy.deepcopy(self.basis.b_id)
                self.n_grid.append(self.gpc_matrix.shape[0])
                self.n_basis.append(self.gpc_matrix.shape[1])

        else:
            self.init_gpc_matrix()

    def save_gpc_matrix_hdf5(self, hdf5_path_gpc_matrix=None, hdf5_path_gpc_matrix_gradient=None):
        """
        Save gPC matrix and gPC gradient matrix in .hdf5 file <"fn_results" + ".hdf5"> under the key "gpc_matrix"
        and "gpc_matrix_gradient". If matrices are already present, check for equality and save only appended
        rows and columns.

        Parameters
        ----------
        hdf5_path_gpc_matrix : str
            Path in .hdf5 file, where the gPC matrix is saved in
        hdf5_path_gpc_matrix_gradient : str
            Path in .hdf5 file, where the gPC gradient matrix is saved in
        """

        if hdf5_path_gpc_matrix is None:
            hdf5_path_gpc_matrix = "gpc_matrix"

        if hdf5_path_gpc_matrix_gradient is None:
            hdf5_path_gpc_matrix_gradient = "gpc_matrix_gradient"

        with h5py.File(self.fn_results + ".hdf5", "a") as f:
            try:
                # write gpc matrix
                gpc_matrix_hdf5 = f[hdf5_path_gpc_matrix][:]
                n_rows_hdf5 = gpc_matrix_hdf5.shape[0]
                n_cols_hdf5 = gpc_matrix_hdf5.shape[1]

                n_rows_current = self.gpc_matrix.shape[0]
                n_cols_current = self.gpc_matrix.shape[1]

                # save only new rows and cols if current matrix > saved matrix
                if n_rows_current >= n_rows_hdf5 and n_cols_current >= n_cols_hdf5 and \
                        (self.gpc_matrix[0:n_rows_hdf5, 0:n_cols_hdf5] == gpc_matrix_hdf5).all():
                    # resize dataset and save new columns and rows
                    f[hdf5_path_gpc_matrix].resize(self.gpc_matrix.shape[1], axis=1)
                    f[hdf5_path_gpc_matrix][:, n_cols_hdf5:] = self.gpc_matrix[0:n_rows_hdf5, n_cols_hdf5:]

                    f[hdf5_path_gpc_matrix].resize(self.gpc_matrix.shape[0], axis=0)
                    f[hdf5_path_gpc_matrix][n_rows_hdf5:, :] = self.gpc_matrix[n_rows_hdf5:, :]

                else:
                    del f[hdf5_path_gpc_matrix]
                    f.create_dataset(hdf5_path_gpc_matrix, (self.gpc_matrix.shape[0],
                                                            self.gpc_matrix.shape[1]),
                                     maxshape=(None, None),
                                     dtype="float64",
                                     data=self.gpc_matrix)

                # write gpc gradient matrix if available
                if self.gpc_matrix_gradient is not None:
                    gpc_matrix_gradient_hdf5 = f[hdf5_path_gpc_matrix_gradient][:]
                    n_rows_hdf5 = gpc_matrix_gradient_hdf5.shape[0]
                    n_cols_hdf5 = gpc_matrix_gradient_hdf5.shape[1]

                    n_rows_current = self.gpc_matrix_gradient.shape[0]
                    n_cols_current = self.gpc_matrix_gradient.shape[1]

                    # save only new rows and cols if current matrix > saved matrix
                    if n_rows_current >= n_rows_hdf5 and n_cols_current >= n_cols_hdf5 and \
                            (self.gpc_matrix_gradient[0:n_rows_hdf5, 0:n_cols_hdf5] == gpc_matrix_gradient_hdf5).all():
                        # resize dataset and save new columns and rows
                        f[hdf5_path_gpc_matrix_gradient].resize(self.gpc_matrix_gradient.shape[1], axis=1)
                        f[hdf5_path_gpc_matrix_gradient][:, n_cols_hdf5:] = self.gpc_matrix_gradient[0:n_rows_hdf5,
                                                                                                     n_cols_hdf5:]

                        f[hdf5_path_gpc_matrix_gradient].resize(self.gpc_matrix_gradient.shape[0], axis=0)
                        f[hdf5_path_gpc_matrix_gradient][n_rows_hdf5:, :] = self.gpc_matrix_gradient[n_rows_hdf5:, :]

                    else:
                        del f[hdf5_path_gpc_matrix_gradient]
                        f.create_dataset(hdf5_path_gpc_matrix_gradient, (self.gpc_matrix_gradient.shape[0],
                                                                         self.gpc_matrix_gradient.shape[1]),
                                         maxshape=(None, None),
                                         dtype="float64",
                                         data=self.gpc_matrix_gradient)

            except KeyError:
                # save whole matrix if not existent
                f.create_dataset(hdf5_path_gpc_matrix, (self.gpc_matrix.shape[0],
                                                        self.gpc_matrix.shape[1]),
                                 maxshape=(None, None),
                                 dtype="float64",
                                 data=self.gpc_matrix)

                # save whole gradient matrix if not existent and available
                if self.gpc_matrix_gradient is not None:
                    f.create_dataset(hdf5_path_gpc_matrix_gradient, (self.gpc_matrix_gradient.shape[0],
                                                                     self.gpc_matrix_gradient.shape[1]),
                                     maxshape=(None, None),
                                     dtype="float64",
                                     data=self.gpc_matrix_gradient)

    def solve(self, results, gradient_results=None, solver=None, settings=None, matrix=None, verbose=False):
        """
        Determines gPC coefficients

        Parameters
        ----------
        results : [n_grid x n_out] np.ndarray of float
            Results from simulations with N_out output quantities
        gradient_results : ndarray of float [n_gradient x n_out x dim], optional, default: None
            Gradient of results in original parameter space in specific grid points
        solver : str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
            - 'LarsLasso' ... Least-Angle Regression using Lasso model (SGPC.Reg, EGPC)
            - 'Tikhonov' ... Tikhonov regularization (SGPC.Reg, EGPC)
            - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
        settings : dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0 or "sparsity": float 0...1
            - 'LarsLasso' ... {"alpha": float 0...1} Regularization parameter
            - 'Tikhonov' ... {"alpha": float} Regularization parameter
            - 'NumInt' ... None
            - 
        matrix : ndarray of float, optional, default: self.gpc_matrix or [self.gpc_matrix, self.gpc_matrix_gradient]
            Matrix to invert. Depending on gradient_enhanced option, this matrix consist of the standard gPC matrix and
            their derivatives.
        verbose : bool
            boolean value to determine if to print out the progress into the standard output

        Returns
        -------
        coeffs: ndarray of float [n_coeffs x n_out]
            gPC coefficients
        """

        ge_str = ""

        if matrix is None:
            matrix = self.gpc_matrix

            if self.gradient is False:
                matrix = self.gpc_matrix
                ge_str = ""
            else:
                if not solver == 'NumInt':
                    if self.gpc_matrix_gradient is not None:
                        matrix = np.vstack((self.gpc_matrix, self.gpc_matrix_gradient))
                    else:
                        matrix = self.gpc_matrix
                    ge_str = "(gradient enhanced)"
                else:
                    Warning("Gradient enhanced version not applicable in case of numerical integration (quadrature).")

        # use default solver if not specified
        if solver is None:
            solver = self.solver

        # use default solver settings if not specified
        if solver is None:
            settings = self.settings

        iprint("Determine gPC coefficients using '{}' solver {}...".format(solver, ge_str),
               tab=0, verbose=verbose)

        # construct results array
        if not solver == 'NumInt' and gradient_results is not None:
            # transform gradient of results according to projection
            if self.p_matrix is not None:
                gradient_results = np.matmul(gradient_results,
                                             self.p_matrix.transpose() * self.p_matrix_norm[np.newaxis, :])

            results_complete = np.vstack((results, ten2mat(gradient_results)))
        else:
            results_complete = results

        self.coherence_matrix = matrix

        if isinstance(self.grid, CO) or (isinstance(self.grid, L1) and not (["D"] in self.grid.criterion)):
            w = np.diag(1/np.linalg.norm(matrix, axis=1))
            matrix = np.matmul(w, matrix)
            results_complete = np.matmul(w, results_complete)

        #################
        # Moore-Penrose #
        #################
        if solver == 'Moore-Penrose':
            # determine pseudoinverse of gPC matrix
            self.matrix_inv = np.linalg.pinv(matrix)

            try:
                coeffs = np.matmul(self.matrix_inv, results_complete)
            except ValueError:
                raise AttributeError("Please check format of parameter sim_results: [n_grid (* dim) x n_out] "
                                     "np.ndarray.")

        ###############################
        # Orthogonal Matching Pursuit #
        ###############################
        elif solver == 'OMP':
            try:
                import fastmat as fm
            except ImportError:
                raise ImportError("Please install the fastmat package to use the OMP solver.")
            # transform gPC matrix to fastmat format
            matrix_fm = fm.Matrix(matrix)

            if results_complete.ndim == 1:
                results_complete = results_complete[:, np.newaxis]

            # determine gPC-coefficients of extended basis using OMP
            if "n_coeffs_sparse" in settings.keys():
                n_coeffs_sparse = int(settings["n_coeffs_sparse"])
            elif "sparsity" in settings.keys():
                n_coeffs_sparse = int(np.ceil(matrix.shape[1]*settings["sparsity"]))
            else:
                raise AttributeError("Please specify 'n_coeffs_sparse' or 'sparsity' in solver settings dictionary!")

            coeffs = fm.algs.OMP(matrix_fm, results_complete, n_coeffs_sparse)

        ################################
        # Least-Angle Regression Lasso #
        ################################
        elif solver == 'LarsLasso':

            if results_complete.ndim == 1:
                results_complete = results_complete[:, np.newaxis]

            # determine gPC-coefficients of extended basis using LarsLasso
            # from sklearn.preprocessing import normalize
            # matrix_norm = normalize(matrix, axis=0, return_norm=False)
            #
            # reg = linear_model.LassoLars(alpha=settings["alpha"], fit_intercept=False, normalize=False)
            # reg.fit(matrix_norm, results_complete)
            # coeffs = reg.coef_

            # reg = linear_model.LassoLars(alpha=settings["alpha"], fit_intercept=False, normalize=False)
            reg = linear_model.LassoLars(alpha=settings["alpha"], fit_intercept=False)
            reg.fit(matrix, results_complete)
            coeffs = reg.coef_

            if coeffs.ndim == 1:
                coeffs = coeffs[:, np.newaxis]
            else:
                coeffs = coeffs.transpose()
                
        #########################
        # Tikhonov Regularization #
        #########################
        elif solver == 'Tikhonov':
            # solves inv(A.T A + alpha*Id) * A.T b based on the sklearn
            if results.ndim == 1:
                data = results[:, None]

            if "alpha" in settings.keys():
                alpha = settings["alpha"]
            else:
                raise AttributeError("Please specify 'regularization_factor' in solver setting dictionary!")

            n_samples, n_features = matrix.shape
            AtA = matrix.T.dot(matrix)
            Atb = matrix.T.dot(results)
            AtA.flat[::n_features-1] += alpha
            coeffs = scipy.linalg.solve(AtA, Atb, assume_a="pos", overwrite_a=True)
            
        #########################
        # Numerical Integration #
        #########################
        elif solver == 'NumInt':
            # check if quadrature rule (grid) fits to the probability density distribution (pdf)
            grid_pdf_fit = True
            for i_p, p in enumerate(self.problem.parameters_random):
                if self.problem.parameters_random[p].pdf_type == 'beta':
                    if not (self.grid.grid_type[i_p] == 'jacobi'):
                        grid_pdf_fit = False
                        break
                elif self.problem.parameters_random[p].pdf_type in ['norm', 'normal']:
                    if not (self.grid.grid_type[i_p] == 'hermite'):
                        grid_pdf_fit = False
                        break

            # if not, calculate joint pdf
            if not grid_pdf_fit:
                joint_pdf = np.ones(self.grid.coords_norm.shape)

                for i_p, p in enumerate(self.problem.parameters_random):
                    joint_pdf[:, i_p] = \
                        self.problem.parameters_random[p].pdf_norm(x=self.grid.coords_norm[:, i_p])

                joint_pdf = np.array([np.prod(joint_pdf, axis=1)]).transpose()

                # weight sim_results with the joint pdf
                results_complete = results_complete * joint_pdf * 2 ** self.problem.dim

            # scale rows of gpc matrix with quadrature weights
            matrix_weighted = np.matmul(np.diag(self.grid.weights), matrix)

            # determine gpc coefficients [n_coeffs x n_output]
            coeffs = np.matmul(results_complete.transpose(), matrix_weighted).transpose()

        else:
            raise AttributeError("Unknown solver: '{}'!")

        return coeffs

    def create_validation_set(self, n_samples, n_cpu=1):
        """
        Creates a ValidationSet instance (calls the model)

        Parameters
        ----------
        n_samples: int
            Number of sampling points contained in the validation set
        n_cpu: int
            Number of parallel function evaluations to evaluate validation set (n_cpu=0 assumes that the
            model is capable to evaluate all grid points in parallel)
        """
        # create set of validation points
        n_samples = n_samples

        if self.problem_original is not None:
            problem = self.problem_original
        else:
            problem = self.problem

        grid = Random(parameters_random=problem.parameters_random,
                      n_grid=n_samples,
                      options={"seed": self.options["seed"]})

        # Evaluate original model at grid points
        com = Computation(n_cpu=n_cpu, matlab_model=self.matlab_model)
        results = com.run(model=problem.model, problem=problem, coords=grid.coords)

        if results.ndim == 1:
            results = results[:, np.newaxis]

        self.validation = ValidationSet(grid=grid, results=results)
