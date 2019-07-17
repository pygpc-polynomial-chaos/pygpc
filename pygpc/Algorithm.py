# -*- coding: utf-8 -*-
import copy
import h5py
import os
import time
import shutil
from .Problem import *
from .SGPC import *
from .io import write_gpc_pkl
from .misc import determine_projection_matrix
from .misc import get_num_coeffs_sparse
from .misc import ten2mat
from .misc import mat2ten
from .misc import get_coords_discontinuity
from .misc import increment_basis
from .misc import get_num_coeffs_sparse
from .Grid import *
from .MEGPC import *
from .Classifier import Classifier


class Algorithm(object):
    """
    Class for GPC algorithms
    """

    def __init__(self, problem, options, grid=None, validation=None):
        """
        Constructor; Initializes GPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options : dict
            Algorithm specific options (see sub-classes for more details)
        grid : Grid object
            Grid object
        validation : ValidationSet object
            ValidationSet object
        """
        self.problem = problem
        self.problem_reduced = []
        self.validation = validation
        self.options = options
        self.grid = grid
        self.grid_gradient = []

        # Generate results folder if it doesn't exist
        if not os.path.exists(os.path.split(self.options["fn_results"])[0]):
            os.makedirs(os.path.split(self.options["fn_results"])[0])

        self.check_basic_options()

    def check_basic_options(self):
        """
        Checks self.options dictionary and sets default

        options["eps"] : float, optional, default=1e-3
            Relative mean error of leave-one-out cross validation
        options["error_norm"] : str, optional, default="relative"
            Choose if error is determined "relative" or "absolute". Use "absolute" error when the
            model generates outputs equal to zero.
        options["error_type"] : str, optional, default="loocv"
            Choose type of error to validate gpc approximation. Use "loocv" (Leave-one-Out cross validation)
            to omit any additional calculations and "nrmsd" (normalized root mean square deviation) to compare
            against a Problem.ValidationSet.
        options["fn_results"] : string, optional, default=None
            If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
        options["gradient_enhanced"] : boolean, optional, default: False
            Use gradient information to determine the gPC coefficients.
        options["gradient_calculation"] : str, optional, default="standard_forward"
            Type of the calculation scheme to determine the gradient in the grid points
            - "standard_forward" ... Forward approximation (creates additional dim*n_grid grid-points in the axis
            directions)
            - "???" ... ???
        options["GPU"] : boolean, optional, default: False
            Use GPU to accelerate pygpc
        options["lambda_eps_gradient"] : float, optional, default: 0.95
            Bound of principal components in %. All eigenvectors are included until lambda_eps of total sum of all
            eigenvalues is included in the system.
        options["matrix_ratio"]: float, optional, default=1.5
            Ration between the number of model evaluations and the number of basis functions.
            If "adaptive_sampling" is activated this factor is only used to
            construct the initial grid depending on the initial number of basis functions determined by "order_start".
            (>1 results in an overdetermined system)
        options["matlab_model"] : boolean, optional, default: False
            Use a Matlab model function
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["n_samples_validation"] : int, optional, default: 1e4
            Number of validation points used to determine the NRMSD if chosen as "error_type". Does not create a
            validation set if there is already one present in the Problem instance (problem.validation).
        options["print_func_time"] : boolean, optional, default: False
            Print function evaluation time for every single run
        options["projection"] : boolean, optional, default: False
            Use projection approach
        options["seed"] : int, optional, default=None
            Set np.random.seed(seed) in random_grid()
        options["solver"]: str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        options["settings"]: dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
        options["verbose"] : boolean, optional, default=True
            Print output of iterations and sub-iterations (True/False)
        """

        if "eps" not in self.options.keys():
            self.options["eps"] = 1e-3

        if "error_norm" not in self.options.keys():
            self.options["error_norm"] = "relative"

        if "error_type" not in self.options.keys():
            self.options["error_type"] = "loocv"

        if "fn_results" not in self.options.keys():
            self.options["fn_results"] = None

        if "gradient_enhanced" not in self.options.keys():
            self.options["gradient_enhanced"] = False

        if "gradient_calculation" not in self.options.keys():
            self.options["gradient_calculation"] = "standard_forward"

        if "GPU" not in self.options.keys():
            self.options["GPU"] = False

        if "lambda_eps_gradient" not in self.options.keys():
            self.options["lambda_eps_gradient"] = 0.95

        if "matrix_ratio" not in self.options.keys():
            self.options["matrix_ratio"] = 2

        if "matlab_model" not in self.options.keys():
            self.options["matlab_model"] = False

        if "method" in self.options.keys():
            if self.options["method"] == "quad":
                self.options["solver"] = 'NumInt'
                self.options["settings"] = None
            elif self.options["method"] == "reg" and not (self.options["solver"] == "Moore-Penrose" or
                                                          self.options["solver"] == "OMP"):
                raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for 'reg' method")

        if "n_cpu" in self.options.keys():
            self.n_cpu = self.options["n_cpu"]

        if "n_samples_validation" not in self.options.keys():
            self.options["n_samples_validation"] = 1e4

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "projection" not in self.options.keys():
            self.options["projection"] = False

        if "seed" not in self.options.keys():
            self.options["seed"] = None

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and ("settings" not in self.options.keys() or not (
                "n_coeffs_sparse" not in self.options["settings"].keys() or
                "sparsity" not in self.options["settings"].keys())):
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if self.options["solver"] == "LarsLasso":
            if "settings" in self.options.keys():
                if self.options["settings"] is dict():
                    if "alpha" not in self.options["settings"].keys():
                        self.options["settings"]["alpha"] = 1e-5
                else:
                    self.options["settings"] = {"alpha": 1e-5}
            else:
                self.options["settings"] = {"alpha": 1e-5}


        if "verbose" not in self.options.keys():
            self.options["verbose"] = True

    def get_gradient(self, grid, results, com, gradient_results=None):
        """
        Determines the gradient of the model function in the grid points (self.grid.coords).
        The method to determine the gradient can be specified in self.options["gradient_calculation"] to be either:

        - "standard_forward" ... Forward approximation of the gradient using n_grid x dim additional sampling points
          stored in self.grid.coords_gradient and self.grid.coords_gradient_norm [n_grid x dim x dim].
        - "???" ... ???

        Parameters
        ----------
        grid : Grid object
            Grid object
        results : ndarray of float [n_grid x n_out]
            Results of model function in grid points
        com : Computation class instance
            Computation class instance to run the computations
        gradient_results : ndarray of float [n_grid_old x n_out x dim], optional, default: None
            Gradient of model function in grid points, already determined in previous calculations.

        Returns
        -------
        gradient_results : ndarray of float [n_grid x n_out x dim]
            Gradient of model function in grid points
        """
        if gradient_results is not None:
            n_gradient_results = gradient_results.shape[0]
        else:
            n_gradient_results = 0

        if n_gradient_results < results.shape[0]:
            #########################################
            # Standard forward gradient calculation #
            #########################################
            if self.options["gradient_calculation"] == "standard_forward":
                # add new grid points for gradient calculation in grid.coords_gradient and grid.coords_gradient_norm
                grid.create_gradient_grid(delta=1e-3)

                results_gradient_tmp = com.run(model=self.problem.model,
                                               problem=self.problem,
                                               coords=ten2mat(grid.coords_gradient[n_gradient_results:, :, :]),
                                               coords_norm=ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]),
                                               i_iter=None,
                                               i_subiter=None,
                                               fn_results=None,
                                               print_func_time=self.options["print_func_time"])

                delta = np.repeat(np.linalg.norm(
                    ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]) - \
                    ten2mat(np.repeat(grid.coords_norm[n_gradient_results:, :, np.newaxis], self.problem.dim, axis=2)),
                    axis=1)[:, np.newaxis], results_gradient_tmp.shape[1], axis=1)

                gradient_results_new = (ten2mat(np.repeat(results[n_gradient_results:, :, np.newaxis], self.problem.dim, axis=2)) - results_gradient_tmp) / delta

                gradient_results_new = mat2ten(mat=gradient_results_new, incr=self.problem.dim)

                if gradient_results is not None:
                    gradient_results = np.vstack((gradient_results, gradient_results_new))
                else:
                    gradient_results = gradient_results_new

            ##############################
            # Gradient approximation ... #
            ##############################
            # TODO: implement gradient approximation schemes using neighbouring grid points

        return gradient_results


class Static(Algorithm):
    """
    Static gPC algorithm
    """
    def __init__(self, problem, options, grid, validation=None):
        """
        Constructor; Initializes static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["order"]: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        options["order_max"]: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["interaction_order"]: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        grid: Grid object instance
            Grid object to use for static gPC (RandomGrid, SparseGrid, TensorGrid)

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle 
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> # initialize static gPC algorithm
        >>> algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(Static, self).__init__(problem=problem, options=options, validation=validation)
        self.grid = grid

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim


    def run(self):
        """
        Runs static gPC algorithm to solve problem.

        Returns
        -------
        gpc : GPC object instance
            GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """
        grad_res_3D = None
        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        # Create gPC object
        if self.options["method"] == "reg":
            gpc = Reg(problem=self.problem,
                      order=self.options["order"],
                      order_max=self.options["order_max"],
                      order_max_norm=self.options["order_max_norm"],
                      interaction_order=self.options["interaction_order"],
                      interaction_order_current=self.options["interaction_order"],
                      options=self.options,
                      validation=self.validation)

        elif self.options["method"] == "quad":
            gpc = Quad(problem=self.problem,
                       order=self.options["order"],
                       order_max=self.options["order_max"],
                       order_max_norm=self.options["order_max_norm"],
                       interaction_order=self.options["interaction_order"],
                       interaction_order_current=self.options["interaction_order"],
                       options=self.options,
                       validation=self.validation)

        else:
            raise AssertionError("Please specify correct gPC method ('reg' or 'quad')")

        gpc.gpu = self.options["GPU"]

        # Write grid in gpc object
        gpc.grid = copy.deepcopy(self.grid)
        gpc.interaction_order_current = copy.deepcopy(self.options["interaction_order"])

        # Initialize gpc matrix
        gpc.init_gpc_matrix()

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run simulations
        iprint("Performing {} simulations!".format(gpc.grid.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res = com.run(model=gpc.problem.model,
                      problem=gpc.problem,
                      coords=gpc.grid.coords,
                      coords_norm=gpc.grid.coords_norm,
                      i_iter=gpc.order_max,
                      i_subiter=gpc.interaction_order,
                      fn_results=None,
                      print_func_time=self.options["print_func_time"])

        iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine gradient [n_grid x n_out x dim]
        if self.options["gradient_enhanced"]:
            start_time = time.time()
            grad_res_3D = self.get_gradient(grid=gpc.grid,
                                            results=res,
                                            com=com)
            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                   tab=0, verbose=self.options["verbose"])

        # Compute gpc coefficients
        coeffs = gpc.solve(results=res,
                           gradient_results=grad_res_3D,
                           solver=self.options["solver"],
                           settings=self.options["settings"],
                           verbose=True)

        # create validation set if necessary
        if self.options["error_type"] == "nrmsd" and gpc.validation is None:
            gpc.create_validation_set(n_samples=self.options["n_samples_validation"],
                                      n_cpu=self.options["n_cpu"])

        # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
        eps = gpc.validate(coeffs=coeffs, results=res, gradient_results=grad_res_3D)

        iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                            self.options["error_type"],
                                            eps), tab=0, verbose=self.options["verbose"])

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:

            fn_gpc_pkl = fn_results + '.pkl'
            write_gpc_pkl(gpc, fn_gpc_pkl)

            with h5py.File(fn_results + ".hdf5", "a") as f:

                f.create_dataset("misc/fn_gpc_pkl", data=np.array([os.path.split(fn_gpc_pkl)[1]]).astype("|S"))
                f.create_dataset("misc/error_type", data=self.options["error_type"])
                f.create_dataset("error", data=eps, maxshape=None, dtype="float64")
                f.create_dataset("grid/coords", data=gpc.grid.coords, maxshape=None, dtype="float64")
                f.create_dataset("grid/coords_norm", data=gpc.grid.coords_norm, maxshape=None, dtype="float64")

                if gpc.grid.coords_gradient is not None:
                    f.create_dataset("grid/coords_gradient", data=gpc.grid.coords_gradient,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_gradient_norm", data=gpc.grid.coords_gradient_norm,
                                     maxshape=None, dtype="float64")

                f.create_dataset("coeffs", data=coeffs,
                                 maxshape=None, dtype="float64")
                f.create_dataset("gpc_matrix", data=gpc.gpc_matrix,
                                 maxshape=None, dtype="float64")

                if gpc.gpc_matrix_gradient is not None:
                    f.create_dataset("gpc_matrix_gradient", data=gpc.gpc_matrix_gradient, maxshape=None, dtype="float64")

                f.create_dataset("model_evaluations/results", data=res, maxshape=None, dtype="float64")

                if grad_res_3D is not None:
                    f.create_dataset("model_evaluations/gradient_results", data=ten2mat(grad_res_3D), maxshape=None, dtype="float64")

                if gpc.validation is not None:
                    f.create_dataset("validation/results", data=gpc.validation.results,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords", data=gpc.validation.grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords_norm", data=gpc.validation.grid.coords_norm,
                                     maxshape=None, dtype="float64")

        com.close()

        return gpc, coeffs, res


class MEStatic(Algorithm):
    """
    Multi-Element Static gPC algorithm
    """
    def __init__(self, problem, options, grid, validation=None):
        """
        Constructor; Initializes multi-element static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["order"]: list of int [dim]
            Maximum individual expansion order [order_1, order_2, ..., order_dim].
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        options["order_max"]: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["interaction_order"]: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        options["qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["classifier"] : str, optional, default: "learning"
            Classification algorithm to subdivide parameter domain.
            - "learning" ... ClassifierLearning algorithm based on Unsupervised and supervised learning
        options["classifier_options"] : dict, optional, default: default settings
            Options of classifier
        grid: Grid object instance
            Grid object to use for static gPC (RandomGrid, SparseGrid, TensorGrid)

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> # initialize static gPC algorithm
        >>> algorithm = pygpc.MEStatic(problem=problem, options=options, grid=grid)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(MEStatic, self).__init__(problem=problem, options=options, validation=validation)
        self.grid = grid

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "qoi" not in self.options.keys():
            self.options["qoi"] = 0

        if "classifier" not in self.options.keys():
            self.options["classifier"] = "learning"

        if "classifier_options" not in self.options.keys():
            self.options["classifier_options"] = None

    def run(self):
        """
        Runs Multi-Element Static gPC algorithm to solve problem.

        Returns
        -------
        megpc : Multi-element GPC object instance
            MEGPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """
        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        grad_res_3D = None
        grid = self.grid

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run simulations
        iprint("Performing {} simulations!".format(grid.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res_all = com.run(model=self.problem.model,
                          problem=self.problem,
                          coords=grid.coords,
                          coords_norm=grid.coords_norm,
                          i_iter=self.options["order_max"],
                          i_subiter=self.options["interaction_order"],
                          fn_results=None,
                          print_func_time=self.options["print_func_time"])

        iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        if self.options["qoi"] == "all":
            qoi_idx = np.arange(res_all.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["qoi"]]
            n_qoi = 1

        # Determine gradient [n_grid x n_out x dim]
        if self.options["gradient_enhanced"]:
            start_time = time.time()
            grad_res_3D_all = self.get_gradient(grid=grid,
                                                results=res_all,
                                                com=com)
            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                   tab=0, verbose=self.options["verbose"])

        megpc = [0 for _ in range(n_qoi)]
        coeffs = [0 for _ in range(n_qoi)]

        for i_qoi, q_idx in enumerate(qoi_idx):

            # crop results to considered qoi
            if self.options["qoi"] != "all":
                res = copy.deepcopy(res_all)
                grad_res_3D = copy.deepcopy(grad_res_3D_all)
                hdf5_subfolder = ""
                output_idx_passed_validation = None

            else:
                res = res_all[:, q_idx][:, np.newaxis]
                hdf5_subfolder = "/qoi_" + str(q_idx)
                output_idx_passed_validation = q_idx

                if grad_res_3D_all is not None:
                    grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

            # Create MEGPC object
            megpc[i_qoi] = MEGPC(problem=self.problem,
                                 options=self.options,
                                 validation=self.validation)

            # Write grid in gpc object
            megpc[i_qoi].grid = copy.deepcopy(grid)

            # determine gpc domains
            megpc[i_qoi].init_classifier(coords=megpc[i_qoi].grid.coords_norm,
                                         results=res_all[:, q_idx][:, np.newaxis],
                                         algorithm=self.options["classifier"],
                                         options=self.options["classifier_options"])

            # initialize sub-gPCs
            for d in np.unique(megpc[i_qoi].domains):
                megpc[i_qoi].add_sub_gpc(problem=megpc[i_qoi].problem,
                                         order=[self.options["order"][0] for _ in range(megpc[i_qoi].problem.dim)],
                                         order_max=self.options["order_max"],
                                         order_max_norm=self.options["order_max_norm"],
                                         interaction_order=self.options["interaction_order"],
                                         interaction_order_current=self.options["interaction_order"],
                                         options=self.options,
                                         domain=d,
                                         validation=None)

            # assign grids to sub-gPCs (rotate sub-grids in case of projection)
            megpc[i_qoi].assign_grids()

            # Initialize gpc matrices
            megpc[i_qoi].init_gpc_matrices()

            # Compute gpc coefficients
            coeffs[i_qoi] = megpc[i_qoi].solve(results=res,
                                               gradient_results=grad_res_3D,
                                               solver=self.options["solver"],
                                               settings=self.options["settings"],
                                               verbose=True)

            # create validation set if necessary
            if self.options["error_type"] == "nrmsd" and megpc[0].validation is None:
                megpc[0].create_validation_set(n_samples=self.options["n_samples_validation"],
                                               n_cpu=self.options["n_cpu"])
            elif self.options["error_type"] == "nrmsd" and megpc[0].validation is not None:
                megpc[i_qoi].validation = copy.deepcopy(megpc[0].validation)

            # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
            eps = megpc[i_qoi].validate(coeffs=coeffs[i_qoi], results=res, gradient_results=grad_res_3D)

            iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                self.options["error_type"],
                                                eps), tab=0, verbose=self.options["verbose"])

            # domain specific error
            eps_domain = [0 for _ in range(len(np.unique(megpc[i_qoi].domains)))]
            for i_gpc, d in enumerate(np.unique(megpc[i_qoi].domains)):
                eps_domain[d] = megpc[i_qoi].validate(coeffs=coeffs[i_qoi],
                                                      results=res,
                                                      domain=d,
                                                      output_idx=output_idx_passed_validation)

            # save data
            if self.options["fn_results"]:

                fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                write_gpc_pkl(megpc[i_qoi], fn_gpc_pkl_qoi)

                with h5py.File(fn_results + ".hdf5", "a") as f:

                    try:
                        fn_gpc_pkl = f["misc/fn_gpc_pkl"]
                        fn_gpc_pkl = np.vstack((fn_gpc_pkl, np.array([os.path.split(fn_gpc_pkl_qoi)[1]])))
                        del f["misc/fn_gpc_pkl"]
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=fn_gpc_pkl.astype("|S"))

                    except KeyError:
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=np.array([os.path.split(fn_gpc_pkl_qoi)[1]]).astype("|S"))

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("error" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=eps_domain[i_gpc],
                                         maxshape=None, dtype="float64")

                        f.create_dataset("coeffs" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=coeffs[i_qoi][i_gpc],
                                         maxshape=None, dtype="float64")

                    f.create_dataset("domains" + hdf5_subfolder,
                                     data=megpc[i_qoi].domains,
                                     maxshape=None, dtype="int64")

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("gpc_matrix" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=megpc[i_qoi].gpc[i_gpc].gpc_matrix,
                                         maxshape=None, dtype="float64")

                    if megpc[i_qoi].gpc[0].gpc_matrix_gradient is not None:
                        if self.options["gradient_enhanced"]:
                            for i_gpc in range(megpc[i_qoi].n_gpc):
                                f.create_dataset("gpc_matrix_gradient" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                                 data=megpc[i_qoi].gpc[i_gpc].gpc_matrix_gradient,
                                                 maxshape=None, dtype="float64")

        if self.options["fn_results"]:

            with h5py.File(fn_results + ".hdf5", "a") as f:
                f.create_dataset("grid/coords", data=grid.coords,
                                 maxshape=None, dtype="float64")
                f.create_dataset("grid/coords_norm", data=grid.coords_norm,
                                 maxshape=None, dtype="float64")

                if megpc[i_qoi].grid.coords_gradient is not None:
                    f.create_dataset("grid/coords_gradient",
                                     data=grid.coords_gradient,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_gradient_norm",
                                     data=grid.coords_gradient_norm,
                                     maxshape=None, dtype="float64")

                f.create_dataset("model_evaluations/results", data=res,
                                 maxshape=None, dtype="float64")
                if grad_res_3D is not None:
                    f.create_dataset("model_evaluations/gradient_results", data=ten2mat(grad_res_3D),
                                     maxshape=None, dtype="float64")

                f.create_dataset("misc/error_type", data=self.options["error_type"])

                if megpc[0].validation is not None:
                    f.create_dataset("validation/results", data=megpc[0].validation.results,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords", data=megpc[0].validation.grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords_norm", data=megpc[0].validation.grid.coords_norm,
                                     maxshape=None, dtype="float64")

        com.close()

        if n_qoi == 1:
            return megpc[0], coeffs[0], res
        else:
            return megpc, coeffs, res


class StaticProjection(Algorithm):
    """
    Static gPC algorithm using Basis Projection approach
    """
    def __init__(self, problem, options, validation=None):
        """
        Constructor; Initializes static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["order"]: int
            Expansion order, each projected variable \\eta is expanded to.
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        options["order_max"]: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["interaction_order"]: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        options["qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["n_grid_gradient"] : float, optional, default: 10
            Number of initial grid points to determine gradient and projection matrix

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> # initialize static gPC algorithm
        >>> algorithm = pygpc.StaticProjection(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run
        """
        super(StaticProjection, self).__init__(problem=problem, options=options, validation=validation)

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "n_grid_gradient" not in self.options.keys():
            self.options["n_grid_gradient"] = 10

        if "qoi" not in self.options.keys():
            self.options["qoi"] = 0

    def run(self):
        """
        Runs static gPC algorithm using Projection to solve problem.

        Returns
        -------
        gpc : GPC object instance
            GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: list ndarray of float [n_qoi][n_basis x n_out]
            GPC coefficients of dofferent qoi
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """

        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        grad_res_3D = None

        # make initial random grid to determine gradients and projection matrix
        grid_original = RandomGrid(parameters_random=self.problem.parameters_random,
                                   options={"n_grid": self.options["n_grid_gradient"]})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run simulations
        iprint("Performing {} simulations!".format(grid_original.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res_all = com.run(model=self.problem.model,
                          problem=self.problem,
                          coords=grid_original.coords,
                          coords_norm=grid_original.coords_norm,
                          i_iter=self.options["order_max"],
                          i_subiter=self.options["interaction_order"],
                          fn_results=None,
                          print_func_time=self.options["print_func_time"])

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine gradient
        start_time = time.time()
        grad_res_3D_all = self.get_gradient(grid=grid_original,
                                            results=res_all,
                                            com=com)
        iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        if self.options["qoi"] == "all":
            qoi_idx = np.arange(res_all.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["qoi"]]
            n_qoi = 1

        # Set up reduced gPC
        self.problem_reduced = [0 for _ in range(n_qoi)]
        gpc = [0 for _ in range(n_qoi)]
        coeffs = [0 for _ in range(n_qoi)]

        for i_qoi, q_idx in enumerate(qoi_idx):

            # crop results to considered qoi
            if self.options["qoi"] != "all":
                res = copy.deepcopy(res_all)
                grad_res_3D = copy.deepcopy(grad_res_3D_all)
                hdf5_subfolder = ""

            else:
                res = res_all[:, q_idx][:, np.newaxis]
                grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]
                hdf5_subfolder = "/qoi_" + str(q_idx)

            # Determine projection matrix
            p_matrix = determine_projection_matrix(gradient_results=grad_res_3D_all,
                                                   qoi_idx=q_idx,
                                                   lambda_eps=self.options["lambda_eps_gradient"])
            p_matrix_norm = np.sum(np.abs(p_matrix), axis=1)

            dim_reduced = p_matrix.shape[0]
            parameters_reduced = OrderedDict()

            for i in range(dim_reduced):
                parameters_reduced["n{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

            self.problem_reduced[i_qoi] = Problem(model=self.problem.model, parameters=parameters_reduced)

            # Create reduced gPC object
            gpc[i_qoi] = Reg(problem=self.problem_reduced[i_qoi],
                             order=[self.options["order"][0] for _ in range(dim_reduced)],
                             order_max=self.options["order_max"],
                             order_max_norm=self.options["order_max_norm"],
                             interaction_order=self.options["interaction_order"],
                             interaction_order_current=self.options["interaction_order"],
                             options=self.options,
                             validation=self.validation)

            # save original problem in gpc object
            gpc[i_qoi].problem_original = self.problem

            # save projection matrix in gPC object
            gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)
            gpc[i_qoi].p_matrix_norm = copy.deepcopy(p_matrix_norm)

            # extend initial grid and perform additional simulations if necessary
            n_coeffs = get_num_coeffs_sparse(order_dim_max=[self.options["order"][0] for _ in range(dim_reduced)],
                                             order_glob_max=self.options["order_max"],
                                             order_inter_max=self.options["interaction_order"],
                                             dim=dim_reduced)

            n_grid_new = n_coeffs * self.options["matrix_ratio"]

            if n_grid_new > grid_original.n_grid:
                iprint("Extending grid from {} to {} grid points ...".format(grid_original.n_grid, n_grid_new),
                       tab=0, verbose=self.options["verbose"])

                # extend grid
                grid_original.extend_random_grid(n_grid_new=n_grid_new, seed=self.options["seed"])

                # Run simulations
                iprint("Performing {} additional simulations!".format(grid_original.coords.shape[0]),
                       tab=0, verbose=self.options["verbose"])

                start_time = time.time()

                res_new = com.run(model=self.problem.model,
                                  problem=self.problem,
                                  coords=grid_original.coords[self.options["n_grid_gradient"]:, :],
                                  coords_norm=grid_original.coords_norm[self.options["n_grid_gradient"]:, :],
                                  i_iter=self.options["order_max"],
                                  i_subiter=self.options["interaction_order"],
                                  fn_results=None,
                                  print_func_time=self.options["print_func_time"])

                res_all = np.vstack((res_all, res_new))

                iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
                       tab=0, verbose=self.options["verbose"])

                # Determine gradient [n_grid x n_out x dim]
                if self.options["gradient_enhanced"]:
                    start_time = time.time()
                    grad_res_3D_all = self.get_gradient(grid=grid_original,
                                                        results=res_all,
                                                        com=com,
                                                        gradient_results=grad_res_3D_all)
                    iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                           tab=0, verbose=self.options["verbose"])

                # crop results to considered qoi
                if self.options["qoi"] != "all":
                    res = copy.deepcopy(res_all)
                    grad_res_3D = copy.deepcopy(grad_res_3D_all)

                else:
                    res = res_all[:, q_idx][:, np.newaxis]
                    grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

            # copy grid to gPC object and initialize transformed grid
            gpc[i_qoi].grid_original = copy.deepcopy(grid_original)
            gpc[i_qoi].grid = copy.deepcopy(grid_original)

            # transform variables of original grid to reduced parameter space
            gpc[i_qoi].grid.coords = np.dot(gpc[i_qoi].grid_original.coords,
                                            gpc[i_qoi].p_matrix.transpose())
            gpc[i_qoi].grid.coords_norm = np.dot(gpc[i_qoi].grid_original.coords_norm,
                                                 gpc[i_qoi].p_matrix.transpose() /
                                                 gpc[i_qoi].p_matrix_norm[np.newaxis, :])

            # gpc_red.interaction_order_current = 1
            # self.options_red = copy.deepcopy(self.options)
            # self.options_red["interaction_order"] = 1
            gpc[i_qoi].options = copy.deepcopy(self.options)

            # Initialize gpc matrix
            gpc[i_qoi].init_gpc_matrix()

            # Someone might not use the gradient to determine the gpc coeffs
            if self.options["gradient_enhanced"]:
                grad_res_3D_passed = grad_res_3D
            else:
                grad_res_3D_passed = None

            # Compute gpc coefficients
            coeffs[i_qoi] = gpc[i_qoi].solve(results=res,
                                             gradient_results=grad_res_3D_passed,
                                             solver=self.options["solver"],
                                             settings=self.options["settings"],
                                             verbose=True)

            # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
            if self.options["error_type"] == "nrmsd" and gpc[0].validation is None:
                gpc[0].create_validation_set(n_samples=self.options["n_samples_validation"],
                                             n_cpu=self.options["n_cpu"])
            elif self.options["error_type"] == "nrmsd" and gpc[0].validation is not None:
                gpc[i_qoi].validation = copy.deepcopy(gpc[0].validation)

            eps = gpc[i_qoi].validate(coeffs=coeffs[i_qoi], results=res, gradient_results=grad_res_3D_passed)

            iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                self.options["error_type"],
                                                eps), tab=0, verbose=self.options["verbose"])

            # save gpc objects and gpc coeffs
            if self.options["fn_results"]:

                fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                write_gpc_pkl(gpc[i_qoi], fn_gpc_pkl_qoi)

                with h5py.File(fn_results + ".hdf5", "a") as f:

                    try:
                        fn_gpc_pkl = f["misc/fn_gpc_pkl"]
                        fn_gpc_pkl = np.vstack((fn_gpc_pkl, np.array([os.path.split(fn_gpc_pkl_qoi)[1]])))
                        del f["misc/fn_gpc_pkl"]
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=fn_gpc_pkl.astype("|S"))

                    except KeyError:
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=np.array([os.path.split(fn_gpc_pkl_qoi)[1]]).astype("|S"))

                    f.create_dataset("error" + hdf5_subfolder,
                                     data=eps,
                                     maxshape=None, dtype="float64")

                    f.create_dataset("coeffs" + hdf5_subfolder,
                                     data=coeffs[i_qoi],
                                     maxshape=None, dtype="float64")

                    f.create_dataset("gpc_matrix" + hdf5_subfolder,
                                     data=gpc[i_qoi].gpc_matrix,
                                     maxshape=None, dtype="float64")

                    if gpc[i_qoi].gpc_matrix_gradient is not None:
                        f.create_dataset("gpc_matrix_gradient" + hdf5_subfolder,
                                         data=gpc[i_qoi].gpc_matrix_gradient,
                                         maxshape=None, dtype="float64")

                    f.create_dataset("p_matrix" + hdf5_subfolder,
                                     data=gpc[i_qoi].p_matrix,
                                     maxshape=None, dtype="float64")

        if self.options["fn_results"]:

            with h5py.File(fn_results + ".hdf5", "a") as f:
                f.create_dataset("grid/coords", data=grid_original.coords,
                                 maxshape=None, dtype="float64")
                f.create_dataset("grid/coords_norm", data=grid_original.coords_norm,
                                 maxshape=None, dtype="float64")

                if gpc[i_qoi].grid.coords_gradient is not None:
                    f.create_dataset("grid/coords_gradient",
                                     data=grid_original.coords_gradient,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_gradient_norm",
                                     data=grid_original.coords_gradient_norm,
                                     maxshape=None, dtype="float64")

                f.create_dataset("model_evaluations/results", data=res,
                                 maxshape=None, dtype="float64")
                if grad_res_3D is not None:
                    f.create_dataset("model_evaluations/gradient_results", data=ten2mat(grad_res_3D),
                                     maxshape=None, dtype="float64")

                f.create_dataset("misc/error_type", data=self.options["error_type"])

                if gpc[0].validation is not None:
                    f.create_dataset("validation/results", data=gpc[0].validation.results,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords", data=gpc[0].validation.grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords_norm", data=gpc[0].validation.grid.coords_norm,
                                     maxshape=None, dtype="float64")

        com.close()

        if n_qoi == 1:
            return gpc[0], coeffs[0], res_all
        else:
            return gpc, coeffs, res_all


class MEStaticProjection(Algorithm):
    """
    Static gPC algorithm using Basis Projection approach
    """
    def __init__(self, problem, options, validation=None):
        """
        Constructor; Initializes static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["order"]: int
            Expansion order, each projected variable \\eta is expanded to.
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        options["order_max"]: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        options["interaction_order"]: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        options["qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["n_grid_gradient"] : float, optional, default: 10
            Number of initial grid points to determine gradient and projection matrix
        options["classifier"] : str, optional, default: "learning"
            Classification algorithm to subdivide parameter domain.
            - "learning" ... ClassifierLearning algorithm based on Unsupervised and supervised learning
        options["classifier_options"] : dict, optional, default: default settings
            Options of classifier

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        Examples
        --------
        >>> import pygpc
        >>> # initialize static gPC algorithm
        >>> algorithm = pygpc.MEStaticProjection(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run
        """
        super(MEStaticProjection, self).__init__(problem=problem, options=options, validation=validation)

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "n_grid_gradient" not in self.options.keys():
            self.options["n_grid_gradient"] = 10

        if "qoi" not in self.options.keys():
            self.options["qoi"] = 0

        if "classifier" not in self.options.keys():
            self.options["classifier"] = "learning"

        if "classifier_options" not in self.options.keys():
            self.options["classifier_options"] = None

    def run(self):
        """
        Runs static multi-element gPC algorithm with projection.

        Returns
        -------
        megpc : MEGPC object instance
            MEGPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
            and sub-gPCs
        coeffs: list of list of ndarray of float [n_qoi][n_gpc][n_basis x n_out]
            GPC coefficients of different qoi and sub-gPCs
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """
        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        # make initial random grid to determine gradients and projection matrix
        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": self.options["n_grid_gradient"]})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run simulations
        iprint("Performing {} simulations!".format(grid.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res_all = com.run(model=self.problem.model,
                          problem=self.problem,
                          coords=grid.coords,
                          coords_norm=grid.coords_norm,
                          i_iter=self.options["order_max"],
                          i_subiter=self.options["interaction_order"],
                          fn_results=None,
                          print_func_time=self.options["print_func_time"])

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine gradient
        start_time = time.time()
        grad_res_3D_all = self.get_gradient(grid=grid,
                                            results=res_all,
                                            com=com)
        iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        if self.options["qoi"] == "all":
            qoi_idx = np.arange(res_all.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["qoi"]]
            n_qoi = 1

        megpc = [0 for _ in range(n_qoi)]
        coeffs = [0 for _ in range(n_qoi)]

        for i_qoi, q_idx in enumerate(qoi_idx):

            # crop results to considered qoi
            if self.options["qoi"] != "all":
                res = copy.deepcopy(res_all)
                grad_res_3D = copy.deepcopy(grad_res_3D_all)
                hdf5_subfolder = ""
                output_idx_passed_validation = None

            else:
                res = res_all[:, q_idx][:, np.newaxis]
                grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]
                hdf5_subfolder = "/qoi_" + str(q_idx)
                output_idx_passed_validation = q_idx

            # Create reduced gPC object
            megpc[i_qoi] = MEGPC(problem=self.problem,
                                 options=self.options,
                                 validation=self.validation)

            megpc[i_qoi].grid = copy.deepcopy(grid)

            # determine gpc domains
            megpc[i_qoi].init_classifier(coords=megpc[i_qoi].grid.coords_norm,
                                         results=res_all[:, q_idx][:, np.newaxis],
                                         algorithm=self.options["classifier"],
                                         options=self.options["classifier_options"])

            problem_reduced = [0 for _ in range(megpc[i_qoi].n_gpc)]
            p_matrix = [0 for _ in range(megpc[i_qoi].n_gpc)]
            p_matrix_norm = [0 for _ in range(megpc[i_qoi].n_gpc)]
            dim_reduced = [0 for _ in range(megpc[i_qoi].n_gpc)]
            parameters_reduced = [OrderedDict() for _ in range(megpc[i_qoi].n_gpc)]
            megpc[i_qoi].gpc = [0 for _ in range(megpc[i_qoi].n_gpc)]
            n_grid_new = [0 for _ in range(megpc[i_qoi].n_gpc)]

            # Determine projection matrices for sub gPCs
            for d in np.unique(megpc[i_qoi].domains):
                p_matrix[d] = determine_projection_matrix(gradient_results=grad_res_3D_all[megpc[i_qoi].domains == d, :, :],
                                                          qoi_idx=q_idx,
                                                          lambda_eps=self.options["lambda_eps_gradient"])

                p_matrix_norm[d] = np.sum(np.abs(p_matrix[d]), axis=1)
                dim_reduced[d] = p_matrix[d].shape[0]

                for i in range(dim_reduced[d]):
                    parameters_reduced[d]["n{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

                problem_reduced[d] = Problem(model=self.problem.model, parameters=parameters_reduced[d])

                # Set up reduced gPC for this domain
                megpc[i_qoi].add_sub_gpc(problem=problem_reduced[d],
                                         order=[self.options["order"][0] for _ in range(dim_reduced[d])],
                                         order_max=self.options["order_max"],
                                         order_max_norm=self.options["order_max_norm"],
                                         interaction_order=self.options["interaction_order"],
                                         interaction_order_current=self.options["interaction_order"],
                                         options=self.options,
                                         domain=d,
                                         validation=None)

                # save original problem in gpc object
                megpc[i_qoi].gpc[d].problem_original = self.problem

                # save projection matrix in gPC object
                megpc[i_qoi].gpc[d].p_matrix = copy.deepcopy(p_matrix[d])
                megpc[i_qoi].gpc[d].p_matrix_norm = copy.deepcopy(p_matrix_norm[d])

                # extend initial grid and perform additional simulations if necessary
                n_coeffs = get_num_coeffs_sparse(order_dim_max=[self.options["order"][0] for _ in range(dim_reduced[d])],
                                                 order_glob_max=self.options["order_max"],
                                                 order_inter_max=self.options["interaction_order"],
                                                 order_inter_current=self.options["interaction_order"],
                                                 dim=dim_reduced[d])

                n_grid_new[d] = n_coeffs * self.options["matrix_ratio"]

            # run additional simulations if n_grid_gradient was not sufficient for the selected order
            if np.max(n_grid_new) > grid.n_grid:
                iprint("Extending grid from {} to {} grid points ...".format(grid.n_grid, np.max(n_grid_new)),
                       tab=0, verbose=self.options["verbose"])

                # extend grid
                grid.extend_random_grid(n_grid_new=np.max(n_grid_new), seed=self.options["seed"])

                # Run simulations
                iprint("Performing {} additional simulations!".format(grid.coords.shape[0]),
                       tab=0, verbose=self.options["verbose"])

                start_time = time.time()

                res_new = com.run(model=self.problem.model,
                                  problem=self.problem,
                                  coords=grid.coords[self.options["n_grid_gradient"]:, :],
                                  coords_norm=grid.coords_norm[self.options["n_grid_gradient"]:, :],
                                  i_iter=self.options["order_max"],
                                  i_subiter=self.options["interaction_order"],
                                  fn_results=None,
                                  print_func_time=self.options["print_func_time"])

                res_all = np.vstack((res_all, res_new))

                iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
                       tab=0, verbose=self.options["verbose"])

                # Determine gradient [n_grid x n_out x dim]
                if self.options["gradient_enhanced"]:
                    start_time = time.time()
                    grad_res_3D_all = self.get_gradient(grid=grid,
                                                        results=res,
                                                        com=com,
                                                        gradient_results=grad_res_3D_all)
                    iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                           tab=0, verbose=self.options["verbose"])

                # crop results to considered qoi
                if self.options["qoi"] != "all":
                    res = copy.deepcopy(res_all)
                    grad_res_3D = copy.deepcopy(grad_res_3D_all)

                else:
                    res = res_all[:, q_idx][:, np.newaxis]
                    grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

                # update classifier with new grid-points
                megpc[i_qoi].init_classifier(coords=grid.coords_norm,
                                             results=np.vstack((res_all, res_new))[:, q_idx][:, np.newaxis],
                                             algorithm=self.options["classifier"],
                                             options=self.options["classifier_options"])

            # copy grid to gPC object and initialize transformed grid
            megpc[i_qoi].grid = copy.deepcopy(grid)

            # copy options to MEGPC object
            megpc[i_qoi].options = copy.deepcopy(self.options)

            # assign grids to sub-gPCs (rotate sub-grids in case of projection)
            megpc[i_qoi].assign_grids()

            # Initialize gpc matrices
            megpc[i_qoi].init_gpc_matrices()

            # Someone might not use the gradient to determine the gpc coeffs
            if megpc[i_qoi].gradient:
                grad_res_3D_passed = grad_res_3D
            else:
                grad_res_3D_passed = None

            # Compute gpc coefficients
            coeffs[i_qoi] = megpc[i_qoi].solve(results=res,
                                               gradient_results=grad_res_3D_passed,
                                               solver=self.options["solver"],
                                               settings=self.options["settings"],
                                               verbose=True)

            # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
            if self.options["error_type"] == "nrmsd" and megpc[0].validation is None:
                megpc[0].create_validation_set(n_samples=self.options["n_samples_validation"],
                                               n_cpu=self.options["n_cpu"])
            elif self.options["error_type"] == "nrmsd" and megpc[0].validation is not None:
                megpc[i_qoi].validation = copy.deepcopy(megpc[0].validation)

            eps = megpc[i_qoi].validate(coeffs=coeffs[i_qoi], results=res, gradient_results=grad_res_3D_passed)

            iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                self.options["error_type"],
                                                eps), tab=0, verbose=self.options["verbose"])

            # domain specific error
            eps_domain = [0 for _ in range(len(np.unique(megpc[i_qoi].domains)))]
            for i_gpc, d in enumerate(np.unique(megpc[i_qoi].domains)):
                eps_domain[d] = megpc[i_qoi].validate(coeffs=coeffs[i_qoi],
                                                      results=res,
                                                      domain=d,
                                                      output_idx=output_idx_passed_validation)
            # save data
            if self.options["fn_results"]:

                fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                write_gpc_pkl(megpc[i_qoi], fn_gpc_pkl_qoi)

                with h5py.File(fn_results + ".hdf5", "a") as f:

                    try:
                        fn_gpc_pkl = f["misc/fn_gpc_pkl"]
                        fn_gpc_pkl = np.vstack((fn_gpc_pkl, np.array([os.path.split(fn_gpc_pkl_qoi)[1]])))
                        del f["misc/fn_gpc_pkl"]
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=fn_gpc_pkl.astype("|S"))

                    except KeyError:
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=np.array([os.path.split(fn_gpc_pkl_qoi)[1]]).astype("|S"))

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("error" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=eps_domain[i_gpc],
                                         maxshape=None, dtype="float64")

                        f.create_dataset("coeffs" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=coeffs[i_qoi][i_gpc],
                                         maxshape=None, dtype="float64")

                    f.create_dataset("domains" + hdf5_subfolder,
                                     data=megpc[i_qoi].domains,
                                     maxshape=None, dtype="int64")

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("gpc_matrix" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=megpc[i_qoi].gpc[i_gpc].gpc_matrix,
                                         maxshape=None, dtype="float64")

                    if megpc[i_qoi].gpc[0].gpc_matrix_gradient is not None:
                        if self.options["gradient_enhanced"]:
                            for i_gpc in range(megpc[i_qoi].n_gpc):
                                f.create_dataset("gpc_matrix_gradient" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                                 data=megpc[i_qoi].gpc[i_gpc].gpc_matrix_gradient,
                                                 maxshape=None, dtype="float64")

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("p_matrix" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=megpc[i_qoi].gpc[i_gpc].p_matrix,
                                         maxshape=None, dtype="float64")

        if self.options["fn_results"]:

            with h5py.File(fn_results + ".hdf5", "a") as f:
                f.create_dataset("grid/coords", data=grid.coords,
                                 maxshape=None, dtype="float64")
                f.create_dataset("grid/coords_norm", data=grid.coords_norm,
                                 maxshape=None, dtype="float64")

                if megpc[i_qoi].grid.coords_gradient is not None:
                    f.create_dataset("grid/coords_gradient",
                                     data=grid.coords_gradient,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_gradient_norm",
                                     data=grid.coords_gradient_norm,
                                     maxshape=None, dtype="float64")

                f.create_dataset("model_evaluations/results", data=res,
                                 maxshape=None, dtype="float64")
                if grad_res_3D is not None:
                    f.create_dataset("model_evaluations/gradient_results", data=ten2mat(grad_res_3D),
                                     maxshape=None, dtype="float64")

                f.create_dataset("misc/error_type", data=self.options["error_type"])

                if megpc[0].validation is not None:
                    f.create_dataset("validation/results", data=megpc[0].validation.results,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords", data=megpc[0].validation.grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords_norm", data=megpc[0].validation.grid.coords_norm,
                                     maxshape=None, dtype="float64")

        com.close()

        if n_qoi == 1:
            return megpc[0], coeffs[0], res
        else:
            return megpc, coeffs, res


class RegAdaptive(Algorithm):
    """
    Adaptive regression approach based on leave one out cross validation error estimation
    """

    def __init__(self, problem, options, validation=None):
        """
        Parameters
        ----------
        problem: Problem class instance
            GPC problem under investigation
        options["order_start"] : int, optional, default=0
              Initial gPC expansion order (maximum order)
        options["order_end"] : int, optional, default=10
            Maximum Gpc expansion order to expand to (algorithm will terminate afterwards)
        options["interaction_order"]: int, optional, default=dim
            Define maximum interaction order of parameters (default: all interactions)
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["adaptive_sampling"] : boolean, optional, default: True
            Adds samples adaptively to the expansion until the error is converged and continues by
            adding new basis functions.

        Examples
        --------
        >>> import pygpc
        >>> # initialize adaptive gPC algorithm
        >>> algorithm = pygpc.RegAdaptive(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(RegAdaptive, self).__init__(problem=problem, options=options, validation=validation)

        # check contents of settings dict and set defaults
        if "order_start" not in self.options.keys():
            self.options["order_start"] = 0

        if "order_end" not in self.options.keys():
            self.options["order_end"] = 10

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "adaptive_sampling" not in self.options.keys():
            self.options["adaptive_sampling"] = True

    def run(self):
        """
        Runs adaptive gPC algorithm to solve problem.

        Returns
        -------
        gpc : GPC object instance
            GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """

        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        # initialize iterators
        eps = self.options["eps"] + 1.0
        i_grid = 0
        order = self.options["order_start"]
        first_iter = True
        grad_res_3D = None
        basis_order = np.array([self.options["order_start"],
                                min(self.options["interaction_order"], self.options["order_start"])])

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Initialize Reg gPC object
        gpc = Reg(problem=self.problem,
                  order=self.options["order_start"] * np.ones(self.problem.dim),
                  order_max=self.options["order_start"],
                  order_max_norm=self.options["order_max_norm"],
                  interaction_order=self.options["interaction_order"],
                  interaction_order_current=self.options["interaction_order"],
                  options=self.options,
                  validation=self.validation)
        extended_basis = True

        # Add a validation set if nrmsd is chosen and no validation set is yet present
        if self.options["error_type"] == "nrmsd" and not isinstance(self.validation, ValidationSet):
            gpc.create_validation_set(n_samples=self.options["n_samples_validation"],
                                      n_cpu=self.options["n_cpu"])

        # Initialize Grid object
        if self.options["solver"] == "Moore-Penrose":
            n_grid_init = np.ceil(self.options["matrix_ratio"] * gpc.basis.n_basis)
        else:
            n_grid_init = gpc.basis.n_basis

        gpc.grid = RandomGrid(parameters_random=self.problem.parameters_random,
                              options={"n_grid": n_grid_init, "seed": self.options["seed"]})

        gpc.solver = self.options["solver"]
        gpc.settings = self.options["settings"]
        gpc.options = copy.deepcopy(self.options)

        # Initialize gpc matrix
        gpc.init_gpc_matrix()
        gpc.n_grid.pop(0)
        gpc.n_basis.pop(0)

        if gpc.options["gradient_enhanced"]:
            gpc.grid.create_gradient_grid()

        # Main iterations (order)
        while eps > self.options["eps"]:

            if first_iter:
                basis_increment = 0
            else:
                basis_increment = 1

            # increase basis
            basis_order[0], basis_order[1] = increment_basis(order_current=basis_order[0],
                                                             interaction_order_current=basis_order[1],
                                                             interaction_order_max=self.options["interaction_order"],
                                                             incr=basis_increment)

            if basis_order[0] > self.options["order_end"]:
                break

            # update basis
            b_added = gpc.basis.set_basis_poly(order=basis_order[0] * np.ones(self.problem.dim),
                                               order_max=basis_order[0],
                                               order_max_norm=self.options["order_max_norm"],
                                               interaction_order=self.options["interaction_order"],
                                               interaction_order_current=basis_order[1],
                                               problem=gpc.problem)

            print_str = "Order/Interaction order: {}/{}".format(basis_order[0], basis_order[1])
            iprint(print_str, tab=0, verbose=self.options["verbose"])
            iprint("=" * len(print_str), tab=0, verbose=self.options["verbose"])

            if b_added is not None:
                extended_basis = True

            if self.options["adaptive_sampling"]:
                iprint("Starting adaptive sampling:", tab=0, verbose=self.options["verbose"])

            add_samples = True   # if adaptive sampling is False, the while loop will be only executed once
            delta_eps_target = 1e-1
            delta_eps = delta_eps_target + 1
            delta_samples = 5e-2

            while add_samples and delta_eps > delta_eps_target and eps > self.options["eps"]:

                if not self.options["adaptive_sampling"]:
                    add_samples = False

                # new sample size
                if extended_basis and self.options["adaptive_sampling"]:
                    # do not increase sample size immediately when basis was extended, try first with old samples
                    n_grid_new = gpc.grid.n_grid
                elif self.options["adaptive_sampling"] and not first_iter:
                    # increase sample size stepwise (adaptive sampling)
                    n_grid_new = int(np.ceil(gpc.grid.n_grid + delta_samples * gpc.basis.n_basis))
                else:
                    # increase sample size according to matrix ratio w.r.t. bnumber of basis functions
                    n_grid_new = int(np.ceil(gpc.basis.n_basis * self.options["matrix_ratio"]))

                # run model if grid points were added
                if i_grid < n_grid_new or extended_basis:
                    # extend grid
                    if i_grid < n_grid_new:
                        iprint("Extending grid from {} to {} by {} sampling points".format(
                            gpc.grid.n_grid, n_grid_new, n_grid_new - gpc.grid.n_grid),
                            tab=0, verbose=self.options["verbose"])

                        gpc.grid.extend_random_grid(n_grid_new=n_grid_new, seed=None)

                        # run simulations
                        iprint("Performing simulations " + str(i_grid + 1) + " to " + str(gpc.grid.coords.shape[0]),
                               tab=0, verbose=self.options["verbose"])

                        start_time = time.time()

                        res_new = com.run(model=gpc.problem.model,
                                          problem=gpc.problem,
                                          coords=gpc.grid.coords[int(i_grid):int(len(gpc.grid.coords))],
                                          coords_norm=gpc.grid.coords_norm[int(i_grid):int(len(gpc.grid.coords))],
                                          i_iter=order,
                                          i_subiter=gpc.interaction_order_current,
                                          fn_results=gpc.fn_results,
                                          print_func_time=self.options["print_func_time"])

                        iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
                               tab=0, verbose=self.options["verbose"])

                        # Append result to solution matrix (RHS)
                        if i_grid == 0:
                            res = res_new
                        else:
                            res = np.vstack([res, res_new])

                        if self.options["gradient_enhanced"]:
                            start_time = time.time()
                            grad_res_3D = self.get_gradient(grid=gpc.grid,
                                                            results=res,
                                                            com=com,
                                                            gradient_results=grad_res_3D)
                            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                                   tab=0, verbose=self.options["verbose"])

                        i_grid = gpc.grid.coords.shape[0]

                    # update gpc matrix
                    gpc.init_gpc_matrix()

                    # determine gpc coefficients
                    coeffs = gpc.solve(results=res,
                                       gradient_results=grad_res_3D,
                                       solver=gpc.solver,
                                       settings=gpc.settings,
                                       verbose=True)

                    # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
                    eps = gpc.validate(coeffs=coeffs,
                                       results=res,
                                       gradient_results=grad_res_3D)

                    if extended_basis:
                        eps_ref = copy.deepcopy(eps)
                    else:
                        delta_eps = np.abs((gpc.error[-1] - gpc.error[-2]) / eps_ref)

                    iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                        self.options["error_type"],
                                                        eps), tab=0, verbose=self.options["verbose"])

                    # extend basis further if error was decreased (except in very first iteration)
                    if not first_iter and extended_basis and gpc.error[-1] < gpc.error[-2]:
                        break

                    extended_basis = False
                    first_iter = False

                    # exit adaptive sampling loop if no adaptive sampling was chosen
                    if not self.options["adaptive_sampling"]:
                        break

            # save gpc object and coeffs for this sub-iteration
            if self.options["fn_results"]:

                fn_gpc_pkl = fn_results + '.pkl'
                write_gpc_pkl(gpc, fn_gpc_pkl)

                with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:

                    # overwrite coeffs
                    if "coeffs" in f.keys():
                        del f['coeffs']
                    f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

                    # Append gradient of results
                    if grad_res_3D is not None:
                        grad_res_2D = ten2mat(grad_res_3D)
                        if "gradient_results" in f["model_evaluations"].keys():
                            n_rows_old = f["model_evaluations/gradient_results"].shape[0]
                            f["model_evaluations/gradient_results"].resize(grad_res_2D.shape[0], axis=0)
                            f["model_evaluations/gradient_results"][n_rows_old:, :] = grad_res_2D[n_rows_old:, :]
                        else:
                            f.create_dataset("model_evaluations/gradient_results",
                                             (grad_res_2D.shape[0], grad_res_2D.shape[1]),
                                             maxshape=(None, None),
                                             dtype="float64",
                                             data=grad_res_2D)

                # save gpc matrix in .hdf5 file
                gpc.save_gpc_matrix_hdf5()

        # determine gpc coefficients
        coeffs = gpc.solve(results=res,
                           gradient_results=grad_res_3D,
                           solver=gpc.solver,
                           settings=gpc.settings,
                           verbose=True)

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:
            write_gpc_pkl(gpc, os.path.splitext(self.options["fn_results"])[0] + '.pkl')

            with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:
                if "coeffs" in f.keys():
                    del f['coeffs']
                f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

                # misc
                f.create_dataset("misc/fn_gpc_pkl", data=np.array([os.path.split(fn_gpc_pkl)[1]]).astype("|S"))
                f.create_dataset("misc/error_type", data=self.options["error_type"])
                f.create_dataset("error", data=eps, maxshape=None, dtype="float64")

                if gpc.validation is not None:
                    f.create_dataset("validation/results", data=gpc.validation.results,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords", data=gpc.validation.grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("validation/grid/coords_norm", data=gpc.validation.grid.coords_norm,
                                     maxshape=None, dtype="float64")

                if self.options["gradient_enhanced"]:
                    f.create_dataset("grid/coords_gradient", data=gpc.grid.coords_gradient,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_gradient_norm", data=gpc.grid.coords_gradient_norm,
                                     maxshape=None, dtype="float64")

        com.close()

        return gpc, coeffs, res


class MERegAdaptiveProjection(Algorithm):
    """
    Adaptive regression approach based on leave one out cross validation error estimation
    """

    def __init__(self, problem, options, validation=None):
        """
        Parameters
        ----------
        problem: Problem class instance
            GPC problem under investigation
        options["order_start"] : int, optional, default=0
              Initial gPC expansion order (maximum order)
        options["order_end"] : int, optional, default=10
            Maximum Gpc expansion order to expand to (algorithm will terminate afterwards)
        options["interaction_order"]: int, optional, default=dim
            Define maximum interaction order of parameters (default: all interactions)
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["adaptive_sampling"] : boolean, optional, default: True
            Adds samples adaptively to the expansion until the error is converged and continues by
            adding new basis functions.
        options["n_samples_discontinuity"] : int, optional, default: 10
            Number of grid points close to discontinuity to refine its location
        options["n_grid_init"] : int, optional, default: 10
            Number of initial simulations to explore the parameter space
        Examples
        --------
        >>> import pygpc
        >>> # initialize adaptive gPC algorithm
        >>> algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(MERegAdaptiveProjection, self).__init__(problem=problem, options=options, validation=validation)

        # check contents of settings dict and set defaults
        if "order_start" not in self.options.keys():
            self.options["order_start"] = 0

        if "order_end" not in self.options.keys():
            self.options["order_end"] = 10

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "adaptive_sampling" not in self.options.keys():
            self.options["adaptive_sampling"] = True

        if "n_samples_discontinuity" not in self.options.keys():
            self.options["n_samples_discontinuity"] = 10

        if "n_grid_init" not in self.options.keys():
            self.options["n_grid_init"] = 10

    def run(self):
        """
        Runs Multi-Element adaptive gPC algorithm to solve problem.

        Returns
        -------
        megpc : Multi-element GPC object instance
            MEGPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: list of ndarray of float [n_gpc][n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """

        fn_results = os.path.splitext(self.options["fn_results"])[0]
        problem_original = copy.deepcopy(self.problem)

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        # initialize iterators
        grad_res_3D = None
        grad_res_3D_all = None
        basis_increment = 0

        n_grid_init = self.options["n_grid_init"]

        # make initial random grid to determine number of output variables and to estimate projection
        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": n_grid_init})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run initial simulations to determine initial projection matrix
        iprint("Performing {} initial simulations!".format(grid.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res_all = com.run(model=self.problem.model,
                          problem=self.problem,
                          coords=grid.coords,
                          coords_norm=grid.coords_norm,
                          i_iter=self.options["order_start"],
                          i_subiter=self.options["interaction_order"],
                          fn_results=os.path.splitext(self.options["fn_results"])[0],  # + "_temp"
                          print_func_time=self.options["print_func_time"])

        i_grid = grid.n_grid

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        if self.options["qoi"] == "all":
            qoi_idx = np.arange(res_all.shape[1])
            n_qoi = len(qoi_idx)
            error = [None for _ in range(n_qoi)]
        else:
            qoi_idx = [self.options["qoi"]]
            n_qoi = 1
            error = [0]

        # Determine gradient [n_grid x n_out x dim]
        if self.options["gradient_enhanced"] or self.options["projection"]:
            start_time = time.time()
            grad_res_3D_all = self.get_gradient(grid=grid,
                                                results=res_all,
                                                com=com)
            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                   tab=0, verbose=self.options["verbose"])

        megpc = [0 for _ in range(n_qoi)]
        coeffs = [0 for _ in range(n_qoi)]

        for i_qoi, q_idx in enumerate(qoi_idx):
            print_str = "Determining gPC approximation for QOI #{}:".format(q_idx)
            iprint(print_str, tab=0, verbose=self.options["verbose"])
            iprint("=" * len(print_str), tab=0, verbose=self.options["verbose"])

            first_iter = True

            # crop results to considered qoi
            if self.options["qoi"] != "all":
                res = copy.deepcopy(res_all)
                grad_res_3D = copy.deepcopy(grad_res_3D_all)
                hdf5_subfolder = ""
                output_idx_passed_validation = None
                # the gPC is constructed for all QOI but only using info for projection etc of desired QOI
                # validation is done for all qoi

            else:
                res = res_all[:, q_idx][:, np.newaxis]
                hdf5_subfolder = "/qoi_" + str(q_idx)
                output_idx_passed_validation = q_idx

                if grad_res_3D_all is not None:
                    grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

            # Create MEGPC object
            megpc[i_qoi] = MEGPC(problem=self.problem,
                                 options=self.options,
                                 validation=self.validation)

            # Write grid in gpc object
            megpc[i_qoi].grid = copy.deepcopy(grid)

            # determine gpc domains
            iprint("Determining gPC domains ...", tab=0, verbose=self.options["verbose"])
            megpc[i_qoi].init_classifier(coords=megpc[i_qoi].grid.coords_norm,
                                         results=res_all[:, q_idx][:, np.newaxis],
                                         algorithm=self.options["classifier"],
                                         options=self.options["classifier_options"])

            error[i_qoi] = [[] for _ in range(len(np.unique(megpc[i_qoi].classifier.domains)))]
            p_matrix = [0 for _ in range(megpc[i_qoi].n_gpc)]
            p_matrix_norm = [0 for _ in range(megpc[i_qoi].n_gpc)]
            dim = [0 for _ in range(megpc[i_qoi].n_gpc)]
            parameters= [OrderedDict() for _ in range(megpc[i_qoi].n_gpc)]
            problem = [0 for _ in range(megpc[i_qoi].n_gpc)]
            basis_order = OrderedDict()
            n_grid_reinit = [0 for _ in range(megpc[i_qoi].n_gpc)]

            # determine initial projection and initialize sub-gPCs
            for d in np.unique(megpc[i_qoi].domains):

                if self.options["projection"]:
                    p_matrix[d] = determine_projection_matrix(
                        gradient_results=grad_res_3D_all[megpc[i_qoi].domains == d, :, :],
                        qoi_idx=q_idx,
                        lambda_eps=self.options["lambda_eps_gradient"])

                    p_matrix_norm[d] = np.sum(np.abs(p_matrix[d]), axis=1)
                    dim[d] = p_matrix[d].shape[0]

                    for i in range(dim[d]):
                        parameters[d]["n{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

                    problem[d] = Problem(model=self.problem.model, parameters=parameters[d])

                else:
                    p_matrix[d] = None
                    p_matrix_norm[d] = None
                    dim[d] = problem_original.dim
                    parameters[d] = problem_original.parameters_random
                    problem[d] = copy.deepcopy(problem_original)

                # Set up reduced gPC for this domain
                megpc[i_qoi].add_sub_gpc(problem=problem[d],
                                         order=[self.options["order_start"] for _ in range(dim[d])],
                                         order_max=self.options["order_start"],
                                         order_max_norm=self.options["order_max_norm"],
                                         interaction_order=self.options["interaction_order"],
                                         interaction_order_current=self.options["interaction_order"],
                                         options=self.options,
                                         domain=d,
                                         validation=None)

                # save original problem in gpc object
                megpc[i_qoi].gpc[d].problem_original = copy.deepcopy(problem_original)

                # save projection matrix in gPC object
                megpc[i_qoi].gpc[d].p_matrix = copy.deepcopy(p_matrix[d])
                megpc[i_qoi].gpc[d].p_matrix_norm = copy.deepcopy(p_matrix_norm[d])

                # initialize dict containing approximation orders of sub-gPCs [order, interaction_order_current]
                basis_order["poly_dom_{}".format(d)] = np.array([self.options["order_start"],
                                                                 self.options["interaction_order"]])

                # initialize solver settings
                megpc[i_qoi].gpc[d].solver = self.options["solver"]
                megpc[i_qoi].gpc[d].settings = self.options["settings"]

                # extend initial grid and perform additional simulations if necessary
                if not self.options["adaptive_sampling"] or megpc[i_qoi].gpc[d].solver == "Moore-Penrose":
                    n_coeffs = get_num_coeffs_sparse(
                        order_dim_max=[self.options["order_start"] for _ in range(dim[d])],
                        order_glob_max=self.options["order_start"],
                        order_inter_max=self.options["interaction_order"],
                        order_inter_current=self.options["interaction_order"],
                        dim=dim[d])

                    n_grid_reinit[d] = n_coeffs * self.options["matrix_ratio"]

                # Check if we have enough samples in this particular domain for the given order we start
                if n_grid_reinit[d] > np.sum(megpc[i_qoi].domains == d):

                    # extend random grid
                    grid.extend_random_grid(n_grid_new=grid.n_grid - np.sum(megpc[i_qoi].domains == d) + n_grid_reinit[d],
                                            seed=None,
                                            domain=d)

            if grid.n_grid > i_grid:

                # Run some more initial simulations
                iprint("Performing {} more initial simulations "
                       "to fulfil order constraint!".format(grid.n_grid - i_grid),
                       tab=0, verbose=self.options["verbose"])

                start_time = time.time()

                res_new = com.run(model=self.problem.model,
                                  problem=self.problem,
                                  coords=grid.coords[i_grid:, ],
                                  coords_norm=grid.coords_norm[i_grid:, ],
                                  i_iter=self.options["order_start"],
                                  i_subiter=self.options["interaction_order"],
                                  fn_results=os.path.splitext(self.options["fn_results"])[0],
                                  print_func_time=self.options["print_func_time"])

                # add results to results array
                res_all = np.vstack((res_all, res_new))
                i_grid = grid.n_grid

                iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
                       tab=0, verbose=self.options["verbose"])

                # Determine gradient [n_grid x n_out x dim]
                if self.options["gradient_enhanced"] or self.options["projection"]:
                    start_time = time.time()
                    grad_res_3D_all = self.get_gradient(grid=grid,
                                                        results=res_all,
                                                        com=com,
                                                        gradient_results=grad_res_3D_all)
                    iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                           tab=0, verbose=self.options["verbose"])

                megpc[i_qoi].grid = copy.deepcopy(grid)

            # create validation set if necessary
            if self.options["error_type"] == "nrmsd" and megpc[0].validation is None:
                iprint("Determining validation set of size {} "
                       "for NRMSD error calculation ...".format(int(self.options["n_samples_validation"])),
                       tab=0, verbose=self.options["verbose"])
                megpc[0].create_validation_set(n_samples=self.options["n_samples_validation"],
                                               n_cpu=self.options["n_cpu"],
                                               gradient=self.options["gradient_enhanced"])

                iprint("Saving validation set to {} ".format(self.options["fn_results"] + "_validation_set.hdf5"),
                       tab=0, verbose=self.options["verbose"])
                megpc[0].validation.write(fname=self.options["fn_results"] + "_validation.hdf5")

            elif self.options["error_type"] == "nrmsd" and megpc[0].validation is not None:
                megpc[i_qoi].validation = copy.deepcopy(megpc[0].validation)

            extended_basis = True

            # initialize domain specific error
            eps = np.array([self.options["eps"] + 1.0 for _ in range(megpc[i_qoi].n_gpc)])

            # Main iterations (order)
            while (eps > self.options["eps"]).any():

                stop_by_order = [(basis_order["poly_dom_{}".format(i)] == [self.options["order_end"],
                                                                           self.options["interaction_order"]]).all() for
                                 i in range(megpc[i_qoi].n_gpc)]
                stop_by_error = eps < self.options["eps"]

                if np.logical_or(stop_by_order, stop_by_error).all():
                    break

                iprint("Refining domain boundary ...", tab=0, verbose=self.options["verbose"])

                # determine grid points close to discontinuity
                coords_norm_disc = get_coords_discontinuity(classifier=megpc[i_qoi].classifier,
                                                            x_min=[-1 for _ in range(megpc[i_qoi].problem.dim)],
                                                            x_max=[+1 for _ in range(megpc[i_qoi].problem.dim)],
                                                            n_coords_disc=self.options["n_samples_discontinuity"],
                                                            border_sampling="structured")

                coords_disc = grid.get_denormalized_coordinates(coords_norm_disc)

                # add grid points close to discontinuity to global grid
                grid.extend_random_grid(coords=coords_disc,
                                        coords_norm=coords_norm_disc,
                                        gradient=self.options["gradient_enhanced"])

                # run simulations close to discontinuity
                iprint("Performing {} simulations to refine discontinuity location!".format(
                    self.options["n_samples_discontinuity"]), tab=0, verbose=self.options["verbose"])

                start_time = time.time()

                res_disc = com.run(model=self.problem.model,
                                   problem=self.problem,
                                   coords=coords_disc,
                                   coords_norm=coords_norm_disc,
                                   i_iter=None,
                                   i_subiter=None,
                                   fn_results=os.path.splitext(self.options["fn_results"])[0],  # + "_temp"
                                   print_func_time=self.options["print_func_time"])

                iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
                       tab=0, verbose=self.options["verbose"])

                # add results to results array
                res_all = np.vstack((res_all, res_disc))

                # Determine gradient [n_grid x n_out x dim]
                if self.options["gradient_enhanced"] or self.options["projection"]:
                    start_time = time.time()
                    grad_res_3D_all = self.get_gradient(grid=grid,
                                                        results=res_all,
                                                        com=com,
                                                        gradient_results=grad_res_3D_all)
                    iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                           tab=0, verbose=self.options["verbose"])

                i_grid = grid.n_grid

                # crop results to considered qoi
                if self.options["qoi"] != "all":
                    res = copy.deepcopy(res_all)
                    grad_res_3D = copy.deepcopy(grad_res_3D_all)

                else:
                    res = res_all[:, q_idx][:, np.newaxis]

                    if grad_res_3D_all is not None:
                        grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

                # Write grid in gpc object
                megpc[i_qoi].grid = copy.deepcopy(grid)

                # update classifier
                iprint("Updating classifier ...", tab=0, verbose=self.options["verbose"])
                megpc[i_qoi].update_classifier(coords=megpc[i_qoi].grid.coords_norm,
                                               results=res_all[:, q_idx][:, np.newaxis])

                # update sub-gPCs if number of domains changed
                if len(np.unique(megpc[i_qoi].domains)) > len(megpc[i_qoi].gpc):

                    iprint("New domains found! Updating number of sub-gPCs from {} to {} ".
                           format(len(megpc[i_qoi].gpc), len(np.unique(megpc[i_qoi].domains))),
                           tab=0, verbose=self.options["verbose"])

                    del megpc[i_qoi].gpc

                    basis_order.append([self.options["order_start"], self.options["interaction_order"]])
                    eps = np.hstack((eps, np.array(self.options["eps"] + 1)))

                    for i_gpc, d in enumerate(np.unique(megpc[i_qoi].domains)):
                        megpc[i_qoi].add_sub_gpc(problem=problem_original,
                                                 order=basis_order["poly_dom_{}".format(d)][0] * np.ones(
                                                     self.problem.dim),
                                                 order_max=self.options["order_start"],
                                                 order_max_norm=self.options["order_max_norm"],
                                                 interaction_order=self.options["interaction_order"],
                                                 interaction_order_current=basis_order["poly_dom_{}".format(d)][1],
                                                 options=self.options,
                                                 domain=d,
                                                 validation=None)

                        # save original problem in gpc object
                        megpc[i_qoi].gpc[d].problem_original = copy.deepcopy(problem_original)

                        # initialize domain specific interaction order and other settings
                        megpc[i_qoi].gpc[i_gpc].solver = self.options["solver"]
                        megpc[i_qoi].gpc[i_gpc].settings = self.options["settings"]

                # update projection matrices
                if self.options["projection"]:
                    p_matrix = [0 for _ in range(megpc[i_qoi].n_gpc)]
                    p_matrix_norm = [0 for _ in range(megpc[i_qoi].n_gpc)]
                    dim = [0 for _ in range(megpc[i_qoi].n_gpc)]
                    parameters = [OrderedDict() for _ in range(megpc[i_qoi].n_gpc)]
                    problem = [0 for _ in range(megpc[i_qoi].n_gpc)]

                    for d in np.unique(megpc[i_qoi].domains):

                        p_matrix[d] = determine_projection_matrix(
                            gradient_results=grad_res_3D_all[megpc[i_qoi].domains == d, :, :],
                            qoi_idx=q_idx,
                            lambda_eps=self.options["lambda_eps_gradient"])

                        p_matrix_norm[d] = np.sum(np.abs(p_matrix[d]), axis=1)
                        dim[d] = p_matrix[d].shape[0]

                        for i in range(dim[d]):
                            parameters[d]["n{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

                        problem[d] = Problem(model=self.problem.model, parameters=parameters[d])

                        # replace sub-gpc with the one containing the reduced problem
                        megpc[i_qoi].add_sub_gpc(problem=problem[d],
                                                 order=basis_order["poly_dom_{}".format(d)][0] * np.ones(dim[d]),
                                                 order_max=self.options["order_start"],
                                                 order_max_norm=self.options["order_max_norm"],
                                                 interaction_order=self.options["interaction_order"],
                                                 interaction_order_current=basis_order["poly_dom_{}".format(d)][1],
                                                 options=self.options,
                                                 domain=d,
                                                 validation=None)

                        # save original problem in gpc object
                        megpc[i_qoi].gpc[d].problem_original = copy.deepcopy(problem_original)

                        # save projection matrix in gPC object
                        megpc[i_qoi].gpc[d].p_matrix = copy.deepcopy(p_matrix[d])
                        megpc[i_qoi].gpc[d].p_matrix_norm = copy.deepcopy(p_matrix_norm[d])

                        # initialize domain specific interaction order and other settings
                        megpc[i_qoi].gpc[d].solver = self.options["solver"]
                        megpc[i_qoi].gpc[d].settings = self.options["settings"]

                # update gpc approximation with new grid points close to discontinuity
                # assign grids to sub-gPCs (rotate sub-grids in case of projection)
                megpc[i_qoi].assign_grids()

                # Initialize gpc matrices
                megpc[i_qoi].init_gpc_matrices()

                # Compute gpc coefficients
                if self.options["gradient_enhanced"]:
                    grad_res_3D_passed = grad_res_3D
                else:
                    grad_res_3D_passed = None

                coeffs[i_qoi] = megpc[i_qoi].solve(results=res,
                                                   gradient_results=grad_res_3D_passed,
                                                   solver=self.options["solver"],
                                                   settings=self.options["settings"],
                                                   verbose=True)

                # domain specific error
                for i_gpc, d in enumerate(np.unique(megpc[i_qoi].domains)):
                    eps[d] = megpc[i_qoi].validate(coeffs=coeffs[i_qoi],
                                                   results=res,
                                                   domain=d,
                                                   output_idx=output_idx_passed_validation)
                    error[i_qoi][d].append(eps[d])

                    iprint("-> Domain: {} {} {} "
                           "error = {}".format(d,
                                               self.options["error_norm"],
                                               self.options["error_type"],
                                               eps[d]), tab=0, verbose=self.options["verbose"])

                # loop over domains and increase order if necessary
                for i_gpc, d in enumerate(np.unique(megpc[i_qoi].domains)):

                    skip = (basis_order["poly_dom_{}".format(d)] ==
                            [self.options["order_end"], self.options["interaction_order"]]).all()

                    if (eps[d] > self.options["eps"]) and not skip:

                        # increase basis by 1 interaction order
                        order_new = increment_basis(order_current=basis_order["poly_dom_{}".format(d)][0],
                                                    interaction_order_current=basis_order["poly_dom_{}".format(d)][1],
                                                    interaction_order_max=self.options["interaction_order"],
                                                    incr=basis_increment)

                        basis_order["poly_dom_{}".format(d)][0] = order_new[0]
                        basis_order["poly_dom_{}".format(d)][1] = order_new[1]

                        print_str = "Domain: {}, Order: #{}, Sub-iteration: #{}".format(
                            d,
                            basis_order["poly_dom_{}".format(d)][0],
                            basis_order["poly_dom_{}".format(d)][1])

                        iprint(print_str, tab=0, verbose=self.options["verbose"])
                        iprint("=" * len(print_str), tab=0, verbose=self.options["verbose"])

                        # update basis
                        b_added = megpc[i_qoi].gpc[d].basis.set_basis_poly(
                            order=basis_order["poly_dom_{}".format(d)][0] * np.ones(dim[d]),
                            order_max=basis_order["poly_dom_{}".format(d)][0],
                            order_max_norm=self.options["order_max_norm"],
                            interaction_order=self.options["interaction_order"],
                            interaction_order_current=basis_order["poly_dom_{}".format(d)][1],
                            problem=problem[d])

                        # continue algorithm if no basis function was added because of max norm constraint
                        if b_added is not None:
                            extended_basis = True
                        else:
                            extended_basis = False
                            # iprint("-> Domain: {} {} {} "
                            #        "error = {}".format(d,
                            #                            self.options["error_norm"],
                            #                            self.options["error_type"],
                            #                            eps[d]), tab=0, verbose=self.options["verbose"])
                            # iprint("-> No basis functions to add in domain {} ... Continuing ... ".format(d),
                            #        tab=0, verbose=self.options["verbose"])
                            continue

                        # update gpc matrix
                        megpc[i_qoi].gpc[d].init_gpc_matrix()

                        # determine gpc coefficients with new basis but old samples
                        if self.options["gradient_enhanced"]:
                            grad_res_3D_passed = grad_res_3D[megpc[i_qoi].domains == d, :, :]
                        else:
                            grad_res_3D_passed = None

                        coeffs[i_qoi][d] = megpc[i_qoi].gpc[d].solve(results=res[megpc[i_qoi].domains == d, ],
                                                                     gradient_results=grad_res_3D_passed,
                                                                     solver=megpc[i_qoi].gpc[d].solver,
                                                                     settings=megpc[i_qoi].gpc[d].settings,
                                                                     verbose=True)

                        # Add samples
                        add_samples = True  # if adaptive sampling is False, the loop will be only executed once
                        delta_eps_target = 1e-1
                        delta_eps = delta_eps_target + 1
                        delta_samples = 4*5e-2

                        if self.options["adaptive_sampling"]:
                            iprint("Starting adaptive sampling:", tab=0, verbose=self.options["verbose"])

                        # only increase samples if error increased and until error converges again
                        while add_samples and delta_eps > delta_eps_target and eps[d] > self.options["eps"]:

                            if not self.options["adaptive_sampling"]:
                                add_samples = False

                            # new sample size
                            if extended_basis and self.options["adaptive_sampling"]:
                                # do not increase sample size immediately when basis was extended
                                # try first with old samples
                                n_grid_new = megpc[i_qoi].gpc[d].grid.n_grid
                            elif self.options["adaptive_sampling"] and not first_iter:
                                # increase sample size stepwise (adaptive sampling)
                                n_grid_new = int(np.ceil(megpc[i_qoi].gpc[d].grid.n_grid +
                                                         delta_samples * megpc[i_qoi].gpc[d].basis.n_basis))
                            else:
                                # increase sample size according to matrix ratio w.r.t. number of basis functions
                                n_grid_new = int(
                                    np.ceil(megpc[i_qoi].gpc[d].basis.n_basis * self.options["matrix_ratio"]))

                            # run model if grid points were added
                            if megpc[i_qoi].gpc[d].grid.n_grid < n_grid_new or extended_basis:
                                # extend grid
                                if megpc[i_qoi].gpc[d].grid.n_grid < n_grid_new:
                                    iprint("Extending grid in domain {} from {} to {} by {} sampling points "
                                           "(global grid: {})".format(d, megpc[i_qoi].gpc[d].grid.n_grid, n_grid_new,
                                                                      n_grid_new - megpc[i_qoi].gpc[d].grid.n_grid,
                                                                      megpc[i_qoi].grid.n_grid),
                                           tab=0, verbose=self.options["verbose"])

                                    # add grid points in this domain to global grid
                                    grid.extend_random_grid(
                                        n_grid_new=grid.n_grid - megpc[i_qoi].gpc[d].grid.n_grid + n_grid_new,
                                        seed=None,
                                        classifier=megpc[i_qoi].classifier,
                                        domain=d,
                                        gradient=self.options["gradient_enhanced"])

                                    # run simulations
                                    iprint("Performing simulations {} to {}".format(
                                        i_grid + 1, grid.coords.shape[0]),
                                        tab=0, verbose=self.options["verbose"])

                                    start_time = time.time()

                                    res_new = com.run(model=self.problem.model,
                                                      problem=self.problem,
                                                      coords=grid.coords[int(i_grid):, :],
                                                      coords_norm=grid.coords_norm[int(i_grid):, :],
                                                      i_iter=basis_order["poly_dom_{}".format(d)][0],
                                                      i_subiter=basis_order["poly_dom_{}".format(d)][1],
                                                      fn_results=os.path.splitext(self.options["fn_results"])[0],
                                                      print_func_time=self.options["print_func_time"])

                                    iprint('Total parallel function evaluation {} sec'.format(
                                        str(time.time() - start_time)),
                                        tab=0, verbose=self.options["verbose"])

                                    # append to results array containing all qoi
                                    res_all = np.vstack([res_all, res_new])

                                    if self.options["gradient_enhanced"] or self.options["projection"]:
                                        start_time = time.time()
                                        grad_res_3D_all = self.get_gradient(grid=grid,
                                                                            results=res_all,
                                                                            com=com,
                                                                            gradient_results=grad_res_3D_all)
                                        iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                                               tab=0, verbose=self.options["verbose"])

                                    # crop results to considered qoi
                                    if self.options["qoi"] != "all":
                                        res = copy.deepcopy(res_all)
                                        grad_res_3D = copy.deepcopy(grad_res_3D_all)

                                    else:
                                        res = res_all[:, q_idx][:, np.newaxis]

                                        if grad_res_3D_all is not None:
                                            grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

                                    i_grid = grid.coords.shape[0]

                                    # update classifier
                                    iprint("Updating classifier ...", tab=0, verbose=self.options["verbose"])
                                    megpc[i_qoi].update_classifier(coords=grid.coords_norm,
                                                                   results=res_all[:, q_idx][:, np.newaxis])

                                    # TODO: the number of sub-gpcs could change here :/
                                    # update projection matrices
                                    if self.options["projection"]:

                                        for dd in np.unique(megpc[i_qoi].domains):

                                            p_matrix[dd] = determine_projection_matrix(
                                                gradient_results=grad_res_3D_all[megpc[i_qoi].domains == dd, :, :],
                                                qoi_idx=q_idx,
                                                lambda_eps=self.options["lambda_eps_gradient"])

                                            p_matrix_norm[dd] = np.sum(np.abs(p_matrix[dd]), axis=1)
                                            dim[dd] = p_matrix[dd].shape[0]

                                            for i in range(dim[d]):
                                                parameters[d]["n{}".format(i)] = Beta(pdf_shape=[1., 1.],
                                                                                      pdf_limits=[-1., 1.])

                                            problem[d] = Problem(model=self.problem.model, parameters=parameters[d])

                                            # replace sub-gpc with the one containing the reduced problem
                                            megpc[i_qoi].add_sub_gpc(problem=problem[d],
                                                                     order=basis_order["poly_dom_{}".format(d)][
                                                                               0] * np.ones(dim[d]),
                                                                     order_max=self.options["order_start"],
                                                                     order_max_norm=self.options["order_max_norm"],
                                                                     interaction_order=self.options[
                                                                         "interaction_order"],
                                                                     interaction_order_current=
                                                                     basis_order["poly_dom_{}".format(d)][1],
                                                                     options=self.options,
                                                                     domain=d,
                                                                     validation=None)

                                            # save original problem in gpc object
                                            megpc[i_qoi].gpc[d].problem_original = copy.deepcopy(problem_original)

                                            # save projection matrix in gPC object
                                            megpc[i_qoi].gpc[d].p_matrix = copy.deepcopy(p_matrix[d])
                                            megpc[i_qoi].gpc[d].p_matrix_norm = copy.deepcopy(p_matrix_norm[d])

                                            # initialize domain specific interaction order and other settings
                                            megpc[i_qoi].gpc[d].solver = self.options["solver"]
                                            megpc[i_qoi].gpc[d].settings = self.options["settings"]

                                    # update and assign grids
                                    megpc[i_qoi].grid = copy.deepcopy(grid)

                                    # assign grids to sub-gPCs (rotate sub-grids in case of projection)
                                    megpc[i_qoi].assign_grids()

                                    # update gpc matrix
                                    megpc[i_qoi].gpc[d].init_gpc_matrix()

                                    # determine gpc coefficients
                                    if self.options["gradient_enhanced"]:
                                        grad_res_3D_passed = grad_res_3D[megpc[i_qoi].domains == d, :, :]
                                    else:
                                        grad_res_3D_passed = None

                                    coeffs[i_qoi][d] = megpc[i_qoi].gpc[d].solve(
                                        results=res[megpc[i_qoi].domains == d, ],
                                        gradient_results=grad_res_3D_passed,
                                        solver=megpc[i_qoi].gpc[d].solver,
                                        settings=megpc[i_qoi].gpc[d].settings,
                                        verbose=True)

                                # validate gpc approximation
                                eps[d] = megpc[i_qoi].validate(coeffs=coeffs[i_qoi],
                                                               results=res,
                                                               domain=d,
                                                               output_idx=output_idx_passed_validation)
                                error[i_qoi][d].append(eps[d])

                                if extended_basis or first_iter:
                                    eps_ref = copy.deepcopy(eps[d])
                                else:
                                    delta_eps = np.abs((error[i_qoi][d][-1] -
                                                        error[i_qoi][d][-2]) / eps_ref)

                                first_iter = False

                                iprint("-> Domain: {} {} {} "
                                       "error = {}".format(d,
                                                           self.options["error_norm"],
                                                           self.options["error_type"],
                                                           eps[d]), tab=0, verbose=self.options["verbose"])

                                # stop adaptive sampling and extend basis further if error
                                # was decreased (except in very first iteration)
                                if extended_basis and error[i_qoi][d][-1] < error[i_qoi][d][-2]:
                                    break

                                extended_basis = False

                                # exit adaptive sampling loop if no adaptive sampling was chosen
                                if not self.options["adaptive_sampling"]:
                                    break

                    # save gpc object and coeffs for this sub-iteration
                    if self.options["fn_results"]:

                        fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                        write_gpc_pkl(megpc[i_qoi], fn_gpc_pkl_qoi)

                        with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:

                            # overwrite coeffs
                            try:
                                del f["coeffs" + hdf5_subfolder + "/dom_" + str(d)]
                            except KeyError:
                                pass

                            f.create_dataset("coeffs" + hdf5_subfolder + "/dom_" + str(d),
                                             data=coeffs[i_qoi][d], maxshape=None, dtype="float64")

                            # overwrite domains
                            try:
                                del f["domains" + hdf5_subfolder]
                            except KeyError:
                                pass
                            f.create_dataset("domains" + hdf5_subfolder,
                                             data=megpc[i_qoi].domains, maxshape=None, dtype="int64")

                            # save gpc matrix
                            try:
                                del f["gpc_matrix" + hdf5_subfolder + "/dom_" + str(d)]
                            except KeyError:
                                pass
                            f.create_dataset("gpc_matrix" + hdf5_subfolder + "/dom_" + str(d),
                                             data=megpc[i_qoi].gpc[d].gpc_matrix,
                                             maxshape=None, dtype="float64")

                            if megpc[i_qoi].gpc[d].p_matrix is not None:
                                try:
                                    del f["p_matrix" + hdf5_subfolder + "/dom_" + str(d)]
                                except KeyError:
                                    pass
                                f.create_dataset("p_matrix" + hdf5_subfolder + "/dom_" + str(d),
                                                 data=megpc[i_qoi].gpc[d].p_matrix,
                                                 maxshape=None, dtype="float64")

                            # save gradient gpc matrix
                            if megpc[i_qoi].gpc[0].gpc_matrix_gradient is not None:
                                try:
                                    del f["gpc_matrix_gradient" + hdf5_subfolder + "/dom_" + str(d)]
                                except KeyError:
                                    pass
                                if self.options["gradient_enhanced"]:
                                    f.create_dataset("gpc_matrix_gradient" + hdf5_subfolder + "/dom_" + str(d),
                                                     data=megpc[i_qoi].gpc[d].gpc_matrix_gradient,
                                                     maxshape=None, dtype="float64")

                            # save results
                            try:
                                del f["model_evaluations/results"]
                            except KeyError:
                                pass

                            f.create_dataset("model_evaluations/results",
                                             (res_all.shape[0], res_all.shape[1]),
                                             maxshape=(None, None),
                                             dtype="float64",
                                             data=res_all)

                            # save gradient of results
                            if grad_res_3D is not None:

                                try:
                                    del f["model_evaluations/gradient_results"]
                                except KeyError:
                                    pass

                                grad_res_2D_all = ten2mat(grad_res_3D_all)
                                f.create_dataset("model_evaluations/gradient_results",
                                                 (grad_res_2D_all.shape[0], grad_res_2D_all.shape[1]),
                                                 maxshape=(None, None),
                                                 dtype="float64",
                                                 data=grad_res_2D_all)

                            try:
                                del f["error" + hdf5_subfolder + "/dom_" + str(d)]
                            except KeyError:
                                pass
                            f.create_dataset("error" + hdf5_subfolder + "/dom_" + str(d),
                                             data=eps[d],
                                             maxshape=None, dtype="float64")

                basis_increment = 1

            megpc[i_qoi].update_classifier(coords=megpc[i_qoi].grid.coords_norm,
                                           results=res_all[:, q_idx][:, np.newaxis])
            megpc[i_qoi].assign_grids()
            megpc[i_qoi].init_gpc_matrices()

            # determine gpc coefficients
            if self.options["gradient_enhanced"]:
                grad_res_3D_passed = grad_res_3D[:, q_idx, :][:, np.newaxis, :]
            else:
                grad_res_3D_passed = None

            # determine gpc coefficients
            coeffs[i_qoi] = megpc[i_qoi].solve(results=res,
                                               gradient_results=grad_res_3D_passed,
                                               solver=megpc[i_qoi].gpc[d].solver,
                                               settings=megpc[i_qoi].gpc[d].settings,
                                               verbose=True)

            # save gpc object and gpc coeffs
            if self.options["fn_results"]:
                write_gpc_pkl(megpc[i_qoi], fn_gpc_pkl_qoi)

                with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:

                    try:
                        fn_gpc_pkl = f["misc/fn_gpc_pkl"]
                        fn_gpc_pkl = np.vstack((fn_gpc_pkl, np.array([os.path.split(fn_gpc_pkl_qoi)[1]])))
                        del f["misc/fn_gpc_pkl"]
                        f.create_dataset("misc/fn_gpc_pkl", data=fn_gpc_pkl.astype("|S"))

                    except KeyError:
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=np.array([os.path.split(fn_gpc_pkl_qoi)[1]]).astype("|S"))

                    try:
                        del f["grid"]
                    except KeyError:
                        pass

                    f.create_dataset("grid/coords", data=grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_norm", data=grid.coords_norm,
                                     maxshape=None, dtype="float64")

                    if megpc[i_qoi].grid.coords_gradient is not None:
                        f.create_dataset("grid/coords_gradient",
                                         data=grid.coords_gradient,
                                         maxshape=None, dtype="float64")
                        f.create_dataset("grid/coords_gradient_norm",
                                         data=grid.coords_gradient_norm,
                                         maxshape=None, dtype="float64")

                    try:
                        del f["model_evaluations"]
                    except KeyError:
                        pass
                    f.create_dataset("model_evaluations/results", data=res_all,
                                     maxshape=None, dtype="float64")
                    if grad_res_3D is not None:
                        f.create_dataset("model_evaluations/gradient_results", data=ten2mat(grad_res_3D),
                                         maxshape=None, dtype="float64")

                    f.create_dataset("misc/error_type", data=self.options["error_type"])

                    if megpc[0].validation is not None:
                        f.create_dataset("validation/results", data=megpc[0].validation.results,
                                         maxshape=None, dtype="float64")
                        f.create_dataset("validation/grid/coords", data=megpc[0].validation.grid.coords,
                                         maxshape=None, dtype="float64")
                        f.create_dataset("validation/grid/coords_norm", data=megpc[0].validation.grid.coords_norm,
                                         maxshape=None, dtype="float64")

                    try:
                        del f["coeffs"]
                    except KeyError:
                        pass

                    for i_gpc in range(megpc[i_qoi].n_gpc):
                        f.create_dataset("coeffs" + hdf5_subfolder + "/dom_" + str(i_gpc),
                                         data=coeffs[i_qoi][i_gpc],
                                         maxshape=None, dtype="float64")

        com.close()

        return megpc, coeffs, res_all


class RegAdaptiveProjection(Algorithm):
    """
    Adaptive regression approach using projection and leave one out cross validation error estimation
    """

    def __init__(self, problem, options, validation=None):
        """
        Parameters
        ----------
        problem: Problem class instance
            GPC problem under investigation
        options["order_start"] : int, optional, default=0
              Initial gPC expansion order (maximum order)
        options["order_end"] : int, optional, default=10
            Maximum Gpc expansion order to expand to (algorithm will terminate afterwards)
        options["interaction_order"]: int, optional, default=dim
            Define maximum interaction order of parameters (default: all interactions)
        options["order_max_norm"]: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        options["n_grid_gradient"] : float, optional, default: 10
            Number of initial grid points to determine gradient and projection matrix. When the algorithm goes
            into the main interations the number will be increased depending on the options "matrix_ratio"
            and "adaptive_sampling".
        options["qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["adaptive_sampling"] : boolean, optional, default: True
            Adds samples adaptively to the expansion until the error is converged and continues by
            adding new basis functions.

        Examples
        --------
        >>> import pygpc
        >>> # initialize adaptive gPC algorithm
        >>> algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(RegAdaptiveProjection, self).__init__(problem=problem, options=options, validation=validation)

        # check contents of settings dict and set defaults
        if "order_start" not in self.options.keys():
            self.options["order_start"] = 0

        if "order_end" not in self.options.keys():
            self.options["order_end"] = 10

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = problem.dim

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "n_grid_gradient" not in self.options.keys():
            self.options["n_grid_gradient"] = 10

        if "qoi" not in self.options.keys():
            self.options["qoi"] = 0

        if "adaptive_sampling" not in self.options.keys():
            self.options["adaptive_sampling"] = True

    def run(self):
        """
        Runs adaptive gPC algorithm using projection to solve problem.

        Returns
        -------
        gpc : GPC object instance
            GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """
        fn_results = os.path.splitext(self.options["fn_results"])[0]

        if os.path.exists(fn_results + ".hdf5"):
            os.remove(fn_results + ".hdf5")

        if os.path.exists(fn_results + "_temp.hdf5"):
            os.remove(fn_results + "_temp.hdf5")

        grad_res_3D = None

        # initialize iterators
        eps = self.options["eps"] + 1.0
        order = self.options["order_start"]
        error = []
        nrmsd = []
        loocv = []

        # make initial random grid to determine gradients and projection matrix
        grid_original = RandomGrid(parameters_random=self.problem.parameters_random,
                                   options={"n_grid": self.options["n_grid_gradient"]})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu, matlab_model=self.options["matlab_model"])

        # Run initial simulations to determine initial projection matrix
        iprint("Performing {} simulations!".format(grid_original.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        res_all = com.run(model=self.problem.model,
                          problem=self.problem,
                          coords=grid_original.coords,
                          coords_norm=grid_original.coords_norm,
                          i_iter=self.options["order_start"],
                          i_subiter=self.options["interaction_order"],
                          fn_results=os.path.splitext(self.options["fn_results"])[0], #  + "_temp"
                          print_func_time=self.options["print_func_time"])

        i_grid = grid_original.n_grid

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine gradient
        start_time = time.time()
        grad_res_3D_all = self.get_gradient(grid=grid_original,
                                            results=res_all,
                                            com=com,
                                            gradient_results=None)
        iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # set qoi indices
        if self.options["qoi"] == "all":
            qoi_idx = np.arange(res_all.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["qoi"]]
            n_qoi = 1

        # init variables
        self.problem_reduced = [None for _ in range(n_qoi)]
        gpc = [None for _ in range(n_qoi)]
        coeffs = [None for _ in range(n_qoi)]
        grid = [None for _ in range(n_qoi)]
        self.options["order_max"] = None

        # loop over qoi (projection is qoi specific)
        for i_qoi, q_idx in enumerate(qoi_idx):

            basis_order = np.array([self.options["order_start"],
                                    min(self.options["interaction_order"], self.options["order_start"])])

            if self.options["qoi"] == "all":
                qoi_idx_validate = q_idx
            else:
                qoi_idx_validate = np.arange(res_all.shape[1])

            first_iter = True

            # crop results to considered qoi
            if self.options["qoi"] != "all":
                res = copy.deepcopy(res_all)
                grad_res_3D = copy.deepcopy(grad_res_3D_all)
                hdf5_subfolder = ""

            else:
                res = res_all[:, q_idx][:, np.newaxis]
                grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]
                hdf5_subfolder = "/qoi_" + str(q_idx)

            # copy results of initial simulation
            # shutil.copy2(os.path.splitext(self.options["fn_results"])[0] + "_temp.hdf5", fn_results + ".hdf5")

            # Set up initial reduced problem
            # Determine projection matrix
            p_matrix = determine_projection_matrix(gradient_results=grad_res_3D_all,
                                                   qoi_idx=q_idx,
                                                   lambda_eps=self.options["lambda_eps_gradient"])
            p_matrix_norm = np.sum(np.abs(p_matrix), axis=1)

            # Set up initial reduced problem
            dim_reduced = p_matrix.shape[0]
            parameters_reduced = OrderedDict()

            for i in range(dim_reduced):
                parameters_reduced["n{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

            self.problem_reduced[i_qoi] = Problem(model=self.problem.model, parameters=parameters_reduced)

            # Create initial reduced gPC object
            gpc[i_qoi] = Reg(problem=self.problem_reduced[i_qoi],
                             order=[self.options["order_start"] for _ in range(dim_reduced)],
                             order_max=self.options["order_start"],
                             order_max_norm=self.options["order_max_norm"],
                             interaction_order=self.options["interaction_order"],
                             interaction_order_current=self.options["interaction_order"],
                             options=self.options,
                             validation=self.validation)

            # save original problem in gpc object
            gpc[i_qoi].problem_original = self.problem

            extended_basis = False

            # save projection matrix in gPC object
            gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)
            gpc[i_qoi].p_matrix_norm = copy.deepcopy(p_matrix_norm)

            # transformed grid
            grid[i_qoi] = copy.deepcopy(grid_original)

            # transform variables of original grid to reduced parameter space
            grid[i_qoi].coords = np.dot(grid_original.coords,
                                        gpc[i_qoi].p_matrix.transpose())
            grid[i_qoi].coords_norm = np.dot(grid_original.coords_norm,
                                             gpc[i_qoi].p_matrix.transpose() /
                                             gpc[i_qoi].p_matrix_norm[np.newaxis, :])

            # assign transformed grid
            gpc[i_qoi].grid = copy.deepcopy(grid[i_qoi])

            # Initialize gpc matrix
            gpc[i_qoi].init_gpc_matrix()
            gpc[i_qoi].n_grid.pop(0)
            gpc[i_qoi].n_basis.pop(0)

            gpc[i_qoi].solver = self.options["solver"]
            gpc[i_qoi].settings = self.options["settings"]

            # Main iterations (order)
            while eps > self.options["eps"]:

                if first_iter:
                    basis_increment = 0
                else:
                    basis_increment = 1

                # increase basis
                basis_order[0], basis_order[1] = increment_basis(order_current=basis_order[0],
                                                                 interaction_order_current=basis_order[1],
                                                                 interaction_order_max=self.options[
                                                                     "interaction_order"],
                                                                 incr=basis_increment)

                if basis_order[0] > self.options["order_end"]:
                    break

                # update basis
                b_added = gpc[i_qoi].basis.set_basis_poly(order=basis_order[0] * np.ones(self.problem_reduced[i_qoi].dim),
                                                          order_max=basis_order[0],
                                                          order_max_norm=self.options["order_max_norm"],
                                                          interaction_order=self.options["interaction_order"],
                                                          interaction_order_current=basis_order[1],
                                                          problem=self.problem_reduced[i_qoi])

                print_str = "Order/Interaction order: {}/{}".format(basis_order[0], basis_order[1])
                iprint(print_str, tab=0, verbose=self.options["verbose"])
                iprint("=" * len(print_str), tab=0, verbose=self.options["verbose"])

                if b_added is not None:
                    extended_basis = True

                if self.options["adaptive_sampling"]:
                    iprint("Starting adaptive sampling:", tab=0, verbose=self.options["verbose"])

                add_samples = True  # if adaptive sampling is False, the while loop will be only executed once
                delta_eps_target = 1e-1
                delta_eps = delta_eps_target + 1
                delta_samples = 5e-2

                if gpc[i_qoi].error:
                    eps_ref = gpc[i_qoi].error[-1]

                while add_samples and delta_eps > delta_eps_target and eps > self.options["eps"]:

                    if not self.options["adaptive_sampling"]:
                        add_samples = False

                    # new sample size
                    if extended_basis and self.options["adaptive_sampling"]:
                        # don't increase sample size immediately when basis was extended, try first with old samples
                        n_grid_new = gpc[i_qoi].grid.n_grid
                    elif self.options["adaptive_sampling"]:
                        # increase sample size stepwise (adaptive sampling)
                        n_grid_new = int(np.ceil(gpc[i_qoi].grid.n_grid + delta_samples * gpc[i_qoi].basis.n_basis))
                    else:
                        # increase sample size according to matrix ratio w.r.t. number of basis functions
                        n_grid_new = int(np.ceil(gpc[i_qoi].basis.n_basis * self.options["matrix_ratio"]))

                    # run model and update projection matrix if grid points were added
                    # (Skip simulations of first run because we already simulated it)
                    if i_grid < n_grid_new or extended_basis:
                        # extend grid
                        if i_grid < n_grid_new:
                            iprint("Extending grid from {} to {} by {} sampling points".format(
                                gpc[i_qoi].grid.n_grid, n_grid_new, n_grid_new - gpc[i_qoi].grid.n_grid),
                                tab=0, verbose=self.options["verbose"])
                            grid_original.extend_random_grid(n_grid_new=n_grid_new, seed=None)

                            # run simulations
                            iprint("Performing simulations " + str(i_grid + 1) + " to " + str(grid_original.coords.shape[0]),
                                   tab=0, verbose=self.options["verbose"])

                            start_time = time.time()
                            res_new = com.run(model=self.problem.model,
                                              problem=self.problem,
                                              coords=grid_original.coords[i_grid:grid_original.coords.shape[0]],
                                              coords_norm=grid_original.coords_norm[i_grid:grid_original.coords.shape[0]],
                                              i_iter=basis_order[0],
                                              i_subiter=basis_order[1],
                                              fn_results=gpc[i_qoi].fn_results,
                                              print_func_time=self.options["print_func_time"])

                            res_all = np.vstack((res_all, res_new))

                            iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
                                   tab=0, verbose=self.options["verbose"])

                            i_grid = grid_original.coords.shape[0]

                            # Determine gradient
                            start_time = time.time()
                            grad_res_3D_all = self.get_gradient(grid=grid_original,
                                                                results=res_all,
                                                                com=com,
                                                                gradient_results=grad_res_3D_all)
                            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                                   tab=0, verbose=self.options["verbose"])

                            # crop results to considered qoi
                            if self.options["qoi"] != "all":
                                res = copy.deepcopy(res_all)
                                grad_res_3D = copy.deepcopy(grad_res_3D_all)

                            else:
                                res = res_all[:, q_idx][:, np.newaxis]
                                grad_res_3D = grad_res_3D_all[:, q_idx, :][:, np.newaxis, :]

                            # Determine projection matrix
                            p_matrix = determine_projection_matrix(gradient_results=grad_res_3D_all,
                                                                   qoi_idx=q_idx,
                                                                   lambda_eps=self.options["lambda_eps_gradient"])
                            p_matrix_norm = np.sum(np.abs(p_matrix), axis=1)

                            # save projection matrix in gPC object
                            gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)
                            gpc[i_qoi].p_matrix_norm = copy.deepcopy(p_matrix_norm)

                            # Set up reduced gPC
                            dim_reduced = p_matrix.shape[0]
                            iprint("Dimension of reduced problem: {}".format(dim_reduced),
                                   tab=0, verbose=self.options["verbose"])

                            # Update gPC object if dimension has changed
                            if dim_reduced != gpc[i_qoi].problem.dim:
                                parameters_reduced = OrderedDict()

                                for i in range(dim_reduced):
                                    parameters_reduced["n{}".format(i)] = Beta(pdf_shape=[1., 1.],
                                                                               pdf_limits=[-1., 1.])

                                self.problem_reduced[i_qoi] = Problem(model=self.problem.model,
                                                                      parameters=parameters_reduced)

                                # Create reduced gPC object of order - 1 and add rest of basisfunctions
                                # of this subiteration afterwards
                                gpc[i_qoi] = Reg(problem=self.problem_reduced[i_qoi],
                                                 order=basis_order[0] * np.ones(self.problem_reduced[i_qoi].dim),
                                                 order_max=basis_order[0],
                                                 order_max_norm=self.options["order_max_norm"],
                                                 interaction_order=self.options["interaction_order"],
                                                 interaction_order_current=basis_order[1],
                                                 options=self.options,
                                                 validation=self.validation)

                                # save projection matrix in gPC object
                                gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)
                                gpc[i_qoi].p_matrix_norm = copy.deepcopy(p_matrix_norm)

                                # transformed grid
                                grid[i_qoi] = copy.deepcopy(grid_original)

                                # transform variables of original grid to reduced parameter space
                                grid[i_qoi].coords = np.dot(grid_original.coords,
                                                            gpc[i_qoi].p_matrix.transpose())
                                grid[i_qoi].coords_norm = np.dot(grid_original.coords_norm,
                                                                 gpc[i_qoi].p_matrix.transpose() /
                                                                 gpc[i_qoi].p_matrix_norm[np.newaxis, :])

                                # assign transformed grid
                                gpc[i_qoi].grid = copy.deepcopy(grid[i_qoi])

                                # Save settings and options in gpc object
                                gpc[i_qoi].interaction_order_current = interaction_order_current
                                gpc[i_qoi].solver = self.options["solver"]
                                gpc[i_qoi].settings = self.options["settings"]
                                gpc[i_qoi].options = copy.deepcopy(self.options)
                                gpc[i_qoi].error = error
                                gpc[i_qoi].relative_error_nrmsd = nrmsd
                                gpc[i_qoi].relative_error_loocv = loocv

                            else:
                                gpc[i_qoi].grid.coords = np.dot(grid_original.coords,
                                                                gpc[i_qoi].p_matrix.transpose())
                                gpc[i_qoi].grid.coords_norm = np.dot(grid_original.coords_norm,
                                                                     gpc[i_qoi].p_matrix.transpose() /
                                                                     gpc[i_qoi].p_matrix_norm[np.newaxis, :])

                    gpc[i_qoi].init_gpc_matrix()

                    # Someone might not use the gradient to determine the gpc coeffs
                    if gpc[i_qoi].gradient:
                        grad_res_3D_passed = grad_res_3D
                    else:
                        grad_res_3D_passed = None

                    # determine gpc coefficients
                    coeffs[i_qoi] = gpc[i_qoi].solve(results=res,
                                                     gradient_results=grad_res_3D_passed,
                                                     solver=gpc[i_qoi].solver,
                                                     settings=gpc[i_qoi].settings,
                                                     verbose=True)

                    # Add a validation set if nrmsd is chosen and no validation set is yet present
                    if self.options["error_type"] == "nrmsd" and not isinstance(gpc[0].validation, ValidationSet):
                        gpc[0].create_validation_set(n_samples=self.options["n_samples_validation"],
                                                     n_cpu=self.options["n_cpu"])

                    elif self.options["error_type"] == "nrmsd" and isinstance(gpc[0].validation, ValidationSet):
                        gpc[i_qoi].validation = copy.deepcopy(gpc[0].validation)

                    # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
                    eps = gpc[i_qoi].validate(coeffs=coeffs[i_qoi],
                                              results=res,
                                              gradient_results=grad_res_3D_passed,
                                              qoi_idx=qoi_idx_validate)

                    # save error in case that dimension has changed and the gpc object had to be reinitialized
                    error = copy.deepcopy(gpc[i_qoi].error)
                    nrmsd = copy.deepcopy(gpc[i_qoi].relative_error_nrmsd)
                    loocv = copy.deepcopy(gpc[i_qoi].relative_error_loocv)

                    if extended_basis or first_iter:
                        eps_ref = copy.deepcopy(eps)
                    else:
                        delta_eps = np.abs((gpc[i_qoi].error[-1] - gpc[i_qoi].error[-2]) / eps_ref)

                    first_iter = False

                    iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                        self.options["error_type"],
                                                        eps), tab=0, verbose=self.options["verbose"])

                    # extend basis further if error was decreased (except in very first iteration)
                    if order != self.options["order_start"]:
                        if extended_basis and gpc[i_qoi].error[-1] < gpc[i_qoi].error[-2]:
                            break

                    extended_basis = False

                    # exit adaptive sampling loop if no adaptive sampling was chosen
                    if not self.options["adaptive_sampling"]:
                        break

                # save gpc object and coeffs for this sub-iteration
                if self.options["fn_results"]:

                    fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                    write_gpc_pkl(gpc[i_qoi], fn_gpc_pkl_qoi)

                    with h5py.File(os.path.splitext(fn_results)[0] + ".hdf5", "a") as f:
                        # overwrite coeffs
                        try:
                            del f["coeffs" + hdf5_subfolder]
                        except KeyError:
                            pass

                        f.create_dataset("coeffs" + hdf5_subfolder,
                                         data=coeffs[i_qoi], maxshape=None, dtype="float64")

                        # save projection matrix
                        try:
                            del f["p_matrix" + hdf5_subfolder]
                        except KeyError:
                            f.create_dataset("p_matrix" + hdf5_subfolder,
                                             data=p_matrix, maxshape=None, dtype="float64")

                        # Append gradient of results (not projected)
                        grad_res_2D = ten2mat(grad_res_3D)

                        if "gradient_results" in f["model_evaluations"].keys():
                            n_rows_old = f["model_evaluations/gradient_results"].shape[0]
                            f["model_evaluations/gradient_results"].resize(grad_res_2D.shape[0], axis=0)
                            f["model_evaluations/gradient_results"][n_rows_old:, :] = grad_res_2D[n_rows_old:, :]
                        else:
                            f.create_dataset("model_evaluations/gradient_results",
                                             (grad_res_2D.shape[0], grad_res_2D.shape[1]),
                                             maxshape=(None, None),
                                             dtype="float64",
                                             data=grad_res_2D)

                        try:
                            del f["gpc_matrix" + hdf5_subfolder]
                        except KeyError:
                            pass
                        f.create_dataset("gpc_matrix" + hdf5_subfolder,
                                         data=gpc[i_qoi].gpc_matrix,
                                         maxshape=None, dtype="float64")

                        if gpc[i_qoi].gpc_matrix_gradient is not None:
                            try:
                                del f["gpc_matrix_gradient" + hdf5_subfolder]
                            except KeyError:
                                pass
                            f.create_dataset("gpc_matrix_gradient" + hdf5_subfolder,
                                             data=gpc[i_qoi].gpc_matrix_gradient,
                                             maxshape=None, dtype="float64")

                        try:
                            del f["error" + hdf5_subfolder]
                        except KeyError:
                            pass
                        f.create_dataset("error" + hdf5_subfolder,
                                         data=eps,
                                         maxshape=None, dtype="float64")

            # determine gpc coefficients
            coeffs[i_qoi] = gpc[i_qoi].solve(results=res,
                                             gradient_results=grad_res_3D_passed,
                                             solver=gpc[i_qoi].solver,
                                             settings=gpc[i_qoi].settings,
                                             verbose=True)

            # save gpc object gpc coeffs and projection matrix
            if self.options["fn_results"]:

                fn_gpc_pkl_qoi = fn_results + "_qoi_" + str(q_idx) + '.pkl'
                write_gpc_pkl(gpc[i_qoi], fn_gpc_pkl_qoi)

                with h5py.File(fn_results + ".hdf5", "a") as f:

                    try:
                        fn_gpc_pkl = f["misc/fn_gpc_pkl"][:]
                        fn_gpc_pkl = np.vstack((fn_gpc_pkl, np.array([os.path.split(fn_gpc_pkl_qoi)[1]])))
                        del f["misc/fn_gpc_pkl"]
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=fn_gpc_pkl.astype("|S"))

                    except KeyError:
                        f.create_dataset("misc/fn_gpc_pkl",
                                         data=np.array([os.path.split(fn_gpc_pkl_qoi)[1]]).astype("|S"))

                    try:
                        del f["coeffs" + hdf5_subfolder]
                    except KeyError:
                        pass
                    f.create_dataset("coeffs" + hdf5_subfolder, data=coeffs[i_qoi], maxshape=None, dtype="float64")

                    try:
                        del f["p_matrix" + hdf5_subfolder]
                    except KeyError:
                        pass
                    f.create_dataset("p_matrix" + hdf5_subfolder, data=p_matrix, maxshape=None, dtype="float64")

                    f.create_dataset("misc/error_type", data=self.options["error_type"])

                    if self.options["gradient_enhanced"]:
                        f.create_dataset("grid/coords_gradient", data=gpc[-1].grid.coords_gradient,
                                         maxshape=None, dtype="float64")
                        f.create_dataset("grid/coords_gradient_norm", data=gpc[-1].grid.coords_gradient_norm,
                                         maxshape=None, dtype="float64")

            # reset iterators
            eps = self.options["eps"] + 1.0
            order = self.options["order_start"]
            error = []
            nrmsd = []
            loocv = []

        with h5py.File(fn_results + ".hdf5", "a") as f:
            if gpc[0].validation is not None:
                f.create_dataset("validation/results", data=gpc[0].validation.results,
                                 maxshape=None, dtype="float64")
                f.create_dataset("validation/grid/coords", data=gpc[0].validation.grid.coords,
                                 maxshape=None, dtype="float64")
                f.create_dataset("validation/grid/coords_norm", data=gpc[0].validation.grid.coords_norm,
                                 maxshape=None, dtype="float64")
                f.create_dataset("misc/error_type", data=self.options["error_type"])

        # remove results of initial simulation
        # os.remove(os.path.splitext(self.options["fn_results"])[0] + "_temp.hdf5")

        com.close()

        if n_qoi == 1:
            return gpc[0], coeffs[0], res
        else:
            return gpc, coeffs, res
