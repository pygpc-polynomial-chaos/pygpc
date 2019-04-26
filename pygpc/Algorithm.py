# -*- coding: utf-8 -*-
import copy
import h5py
import os
import time

from .EGPC import *
from .Problem import *
from .SGPC import *
from .io import write_gpc_pkl
from .misc import determine_projection_matrix
from .misc import get_num_coeffs_sparse
from .Grid import *


class Algorithm(object):
    """
    Class for GPC algorithms
    """

    def __init__(self, problem, options, grid=None):
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
        """
        self.problem = problem
        self.problem_reduced = []
        self.options = options
        self.grid = grid
        self.grid_gradient = []

        # Generate results folder if it doesn't exist
        if not os.path.exists(os.path.split(self.options["fn_results"])[0]):
            os.makedirs(os.path.split(self.options["fn_results"])[0])

    def get_gradient(self, grid, results, gradient_results=None):
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

        if self.options["gradient_calculation"] == "standard_forward":
            # add new grid points for gradient calculation in grid.coords_gradient and grid.coords_gradient_norm
            grid.create_gradient_grid()

            # Initialize parallel Computation class
            com = Computation(n_cpu=self.n_cpu)

            results_gradient_tmp = com.run(model=self.problem.model,
                                           problem=self.problem,
                                           coords=np.vstack(grid.coords_gradient[n_gradient_results:, :, :].transpose(2, 0, 1)),
                                           coords_norm=np.vstack(grid.coords_gradient_norm[n_gradient_results:, :, :].transpose(2, 0, 1)),
                                           i_iter=self.options["order_max"],
                                           i_subiter=self.options["interaction_order"],
                                           fn_results=None,
                                           print_func_time=self.options["print_func_time"])

            # [n_grid x n_out x dim]
            results_gradient = np.zeros((grid.coords.shape[0]-n_gradient_results, results_gradient_tmp.shape[1], self.problem.dim))
            for i in range(self.problem.dim):
                results_gradient[:, :, i] = results_gradient_tmp[(i*(grid.coords.shape[0]-n_gradient_results)):(i+1)*(grid.coords.shape[0]-n_gradient_results), :]

            delta = np.repeat(np.linalg.norm(grid.coords_gradient_norm[n_gradient_results:, :, :] -
                                             np.repeat(grid.coords_norm[n_gradient_results:, :, np.newaxis], self.problem.dim, axis=2),
                                             axis=2)[:, :, np.newaxis].transpose(0, 2, 1),
                              results_gradient.shape[1], axis=1)

            # [n_grid x n_out x dim]
            gradient_results_new = (np.repeat(results[n_gradient_results:, :, np.newaxis], self.problem.dim, axis=2) - results_gradient) / delta

            if gradient_results is not None:
                gradient_results = np.vstack((gradient_results, gradient_results_new))
            else:
                gradient_results = gradient_results_new

        return gradient_results


class Static(Algorithm):
    """
    Static gPC algorithm
    """
    def __init__(self, problem, options, grid):
        """
        Constructor; Initializes static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["solver"]: str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
            - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
        options["settings"]: dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
            - 'NumInt' ... None
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
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["error_norm"] : str, optional, default="relative"
            Choose if error is determined "relative" or "absolute". Use "absolute" error when the
            model generates outputs equal to zero.
        options["error_type"] : str, optional, default="loocv"
            Choose type of error to validate gpc approximation. Use "loocv" (Leave-one-Out cross validation)
            to omit any additional calculations and "nrmsd" (normalized root mean square deviation) to compare
            against a Problem.ValidationSet.
        grid: Grid object instance
            Grid object to use for static gPC (RandomGrid, SparseGrid, TensorGrid)
        options["n_samples_validation"] : int, optional, default: 1e4
            Number of validation points used to determine the NRMSD if chosen as "error_type". Does not create a
            validation set if there is already one present in the Problem instance (problem.validation).

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
        super(Static, self).__init__(problem, options)
        self.grid = grid

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if self.options["method"] == "reg" and not (self.options["solver"] == "Moore-Penrose" or
                                                    self.options["solver"] == "OMP"):
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for 'reg' method")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["method"] == "quad":
            self.options["solver"] = 'NumInt'
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and ("settings" not in self.options.keys() or not(
                                                "n_coeffs_sparse" not in self.options["settings"].keys() or
                                                "sparsity" not in self.options["settings"].keys())):
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if self.options["solver"] == "LarsLasso" and ("settings" not in self.options.keys() or not(
                                                "alpha" not in self.options["settings"].keys())):
            self.options["settings"]["alpha"] = 1e-5

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "fn_results" not in self.options.keys():
            self.options["fn_results"] = None

        if "verbose" not in self.options.keys():
            self.options["verbose"] = True

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim

        if "n_cpu" in self.options.keys():
            self.n_cpu = self.options["n_cpu"]

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "error_norm" not in self.options.keys():
            self.options["error_norm"] = "relative"

        if "error_type" not in self.options.keys():
            self.options["error_type"] = "loocv"

        if "n_samples_validation" not in self.options.keys():
            self.options["n_samples_validation"] = 1e4

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
        # Create gPC object
        if self.options["method"] == "reg":
            gpc = Reg(problem=self.problem,
                      order=self.options["order"],
                      order_max=self.options["order_max"],
                      order_max_norm=self.options["order_max_norm"],
                      interaction_order=self.options["interaction_order"],
                      options=self.options)

        elif self.options["method"] == "quad":
            gpc = Quad(problem=self.problem,
                       order=self.options["order"],
                       order_max=self.options["order_max"],
                       order_max_norm=self.options["order_max_norm"],
                       interaction_order=self.options["interaction_order"],
                       options=self.options)

        else:
            raise AssertionError("Please specify correct gPC method ('reg' or 'quad')")

        # Write grid in gpc object
        gpc.grid = copy.deepcopy(self.grid)
        gpc.interaction_order_current = copy.deepcopy(self.options["interaction_order"])

        # Initialize gpc matrix
        gpc.init_gpc_matrix()

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu)

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
                      fn_results=gpc.fn_results,
                      print_func_time=self.options["print_func_time"])

        iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        com.close()

        # Compute gpc coefficients
        coeffs = gpc.solve(sim_results=res,
                           solver=self.options["solver"],
                           settings=self.options["settings"],
                           verbose=True)

        # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
        eps = gpc.validate(coeffs=coeffs, sim_results=res)

        iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                            self.options["error_type"],
                                            eps), tab=0, verbose=self.options["verbose"])

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:
            write_gpc_pkl(gpc, os.path.splitext(self.options["fn_results"])[0] + '.pkl')

            with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:
                if "coeffs" in f.keys():
                    del f['coeffs']
                f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

                if "gpc_matrix" in f.keys():
                    del f['gpc_matrix']
                f.create_dataset("gpc_matrix", data=gpc.gpc_matrix, maxshape=None, dtype="float64")

        return gpc, coeffs, res


class StaticProjection(Algorithm):
    """
    Static gPC algorithm using Basis Projection approach
    """
    def __init__(self, problem, options):
        """
        Constructor; Initializes static gPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options["method"]: str
            GPC method to apply ['Reg', 'Quad']
        options["solver"]: str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
            - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
        options["settings"]: dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
            - 'NumInt' ... None
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
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["error_norm"] : str, optional, default="relative"
            Choose if error is determined "relative" or "absolute". Use "absolute" error when the
            model generates outputs equal to zero.
        options["error_type"] : str, optional, default="loocv"
            Choose type of error to validate gpc approximation. Use "loocv" (Leave-one-Out cross validation)
            to omit any additional calculations and "nrmsd" (normalized root mean square deviation) to compare
            against a Problem.ValidationSet.
        options["projection_qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["gradient_calculation"] : str, optional, default="standard_forward"
            Type of the calculation scheme to determine the gradient in the grid points
            - "standard_forward" ... Forward approximation (creates additional dim*n_grid grid-points in the axis
            directions)
            - "???" ... ???
        options["n_grid_gradient"] : float, optional, default: 10
            Number of initial grid points to determine gradient and projection matrix
        options["matrix_ratio"]: float, optional, default=1.5
            Ration between the number of model evaluations and the number of basis functions.
            (>1 results in an overdetermined system)
        options["lambda_eps_gradient"] : float, optional, default: 0.95
            Bound of principal components in %. All eigenvectors are included until lambda_eps of total sum of all
            eigenvalues is included in the system.
        options["n_samples_validation"] : int, optional, default: 1e4
            Number of validation points used to determine the NRMSD if chosen as "error_type". Does not create a
            validation set if there is already one present in the Problem instance (problem.validation).

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
        super(StaticProjection, self).__init__(problem, options)

        # check contents of settings dict and set defaults
        if "method" not in self.options.keys():
            raise AssertionError("Please specify 'method' with either 'reg' or 'quad' in options dictionary")

        if self.options["method"] == "reg" and not (self.options["solver"] == "Moore-Penrose" or
                                                    self.options["solver"] == "OMP"):
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for 'reg' method")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["method"] == "quad":
            self.options["solver"] = 'NumInt'
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and ("settings" not in self.options.keys() or not(
                                                "n_coeffs_sparse" not in self.options["settings"].keys() or
                                                "sparsity" not in self.options["settings"].keys())):
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if self.options["solver"] == "LarsLasso" and ("settings" not in self.options.keys() or not(
                                                "alpha" not in self.options["settings"].keys())):
            self.options["settings"]["alpha"] = 1e-5

        if "order" not in self.options.keys():
            raise AssertionError("Please specify 'order'=[order_1, order_2, ..., order_dim] in options dictionary")

        if "order_max" not in self.options.keys():
            raise AssertionError("Please specify 'order_max' in options dictionary")

        if "fn_results" not in self.options.keys():
            self.options["fn_results"] = None

        if "verbose" not in self.options.keys():
            self.options["verbose"] = True

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = self.problem.dim

        if "n_cpu" in self.options.keys():
            self.n_cpu = self.options["n_cpu"]

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "error_norm" not in self.options.keys():
            self.options["error_norm"] = "relative"

        if "error_type" not in self.options.keys():
            self.options["error_type"] = "loocv"

        if "gradient_calculation" not in self.options.keys():
            self.options["gradient_calculation"] = "standard_forward"

        if "n_grid_gradient" not in self.options.keys():
            self.options["n_grid_gradient"] = 10

        if "projection_qoi" not in self.options.keys():
            self.options["projection_qoi"] = 0

        if "matrix_ratio" not in self.options.keys():
            self.options["matrix_ratio"] = 2

        if "seed" not in self.options.keys():
            self.options["seed"] = None

        if "lambda_eps_gradient" not in self.options.keys():
            self.options["lambda_eps_gradient"] = 0.95

        if "n_samples_validation" not in self.options.keys():
            self.options["n_samples_validation"] = 1e4

    def run(self):
        """
        Runs static gPC algorithm using Projection to solve problem.

        Returns
        -------
        gpc : GPC object instance
            GPC object containing all information i.e., Problem, Model, Grid, Basis, RandomParameter instances
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        res : ndarray of float [n_grid x n_out]
            Simulation results at n_grid points of the n_out output variables
        """

        # make initial random grid to determine gradients and projection matrix
        grid_original = RandomGrid(parameters_random=self.problem.parameters_random,
                                        options={"n_grid": self.options["n_grid_gradient"]})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu)

        # Run simulations
        iprint("Performing {} simulations!".format(grid_original.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        # self.problem.create_validation_set(n_samples=1e4, n_cpu=0)

        res_init = com.run(model=self.problem.model,
                           problem=self.problem,
                           coords=grid_original.coords,
                           coords_norm=grid_original.coords_norm,
                           i_iter=self.options["order_max"],
                           i_subiter=self.options["interaction_order"],
                           fn_results=None,
                           print_func_time=self.options["print_func_time"])
        com.close()

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine gradient
        start_time = time.time()
        grad_res = self.get_gradient(grid=grid_original, results=res)
        iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        if self.options["projection_qoi"] == "all":
            qoi_idx = np.arange(res_init.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["projection_qoi"]]
            n_qoi = 1

        # Set up reduced gPC
        self.problem_reduced = [0 for _ in range(n_qoi)]
        gpc = [0 for _ in range(n_qoi)]
        coeffs = [0 for _ in range(n_qoi)]

        for i_qoi, q_idx in enumerate(qoi_idx):
            if n_qoi == 1:
                fn_results = os.path.splitext(self.options["fn_results"])[0]
                res = res_init
            else:
                fn_results = os.path.splitext(self.options["fn_results"])[0] + "_qoi_{}_".format(q_idx)
                res = res_init[:, q_idx][:, np.newaxis]

            # Determine projection matrix
            p_matrix = determine_projection_matrix(gradient_results=grad_res,
                                                   qoi_idx=q_idx,
                                                   lambda_eps=self.options["lambda_eps_gradient"])

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
                             options=self.options)

            # save projection matrix in gPC object
            gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)

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

                # self.problem.create_validation_set(n_samples=1e4, n_cpu=0)

                res_new = com.run(model=self.problem.model,
                                  problem=self.problem,
                                  coords=grid_original.coords[self.options["n_grid_gradient"]:, :],
                                  coords_norm=grid_original.coords_norm[self.options["n_grid_gradient"]:, :],
                                  i_iter=self.options["order_max"],
                                  i_subiter=self.options["interaction_order"],
                                  fn_results=None,
                                  print_func_time=self.options["print_func_time"])
                com.close()

                if n_qoi != 1:
                    res_new = res_new[:, q_idx][:, np.newaxis]

                iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
                       tab=0, verbose=self.options["verbose"])

                res = np.vstack((res, res_new))

            # copy grid to gPC object and initialize transformed grid
            gpc[i_qoi].grid_original = copy.deepcopy(grid_original)
            gpc[i_qoi].grid = copy.deepcopy(grid_original)

            # transform variables of original grid to reduced parameter space
            gpc[i_qoi].grid.coords = np.dot(gpc[i_qoi].grid_original.coords,
                                            gpc[i_qoi].p_matrix.transpose())
            gpc[i_qoi].grid.coords_norm = np.dot(gpc[i_qoi].grid_original.coords_norm,
                                                 gpc[i_qoi].p_matrix.transpose())

            # gpc_red.interaction_order_current = 1
            # self.options_red = copy.deepcopy(self.options)
            # self.options_red["interaction_order"] = 1
            gpc[i_qoi].options = copy.deepcopy(self.options)

            # Initialize gpc matrix
            gpc[i_qoi].init_gpc_matrix()

            # Compute gpc coefficients
            coeffs[i_qoi] = gpc[i_qoi].solve(sim_results=res,
                                             solver=self.options["solver"],
                                             settings=self.options["settings"],
                                             verbose=True)

            # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
            gpc[i_qoi].problem.validation = self.problem.validation
            eps = gpc[i_qoi].validate(coeffs=coeffs[i_qoi], sim_results=res)

            iprint("-> {} {} error = {}".format(self.options["error_norm"],
                                                self.options["error_type"],
                                                eps), tab=0, verbose=self.options["verbose"])

            # save gpc objects and gpc coeffs
            if self.options["fn_results"]:

                write_gpc_pkl(gpc[i_qoi], fn_results + '.pkl')

                with h5py.File(fn_results + ".hdf5", "a") as f:

                    if "grid" in f.keys():
                        del f['grid']
                    f.create_dataset("grid/coords", data=gpc[i_qoi].grid.coords,
                                     maxshape=None, dtype="float64")
                    f.create_dataset("grid/coords_norm", data=gpc[i_qoi].grid.coords_norm,
                                     maxshape=None, dtype="float64")

                    if gpc[i_qoi].grid.coords_gradient.any():
                        f.create_dataset("grid/coords_gradient", data=gpc[i_qoi].grid.coords_gradient,
                                         maxshape=None, dtype="float64")
                        f.create_dataset("grid/coords_gradient_norm", data=gpc[i_qoi].grid.coords_gradient_norm,
                                         maxshape=None, dtype="float64")

                    if "coeffs" in f.keys():
                        del f['coeffs']
                    f.create_dataset("coeffs", data=coeffs[i_qoi], maxshape=None, dtype="float64")

                    if "gpc_matrix" in f.keys():
                        del f['gpc_matrix']
                    f.create_dataset("gpc_matrix", data=gpc[i_qoi].gpc_matrix, maxshape=None, dtype="float64")

                    if "p_matrix" in f.keys():
                        del f['p_matrix']
                    f.create_dataset("p_matrix", data=gpc[i_qoi].p_matrix, maxshape=None, dtype="float64")

                    if "results" in f.keys():
                        del f['results']
                    f.create_dataset("results", data=res, maxshape=None, dtype="float64")

                    if "results_gradient" in f.keys():
                        del f['results_gradient']
                    f.create_dataset("results_gradient", data=grad_res, maxshape=None, dtype="float64")

        if n_qoi == 1:
            return gpc[0], coeffs[0], res
        else:
            return gpc, coeffs, res


class RegAdaptive(Algorithm):
    """
    Adaptive regression approach based on leave one out cross validation error estimation
    """

    def __init__(self, problem, options):
        """
        Parameters
        ----------
        problem: Problem class instance
            GPC problem under investigation
        options["solver"]: str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        options["settings"]: dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
        options["order_start"] : int, optional, default=0
              Initial gPC expansion order (maximum order)
        options["order_end"] : int, optional, default=10
            Maximum Gpc expansion order to expand to (algorithm will terminate afterwards)
        options["interaction_order"]: int, optional, default=dim
            Define maximum interaction order of parameters (default: all interactions)
        options["fn_results"] : string, optional, default=None
            If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
        options["eps"] : float, optional, default=1e-3
            Relative mean error of leave-one-out cross validation
        options["verbose"] : boolean, optional, default=True
            Print output of iterations and sub-iterations (True/False)
        options["seed"] : int, optional, default=None
            Set np.random.seed(seed) in random_grid()
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["matrix_ratio"]: float, optional, default=1.5
            Ration between the number of model evaluations and the number of basis functions.
            If "adaptive_sampling" is activated this factor is only used to
            construct the initial grid depending on the initial number of basis functions determined by "order_start".
            (>1 results in an overdetermined system)
        options["error_norm"] : str, optional, default="relative"
            Choose if error is determined "relative" or "absolute". Use "absolute" error when the
            model generates outputs equal to zero.
        options["error_type"] : str, optional, default="loocv"
            Choose type of error to validate gpc approximation. Use "loocv" (Leave-one-Out cross validation)
            to omit any additional calculations and "nrmsd" (normalized root mean square deviation) to compare
            against a Problem.ValidationSet.
        options["adaptive_sampling"] : boolean, optional, default: True
            Adds samples adaptively to the expansion until the error is converged and continues by
            adding new basis functions.
        options["n_samples_validation"] : int, optional, default: 1e4
            Number of validation points used to determine the NRMSD if chosen as "error_type". Does not create a
            validation set if there is already one present in the Problem instance (problem.validation).

        Examples
        --------
        >>> import pygpc
        >>> # initialize adaptive gPC algorithm
        >>> algorithm = pygpc.RegAdaptive(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(RegAdaptive, self).__init__(problem, options)

        # check contents of settings dict and set defaults
        if "order_start" not in self.options.keys():
            self.options["order_start"] = 0

        if "order_end" not in self.options.keys():
            self.options["order_end"] = 10

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = problem.dim

        if "fn_results" not in self.options.keys():
            self.options["fn_results"] = None

        if "eps" not in self.options.keys():
            self.options["eps"] = 1e-3

        if "verbose" not in self.options.keys():
            self.options["verbose"] = True

        if "seed" not in self.options.keys():
            self.options["seed"] = None

        if "n_cpu" in self.options.keys():
            self.n_cpu = self.options["n_cpu"]

        if "matrix_ratio" not in self.options.keys():
            self.options["matrix_ratio"] = 2

        if "solver" not in self.options.keys():
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for adaptive algorithm.")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and "settings" not in self.options.keys():
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if self.options["solver"] == "LarsLasso" and ("settings" not in self.options.keys() or
                                                      "alpha" not in self.options["settings"].keys()):
            self.options["settings"]["alpha"] = 1e-5

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "error_norm" not in self.options.keys():
            self.options["error_norm"] = "relative"

        if "error_type" not in self.options.keys():
            self.options["error_type"] = "loocv"

        if "adaptive_sampling" not in self.options.keys():
            self.options["adaptive_sampling"] = True

        if "n_samples_validation" not in self.options.keys():
            self.options["n_samples_validation"] = 1e4

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
        # initialize iterators
        eps = self.options["eps"] + 1.0
        i_grid = 0
        order = self.options["order_start"]
        res_complete = None
        first_iter = True

        # Add a validation set if nrmsd is chosen and no validation set is yet present
        if self.options["error_type"] == "nrmsd" and not isinstance(self.problem.validation, ValidationSet):
            self.problem.create_validation_set(n_samples=self.options["n_samples_validation"],
                                               n_cpu=self.options["n_cpu"])

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu)

        # Initialize Reg gPC object
        gpc = Reg(problem=self.problem,
                  order=self.options["order_start"] * np.ones(self.problem.dim),
                  order_max=self.options["order_start"],
                  order_max_norm=self.options["order_max_norm"],
                  interaction_order=self.options["interaction_order"],
                  options=self.options)
        extended_basis = True

        # Initialize Grid object
        gpc.grid = RandomGrid(parameters_random=self.problem.parameters_random,
                              options={"n_grid": np.ceil(self.options["matrix_ratio"] * gpc.basis.n_basis),
                                       "seed": self.options["seed"]})

        gpc.interaction_order_current = min(self.options["interaction_order"], self.options["order_start"])
        gpc.solver = self.options["solver"]
        gpc.settings = self.options["settings"]
        gpc.options = copy.deepcopy(self.options)

        # Initialize gpc matrix
        gpc.init_gpc_matrix()
        gpc.n_grid.pop(0)
        gpc.n_basis.pop(0)

        # Main iterations (order)
        while (eps > self.options["eps"]) and order <= self.options["order_end"]:

            iprint("Order #{}".format(order), tab=0, verbose=self.options["verbose"])
            iprint("==========", tab=0, verbose=self.options["verbose"])

            # determine new possible set of basis functions for next main iteration
            multi_indices_all_new = get_multi_indices_max_order(self.problem.dim, order, self.options["order_max_norm"])
            multi_indices_all_current = np.array([list(map(lambda x:x.p["i"], _b)) for _b in gpc.basis.b])

            idx_old = np.hstack([np.where((multi_indices_all_current[i, :] == multi_indices_all_new).all(axis=1))
                                 for i in range(multi_indices_all_current.shape[0])])

            multi_indices_all_new = np.delete(multi_indices_all_new, idx_old, axis=0)

            interaction_order_current_max = np.min([order, self.options["interaction_order"]])

            # sub-iterations (interaction orders)
            while (gpc.interaction_order_current <= self.options["interaction_order"] and
                   gpc.interaction_order_current <= interaction_order_current_max) and eps > self.options["eps"]:

                iprint("Sub-iteration #{}".format(gpc.interaction_order_current),
                       tab=0, verbose=self.options["verbose"])
                iprint("================",
                       tab=0, verbose=self.options["verbose"])

                if order != self.options["order_start"]:

                    # filter out polynomials of interaction_order = interaction_order_current
                    interaction_order_list = np.sum(multi_indices_all_new > 0, axis=1)
                    multi_indices_added = multi_indices_all_new[
                                          interaction_order_list == gpc.interaction_order_current, :]

                    # continue while loop if no basis function was added because of max norm constraint
                    if not multi_indices_added.any():
                        iprint("-> No basis functions to add because of max_norm constraint ... Continuing ... ",
                               tab=0, verbose=self.options["verbose"])

                        # increase current interaction order
                        gpc.interaction_order_current = gpc.interaction_order_current + 1

                        continue

                    # construct 2D list with new BasisFunction objects
                    b_added = [[0 for _ in range(self.problem.dim)] for _ in range(multi_indices_added.shape[0])]

                    for i_basis in range(multi_indices_added.shape[0]):
                        for i_p, p in enumerate(self.problem.parameters_random):  # Ordered Dict of RandomParameter
                            b_added[i_basis][i_p] = self.problem.parameters_random[p].init_basis_function(
                                order=multi_indices_added[i_basis, i_p])

                    # extend basis
                    gpc.basis.extend_basis(b_added)
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

                            res = com.run(model=gpc.problem.model,
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
                                res_complete = res
                            else:
                                res_complete = np.vstack([res_complete, res])

                            i_grid = gpc.grid.coords.shape[0]

                            # Determine QOIs with NaN in results
                            non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
                            n_nan = res_complete.shape[1] - non_nan_mask.size

                            if n_nan > 0:
                                iprint("In {}/{} output quantities NaN's were found.".format(n_nan, res_complete.shape[1]),
                                       tab=0, verbose=self.options["verbose"])

                        # update gpc matrix
                        gpc.update_gpc_matrix()

                        # determine gpc coefficients
                        coeffs = gpc.solve(sim_results=res_complete,
                                           solver=gpc.solver,
                                           settings=gpc.settings,
                                           verbose=True)

                        # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
                        eps = gpc.validate(coeffs=coeffs, sim_results=res_complete[:, non_nan_mask])

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
                    with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:
                        if "coeffs" in f.keys():
                            del f['coeffs']
                        f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

                    write_gpc_pkl(gpc, os.path.splitext(self.options["fn_results"])[0] + '.pkl')

                    fn = os.path.join(os.path.splitext(self.options["fn_results"])[0] +
                                      '_' + str(order).zfill(2) + "_" + str(gpc.interaction_order_current).zfill(2))
                    write_gpc_pkl(gpc, fn + '.pkl')

                # save gpc matrix in .hdf5 file
                gpc.save_gpc_matrix_hdf5()

                # increase current interaction order
                gpc.interaction_order_current = gpc.interaction_order_current + 1

            # reset interaction order counter
            gpc.interaction_order_current = 1

            # increase main order
            order = order + 1

        com.close()

        # determine gpc coefficients
        coeffs = gpc.solve(sim_results=res_complete,
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

        return gpc, coeffs, res_complete


class RegAdaptiveProjection(Algorithm):
    """
    Adaptive regression approach using projection and leave one out cross validation error estimation
    """

    def __init__(self, problem, options):
        """
        Parameters
        ----------
        problem: Problem class instance
            GPC problem under investigation
        options["solver"]: str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        options["settings"]: dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0
        options["order_start"] : int, optional, default=0
              Initial gPC expansion order (maximum order)
        options["order_end"] : int, optional, default=10
            Maximum Gpc expansion order to expand to (algorithm will terminate afterwards)
        options["interaction_order"]: int, optional, default=dim
            Define maximum interaction order of parameters (default: all interactions)
        options["fn_results"] : string, optional, default=None
            If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
        options["eps"] : float, optional, default=1e-3
            Relative mean error of leave-one-out cross validation
        options["verbose"] : boolean, optional, default=True
            Print output of iterations and sub-iterations (True/False)
        options["seed"] : int, optional, default=None
            Set np.random.seed(seed) in random_grid()
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["matrix_ratio"]: float, optional, default=1.5
            Ration between the number of model evaluations and the number of basis functions.
            If "adaptive_sampling" is activated this factor is only used to
            construct the initial grid depending on the initial number of basis functions determined by "order_start".
            (>1 results in an overdetermined system)
        options["error_norm"] : str, optional, default="relative"
            Choose if error is determined "relative" or "absolute". Use "absolute" error when the
            model generates outputs equal to zero.
        options["error_type"] : str, optional, default="loocv"
            Choose type of error to validate gpc approximation. Use "loocv" (Leave-one-Out cross validation)
            to omit any additional calculations and "nrmsd" (normalized root mean square deviation) to compare
            against a Problem.ValidationSet.
        options["projection_qoi"] : int or str, optional, default: 0
            Choose for which QOI the projection is determined for. The other QOIs use the same projection.
            Alternatively, the projection can be determined for every QOI independently (qoi_index or "all").
        options["gradient_calculation"] : str, optional, default="standard_forward"
            Type of the calculation scheme to determine the gradient in the grid points
            - "standard_forward" ... Forward approximation (creates additional dim*n_grid grid-points in the axis
            directions)
            - "???" ... ???
        options["n_grid_gradient"] : float, optional, default: 10
            Number of initial grid points to determine gradient and projection matrix. When the algorithm goes
            into the main interations the number will be increased depending on the options "matrix_ratio"
            and "adaptive_sampling".
        options["lambda_eps_gradient"] : float, optional, default: 0.95
            Bound of principal components in %. All eigenvectors are included until lambda_eps of total sum of all
            eigenvalues is included in the system.
        options["adaptive_sampling"] : boolean, optional, default: True
            Adds samples adaptively to the expansion until the error is converged and continues by
            adding new basis functions.
        options["n_samples_validation"] : int, optional, default: 1e4
            Number of validation points used to determine the NRMSD if chosen as "error_type". Does not create a
            validation set if there is already one present in the Problem instance (problem.validation).

        Examples
        --------
        >>> import pygpc
        >>> # initialize adaptive gPC algorithm
        >>> algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
        >>> # run algorithm
        >>> gpc, coeffs, results = algorithm.run()
        """
        super(RegAdaptiveProjection, self).__init__(problem, options)

        # check contents of settings dict and set defaults
        if "order_start" not in self.options.keys():
            self.options["order_start"] = 0

        if "order_end" not in self.options.keys():
            self.options["order_end"] = 10

        if "interaction_order" not in self.options.keys():
            self.options["interaction_order"] = problem.dim

        if "fn_results" not in self.options.keys():
            self.options["fn_results"] = None

        if "eps" not in self.options.keys():
            self.options["eps"] = 1e-3

        if "verbose" not in self.options.keys():
            self.options["verbose"] = True

        if "seed" not in self.options.keys():
            self.options["seed"] = None

        if "n_cpu" in self.options.keys():
            self.n_cpu = self.options["n_cpu"]

        if "matrix_ratio" not in self.options.keys():
            self.options["matrix_ratio"] = 2

        if "solver" not in self.options.keys():
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for adaptive algorithm.")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and "settings" not in self.options.keys():
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if self.options["solver"] == "LarsLasso" and ("settings" not in self.options.keys() or
                                                      "alpha" not in self.options["settings"].keys()):
            self.options["settings"]["alpha"] = 1e-5

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "order_max_norm" not in self.options.keys():
            self.options["order_max_norm"] = 1.

        if "error_norm" not in self.options.keys():
            self.options["error_norm"] = "relative"

        if "error_type" not in self.options.keys():
            self.options["error_type"] = "loocv"

        if "gradient_calculation" not in self.options.keys():
            self.options["gradient_calculation"] = "standard_forward"

        if "n_grid_gradient" not in self.options.keys():
            self.options["n_grid_gradient"] = 10

        if "projection_qoi" not in self.options.keys():
            self.options["projection_qoi"] = 0

        if "lambda_eps_gradient" not in self.options.keys():
            self.options["lambda_eps_gradient"] = 0.95

        if "n_samples_validation" not in self.options.keys():
            self.options["n_samples_validation"] = 1e4

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

        # initialize iterators
        eps = self.options["eps"] + 1.0
        order = self.options["order_start"]

        # make initial random grid to determine gradients and projection matrix
        grid_original = RandomGrid(parameters_random=self.problem.parameters_random,
                                   options={"n_grid": self.options["n_grid_gradient"]})

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu)

        # Run initial simulations to determine initial projection matrix
        iprint("Performing {} simulations!".format(grid_original.coords.shape[0]),
               tab=0, verbose=self.options["verbose"])

        start_time = time.time()

        # self.problem.create_validation_set(n_samples=1e4, n_cpu=0)

        res_init = com.run(model=self.problem.model,
                           problem=self.problem,
                           coords=grid_original.coords,
                           coords_norm=grid_original.coords_norm,
                           i_iter=self.options["order_start"],
                           i_subiter=self.options["interaction_order"],
                           fn_results=None,
                           print_func_time=self.options["print_func_time"])

        i_grid = grid_original.n_grid

        iprint('Total function evaluation: ' + str(time.time() - start_time) + ' sec',
               tab=0, verbose=self.options["verbose"])

        # Determine QOIs with NaN in results
        non_nan_mask = np.where(np.all(~np.isnan(res_init), axis=0))[0]
        n_nan = res_init.shape[1] - non_nan_mask.size

        if n_nan > 0:
            iprint(
                "In {}/{} output quantities NaN's were found.".format(n_nan, res_init.shape[1]),
                tab=0, verbose=self.options["verbose"])

        if self.options["projection_qoi"] == "all":
            qoi_idx = np.arange(res_init.shape[1])
            n_qoi = len(qoi_idx)
        else:
            qoi_idx = [self.options["projection_qoi"]]
            n_qoi = 1

        self.problem_reduced = [None for _ in range(n_qoi)]
        gpc = [None for _ in range(n_qoi)]
        coeffs = [None for _ in range(n_qoi)]
        grid = [None for _ in range(n_qoi)]
        gradient_results_complete = None
        self.options["order_max"] = None

        for i_qoi, q_idx in enumerate(qoi_idx):
            first_iter = True

            if n_qoi == 1:
                fn_results = os.path.splitext(self.options["fn_results"])[0]
                res_complete = res_init
            else:
                fn_results = os.path.splitext(self.options["fn_results"])[0] + "_qoi_{}_".format(q_idx)
                res_complete = res_init[:, q_idx][:, np.newaxis]

            # Set up initial reduced problem
            # Determine gradient
            start_time = time.time()
            gradient_results_complete = self.get_gradient(grid=grid_original, results=res_complete,
                                                          gradient_results=gradient_results_complete)
            iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                   tab=0, verbose=self.options["verbose"])

            # Determine projection matrix
            p_matrix = determine_projection_matrix(gradient_results=gradient_results_complete,
                                                   qoi_idx=q_idx,
                                                   lambda_eps=self.options["lambda_eps_gradient"])

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
                             options=self.options)
            extended_basis = False

            # save projection matrix in gPC object
            gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)

            # transformed grid
            grid[i_qoi] = copy.deepcopy(grid_original)

            # transform variables of original grid to reduced parameter space
            grid[i_qoi].coords = np.dot(grid_original.coords,
                                        gpc[i_qoi].p_matrix.transpose())
            grid[i_qoi].coords_norm = np.dot(grid_original.coords_norm,
                                             gpc[i_qoi].p_matrix.transpose())

            # assign transformed grid
            gpc[i_qoi].grid = grid[i_qoi]

            # Initialize gpc matrix
            gpc[i_qoi].init_gpc_matrix()
            gpc[i_qoi].n_grid.pop(0)
            gpc[i_qoi].n_basis.pop(0)

            gpc[i_qoi].solver = self.options["solver"]
            gpc[i_qoi].settings = self.options["settings"]
            gpc[i_qoi].options = copy.deepcopy(self.options)

            # Main iterations (order)
            while (eps > self.options["eps"]) and order <= self.options["order_end"]:

                iprint("Order #{}".format(order), tab=0, verbose=self.options["verbose"])
                iprint("==========", tab=0, verbose=self.options["verbose"])

                # determine new possible set of basis functions for next main iteration
                multi_indices_all_new = get_multi_indices_max_order(self.problem_reduced[i_qoi].dim, order, self.options["order_max_norm"])

                multi_indices_all_current = np.array([list(map(lambda x:x.p["i"], _b)) for _b in gpc[i_qoi].basis.b])
                idx_old = np.hstack(
                    [np.where((multi_indices_all_current[i, :] == multi_indices_all_new).all(axis=1))
                     for i in range(multi_indices_all_current.shape[0])])
                multi_indices_all_new = np.delete(multi_indices_all_new, idx_old, axis=0)

                interaction_order_current_max = np.min([order,
                                                        self.options["interaction_order"],
                                                        self.problem_reduced[i_qoi].dim])

                if order == self.options["order_start"]:
                    interaction_order_current = np.min([self.problem_reduced[i_qoi].dim,
                                                        self.options["interaction_order"],
                                                        interaction_order_current_max])

                # sub-iterations (interaction orders)
                while (interaction_order_current <= self.options["interaction_order"] and
                       interaction_order_current <= interaction_order_current_max) and eps > self.options["eps"]:

                    iprint("Sub-iteration #{}".format(interaction_order_current),
                           tab=0, verbose=self.options["verbose"])
                    iprint("================",
                           tab=0, verbose=self.options["verbose"])

                    if order != self.options["order_start"]:

                        # filter out polynomials of interaction_order = interaction_order_current
                        interaction_order_list = np.sum(multi_indices_all_new > 0, axis=1)
                        multi_indices_added = multi_indices_all_new[
                                              interaction_order_list == interaction_order_current, :]

                        # continue while loop if no basis function was added because of max norm constraint
                        if not multi_indices_added.any():
                            iprint("-> No basis functions to add because of max_norm constraint ... Continuing ... ",
                                   tab=0, verbose=self.options["verbose"])

                            # increase current interaction order
                            interaction_order_current = interaction_order_current + 1

                            continue

                        # construct 2D list with new BasisFunction objects
                        b_added = [[0 for _ in range(self.problem_reduced[i_qoi].dim)] for _ in range(multi_indices_added.shape[0])]

                        for i_basis in range(multi_indices_added.shape[0]):
                            for i_p, p in enumerate(self.problem_reduced[i_qoi].parameters_random):  # Ordered Dict of RandomParameter
                                b_added[i_basis][i_p] = self.problem_reduced[i_qoi].parameters_random[p].init_basis_function(
                                    order=multi_indices_added[i_basis, i_p])

                        # extend basis
                        gpc[i_qoi].basis.extend_basis(b_added)
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
                            # do not increase sample size immediately when basis was extended, try first with old samples
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
                                res = com.run(model=self.problem.model,
                                              problem=self.problem,
                                              coords=grid_original.coords[i_grid:grid_original.coords.shape[0]],
                                              coords_norm=grid_original.coords_norm[i_grid:grid_original.coords.shape[0]],
                                              i_iter=order,
                                              i_subiter=interaction_order_current,
                                              fn_results=gpc[i_qoi].fn_results,
                                              print_func_time=self.options["print_func_time"])

                                if n_qoi != 1:
                                    res = res[:, q_idx][:, np.newaxis]

                                iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec',
                                       tab=0, verbose=self.options["verbose"])

                                # Append result to solution matrix (RHS)
                                res_complete = np.vstack([res_complete, res])

                                i_grid = grid_original.coords.shape[0]

                                # Determine QOIs with NaN in results
                                non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
                                n_nan = res_complete.shape[1] - non_nan_mask.size

                                if n_nan > 0:
                                    iprint(
                                        "In {}/{} output quantities NaN's were found.".format(n_nan, res_complete.shape[1]),
                                        tab=0, verbose=self.options["verbose"])

                                # Determine gradient
                                start_time = time.time()
                                gradient_results_complete = self.get_gradient(grid=grid_original, results=res_complete,
                                                                              gradient_results=gradient_results_complete)
                                iprint('Gradient evaluation: ' + str(time.time() - start_time) + ' sec',
                                       tab=0, verbose=self.options["verbose"])

                                # Determine projection matrix
                                p_matrix = determine_projection_matrix(gradient_results=gradient_results_complete,
                                                                       qoi_idx=q_idx,
                                                                       lambda_eps=self.options["lambda_eps_gradient"])

                                # save projection matrix in gPC object
                                gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)

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
                                                     order=[order - 1 for _ in range(dim_reduced)],
                                                     order_max=order - 1,
                                                     order_max_norm=self.options["order_max_norm"],
                                                     interaction_order=self.options["interaction_order"],
                                                     options=self.options)

                                    # save projection matrix in gPC object
                                    gpc[i_qoi].p_matrix = copy.deepcopy(p_matrix)

                                    # add basis functions of this subiteration to be complete again
                                    # determine new possible set of basis functions
                                    multi_indices_all_new = get_multi_indices_max_order(self.problem_reduced[i_qoi].dim,
                                                                                        order,
                                                                                        self.options["order_max_norm"])
                                    multi_indices_all_current = np.array(
                                        [list(map(lambda x: x.p["i"], _b)) for _b in gpc[i_qoi].basis.b])

                                    idx_old = np.hstack(
                                        [np.where((multi_indices_all_current[i, :] == multi_indices_all_new).all(axis=1))
                                         for i in range(multi_indices_all_current.shape[0])])

                                    multi_indices_all_new = np.delete(multi_indices_all_new, idx_old, axis=0)

                                    # remove multi-indices, we did not consider yet in this sub-iteration
                                    # filter out polynomials of interaction_order > interaction_order_current
                                    interaction_order_list = np.sum(multi_indices_all_new > 0, axis=1)
                                    multi_indices_added = multi_indices_all_new[
                                                          interaction_order_list <= interaction_order_current, :]

                                    # construct 2D list with new BasisFunction objects
                                    b_added = [[0 for _ in range(self.problem_reduced[i_qoi].dim)] for _ in
                                               range(multi_indices_added.shape[0])]

                                    for i_basis in range(multi_indices_added.shape[0]):
                                        for i_p, p in enumerate(
                                                self.problem_reduced[i_qoi].parameters_random):  # Ordered Dict of RandomParameter
                                            b_added[i_basis][i_p] = self.problem_reduced[i_qoi].parameters_random[p].init_basis_function(
                                                order=multi_indices_added[i_basis, i_p])

                                    # extend basis
                                    gpc[i_qoi].basis.extend_basis(b_added)

                                    # transformed grid
                                    grid[i_qoi] = copy.deepcopy(grid_original)

                                    # transform variables of original grid to reduced parameter space
                                    grid[i_qoi].coords = np.dot(grid_original.coords,
                                                                gpc[i_qoi].p_matrix.transpose())
                                    grid[i_qoi].coords_norm = np.dot(grid_original.coords_norm,
                                                                     gpc[i_qoi].p_matrix.transpose())

                                    # assign transformed grid
                                    gpc[i_qoi].grid = grid[i_qoi]

                                    # Initialize gpc matrix
                                    gpc[i_qoi].init_gpc_matrix()

                                    gpc[i_qoi].interaction_order_current = interaction_order_current
                                    gpc[i_qoi].solver = self.options["solver"]
                                    gpc[i_qoi].settings = self.options["settings"]
                                    gpc[i_qoi].options = copy.deepcopy(self.options)

                                else:
                                    coords_new = np.dot(grid_original.coords, #[gpc[i_qoi].grid.n_grid:, :]
                                                        gpc[i_qoi].p_matrix.transpose())
                                    coords_norm_new = np.dot(grid_original.coords_norm, #[gpc[i_qoi].grid.n_grid:, :]
                                                             gpc[i_qoi].p_matrix.transpose())

                                    gpc[i_qoi].grid.coords = coords_new
                                    gpc[i_qoi].grid.coords_norm = coords_norm_new

                        gpc[i_qoi].init_gpc_matrix()

                        # determine gpc coefficients
                        coeffs[i_qoi] = gpc[i_qoi].solve(sim_results=res_complete,
                                                         solver=gpc[i_qoi].solver,
                                                         settings=gpc[i_qoi].settings,
                                                         verbose=True)

                        # validate gpc approximation (determine nrmsd or loocv specified in options["error_type"])
                        gpc[i_qoi].problem.validation = self.problem.validation
                        eps = gpc[i_qoi].validate(coeffs=coeffs[i_qoi], sim_results=res_complete[:, non_nan_mask])

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
                        with h5py.File(fn_results + ".hdf5", "a") as f:
                            if "coeffs" in f.keys():
                                del f['coeffs']
                            f.create_dataset("coeffs", data=coeffs[i_qoi], maxshape=None, dtype="float64")

                            if "p_matrix" in f.keys():
                                del f['p_matrix']
                            f.create_dataset("p_matrix", data=p_matrix, maxshape=None, dtype="float64")

                        write_gpc_pkl(gpc[i_qoi], os.path.splitext(self.options["fn_results"])[0] + '.pkl')

                        fn = os.path.join(os.path.splitext(self.options["fn_results"])[0] +
                                          '_' + str(order).zfill(2) + "_" + str(interaction_order_current).zfill(2))
                        write_gpc_pkl(gpc[i_qoi], fn + '.pkl')

                    # save gpc matrix in .hdf5 file
                    gpc[i_qoi].save_gpc_matrix_hdf5()

                    # increase current interaction order
                    interaction_order_current = interaction_order_current + 1

                # reset interaction order counter
                interaction_order_current = 1

                # increase main order
                order = order + 1

            com.close()

            # determine gpc coefficients
            coeffs[i_qoi] = gpc[i_qoi].solve(sim_results=res_complete,
                                             solver=gpc[i_qoi].solver,
                                             settings=gpc[i_qoi].settings,
                                             verbose=True)

            # save gpc object gpc coeffs and projection matrix
            if self.options["fn_results"]:
                write_gpc_pkl(gpc[i_qoi], fn_results + '.pkl')

                with h5py.File(fn_results + ".hdf5", "a") as f:
                    if "coeffs" in f.keys():
                        del f['coeffs']
                    f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

                    if "p_matrix" in f.keys():
                        del f['p_matrix']
                    f.create_dataset("p_matrix", data=p_matrix, maxshape=None, dtype="float64")

        if n_qoi == 1:
            return gpc[0], coeffs[0], res_complete
        else:
            return gpc, coeffs, res_complete
