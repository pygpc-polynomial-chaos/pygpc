# -*- coding: utf-8 -*-
from .SGPC import *
from .EGPC import *
from .Grid import *
from .Computation import *
from .io import write_gpc_pkl
import h5py
import os
import time
import copy


class Algorithm(object):
    """
    Class for GPC algorithms
    """

    def __init__(self, problem, options):
        """
        Constructor; Initializes GPC algorithm

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC problem to investigate
        options : dict
            Algorithm specific options (see sub-classes for more details)
        """
        self.problem = problem
        self.options = options

        # Generate results folder if it doesn't exist
        if not os.path.exists(os.path.split(self.options["fn_results"])[0]):
            os.makedirs(os.path.split(self.options["fn_results"])[0])


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
            Maximum global expansion order (sum of all exponents).
            The maximum expansion order considers the sum of the orders of combined polynomials only
        options["interaction_order"]: int
            Number of random variables, which can interact with each other.
            All polynomials are ignored, which have an interaction order greater than the specified
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        grid: Grid object instance
            Grid object to use for static gPC (RandomGrid, SparseGrid, TensorGrid)

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

        if self.options["solver"] == "OMP" and ("settings" not in self.options.keys() or
                                                "n_coeffs_sparse" not in self.options["settings"].keys()):
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

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
                      interaction_order=self.options["interaction_order"],
                      fn_results=self.options["fn_results"])

        elif self.options["method"] == "quad":
            gpc = Quad(problem=self.problem,
                       order=self.options["order"],
                       order_max=self.options["order_max"],
                       interaction_order=self.options["interaction_order"],
                       fn_results=self.options["fn_results"])

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

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:
            write_gpc_pkl(gpc, os.path.splitext(self.options["fn_results"])[0] + '.pkl')

            with h5py.File(os.path.splitext(self.options["fn_results"])[0] + ".hdf5", "a") as f:
                if "coeffs" in f.keys():
                    del f['coeffs']
                f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

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
            Relative mean error of leave one out cross validation
        options["verbose"] : boolean, optional, default=True
            Print output of iterations and sub-iterations (True/False)
        options["seed"] : int, optional, default=None
            Set np.random.seed(seed) in random_grid()
        options["n_cpu"] : int, optional, default=1
            Number of threads to use for parallel evaluation of the model function.
        options["matrix_ratio"]: float, optional, default=1.5
            Ration between the number of model evaluations and the number of basis functions.
            (>1 results in an overdetermined system)

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
            self.options["matrix_ratio"] = 1.5

        if "solver" not in self.options.keys():
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for adaptive algorithm.")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and "settings" not in self.options.keys():
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

        if "print_func_time" not in self.options.keys():
            self.options["print_func_time"] = False

        if "n_loocv" not in self.options.keys():
            self.options["n_loocv"] = 100

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

        # Initialize parallel Computation class
        com = Computation(n_cpu=self.n_cpu)

        # Initialize Reg gPC object
        gpc = Reg(problem=self.problem,
                  order=self.options["order_start"] * np.ones(self.problem.dim),
                  order_max=self.options["order_start"],
                  interaction_order=self.options["interaction_order"],
                  fn_results=self.options["fn_results"])

        # Initialize Grid object
        gpc.grid = RandomGrid(parameters_random=self.problem.parameters_random,
                              options={"n_grid": np.ceil(self.options["matrix_ratio"] * gpc.basis.n_basis),
                                       "seed": self.options["seed"]})

        gpc.interaction_order_current = min(self.options["interaction_order"], self.options["order_start"])
        gpc.solver = self.options["solver"]
        gpc.settings = self.options["settings"]

        # Initialize gpc matrix
        gpc.init_gpc_matrix()

        # Main iterations (order)
        while (eps > self.options["eps"]) and order < self.options["order_end"]:

            iprint("\n")
            iprint("Order #{}".format(order), tab=0, verbose=self.options["verbose"])
            iprint("==========", tab=0, verbose=self.options["verbose"])

            # determine new possible basis for next main iteration
            multi_indices_all_new = get_multi_indices_max_order(self.problem.dim, order)
            multi_indices_all_new = multi_indices_all_new[np.sum(multi_indices_all_new, axis=1) == order]

            interaction_order_current_max = np.min([order, self.options["interaction_order"]])

            # sub-iterations (interaction orders)
            while (gpc.interaction_order_current <= self.options["interaction_order"] and
                   gpc.interaction_order_current <= interaction_order_current_max) and eps > self.options["eps"]:

                iprint("\n")
                iprint("Sub-iteration #{}".format(gpc.interaction_order_current),
                       tab=0, verbose=self.options["verbose"])
                iprint("================",
                       tab=0, verbose=self.options["verbose"])

                if order != self.options["order_start"]:

                    # filter out polynomials of interaction_order = interaction_order_current
                    interaction_order_list = np.sum(multi_indices_all_new > 0, axis=1)
                    multi_indices_added = multi_indices_all_new[
                                          interaction_order_list == gpc.interaction_order_current, :]

                    # construct 2D list with new BasisFunction objects
                    b_added = [[0 for _ in range(self.problem.dim)] for _ in range(multi_indices_added.shape[0])]

                    for i_basis in range(multi_indices_added.shape[0]):
                        for i_p, p in enumerate(self.problem.parameters_random):  # Ordered Dict of RandomParameter
                            b_added[i_basis][i_p] = self.problem.parameters_random[p].init_basis_function(
                                order=multi_indices_added[i_basis, i_p])

                    # extend basis
                    gpc.basis.extend_basis(b_added)

                    # extend grid
                    gpc.grid.extend_random_grid(n_grid_new=np.ceil(gpc.basis.n_basis * self.options["matrix_ratio"]),
                                                seed=None)

                    # update gpc matrix
                    gpc.update_gpc_matrix()

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
                n_nan = res_complete.shape[1]-non_nan_mask.size

                if n_nan > 0:
                    iprint("In {}/{} output quantities NaN's were found.".format(n_nan, res_complete.shape[1]),
                           tab=0, verbose=self.options["verbose"])

                # determine gpc coefficients
                coeffs = gpc.solve(sim_results=res_complete,
                                   solver=gpc.solver,
                                   settings=gpc.settings,
                                   verbose=True)

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

                # determine error
                eps = gpc.loocv(sim_results=res_complete[:, non_nan_mask],
                                solver=gpc.solver,
                                settings=gpc.settings,
                                n_loocv=self.options["n_loocv"])

                iprint("-> error = {}".format(eps), tab=0, verbose=self.options["verbose"])

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
