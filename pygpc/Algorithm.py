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
               tab=1, verbose=self.options["verbose"])

        start_time = time.time()

        res = com.run(model=gpc.problem.model,
                      problem=gpc.problem,
                      coords=gpc.grid.coords,
                      coords_norm=gpc.grid.coords_norm,
                      i_iter=gpc.order_max,
                      i_subiter=gpc.interaction_order,
                      fn_results=gpc.fn_results)

        iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec\n',
               tab=1, verbose=self.options["verbose"])

        com.close()

        # Compute gpc coefficients
        coeffs = gpc.solve(sim_results=res,
                           solver=self.options["solver"],
                           settings=self.options["settings"])

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:
            fn = os.path.join(os.path.splitext(self.options["fn_results"])[0])
            write_gpc_pkl(gpc, fn + '.pkl')

            with h5py.File(self.options["fn_results"], "a") as f:
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

        if not self.options["fn_results"].endswith('.hdf5'):
            self.options["fn_results"] = os.path.splitext(self.options["fn_results"])[0] + ".hdf5"

        if "solver" not in self.options.keys():
            raise AssertionError("Please specify 'Moore-Penrose' or 'OMP' as solver for adaptive algorithm.")

        if self.options["solver"] == "Moore-Penrose":
            self.options["settings"] = None

        if self.options["solver"] == "OMP" and "settings" not in self.options.keys():
            raise AssertionError("Please specify correct solver settings for OMP in 'settings'")

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

            print("Order #{}".format(order))
            print("==========")

            # determine new possible basis for next main iteration
            multi_indices_all_new = get_multi_indices_max_order(self.problem.dim, order)
            multi_indices_all_new = multi_indices_all_new[np.sum(multi_indices_all_new, axis=1) == order]

            # sub-iterations (interaction orders)
            while (gpc.interaction_order_current <= self.options["interaction_order"]) and eps > self.options["eps"]:

                iprint("Sub-iteration #{}".format(gpc.interaction_order_current),
                       tab=1, verbose=self.options["verbose"])
                iprint("================",
                       tab=1, verbose=self.options["verbose"])

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
                       tab=1, verbose=self.options["verbose"])

                start_time = time.time()

                res = com.run(model=gpc.problem.model,
                              problem=gpc.problem,
                              coords=gpc.grid.coords[int(i_grid):int(len(gpc.grid.coords))],
                              coords_norm=gpc.grid.coords_norm[int(i_grid):int(len(gpc.grid.coords))],
                              i_iter=order,
                              i_subiter=gpc.interaction_order_current,
                              fn_results=gpc.fn_results)

                iprint('Total parallel function evaluation: ' + str(time.time() - start_time) + ' sec\n',
                       tab=1, verbose=self.options["verbose"])

                # Append result to solution matrix (RHS)
                if i_grid == 0:
                    res_complete = res
                else:
                    res_complete = np.vstack([res_complete, res])

                i_grid = gpc.grid.coords.shape[0]

                # Determine location of NaN in results
                non_nan_mask = np.where(np.all(~np.isnan(res_complete), axis=0))[0]
                n_nan = non_nan_mask.size

                if n_nan > 0:
                    iprint("In {}/{} output quantities NaN's were found.".format(n_nan, res_complete.shape[1]),
                           tab=1, verbose=self.options["verbose"])

                # save gpc object and append results for this sub-iteration
                if self.options["fn_results"]:
                    fn = os.path.join(os.path.splitext(self.options["fn_results"])[0] +
                                      '_' + str(order).zfill(2) + "_" + str(gpc.interaction_order_current).zfill(2))
                    write_gpc_pkl(gpc, fn + '_gpc.pkl')

                # determine error
                eps = gpc.loocv(sim_results=res_complete[:, non_nan_mask],
                                solver=gpc.solver,
                                settings=gpc.settings)
                iprint("-> error = {}".format(eps), tab=1, verbose=self.options["verbose"])

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
                           settings=gpc.settings)

        # save gpc object and gpc coeffs
        if self.options["fn_results"]:
            fn = os.path.join(os.path.splitext(self.options["fn_results"])[0])
            write_gpc_pkl(gpc, fn + '.pkl')

            with h5py.File(self.options["fn_results"], "a") as f:
                if "coeffs" in f.keys():
                    del f['coeffs']
                f.create_dataset("coeffs", data=coeffs, maxshape=None, dtype="float64")

        return gpc, coeffs, res_complete

        # def reg_adaptive_tms(self, problem, pdf_type, pdf_shape, limits, func, args=(), fname=None,
        #                      order_start=0, order_end=10, interaction_order_max=None, eps=1E-3, print_out=False,
        #                      seed=None, do_mp=False, n_cpu=4, dispy=False, dispy_sched_host='localhost',
        #                      random_vars='', hdf5_geo_fn=''):
        #     """
        #     Perform adaptive regression approach based on leave one out cross validation error estimation.
        #
        #     Parameters
        #     ----------
        #     random_vars: list of str
        #         string labels of the random variables
        #     pdf_type: list
        #         type of probability density functions of input parameters,
        #         i.e. ["beta", "norm",...]
        #     pdf_shape: list of lists
        #         shape parameters of probability density functions
        #         s1=[...] "beta": p, "norm": mean
        #         s2=[...] "beta": q, "norm": std
        #         pdf_shape = [s1,s2]
        #     limits: list of lists
        #         upper and lower bounds of random variables (only "beta")
        #         a=[...] "beta": lower bound, "norm": n/a define 0
        #         b=[...] "beta": upper bound, "norm": n/a define 0
        #         limits = [a,b]
        #     func: function
        #         the objective function to be minimized
        #         func(x,*args)
        #     args: tuple, optional, default=()
        #         extra arguments passed to function
        #         i.e. f(x,*args)
        #     fname: str, optional, default=None
        #         if fname exists, reg_obj will be created from it
        #         if not exist, it will be created
        #     order_start: int, optional, default=0
        #         initial gpc expansion order
        #     order_end: int, optional, default=10
        #         maximum gpc expansion order
        #     interaction_order_max: int, optional, defailt=None
        #         define maximum interaction order of parameters
        #         if None, perform all interactions
        #     eps: float, optional, default=1E-3
        #         relative mean error bound of leave one out cross validation
        #     print_out: boolean, optional, default=False
        #         boolean value that determines if to print output the iterations and subiterations
        #     seed: int, optional, default=None
        #         seeding point to replicate random grids
        #     do_mp: boolean, optional, default=False
        #         boolean value that determines if to do each func(x,*args) in each iteration with parmap.starmap(func)
        #     n_cpu: int, optional, default=4
        #         if multiprocessing is enabled, utilize n_cpu cores
        #     dispy: boolean, optional, default=False
        #         boolean value that determines if to compute function with dispy cluster
        #     dispy_sched_host: str, optional, default='localhost'
        #         host name where dispyscheduler will be running
        #     hdf5_geo_fn: str, optional, default=''
        #         hdf5 filename with spatial information: /mesh/elm/*
        #
        #     Returns
        #     -------
        #     gobj: gpc object
        #         gpc object
        #     res: [N_grid x N_out] np.ndarray
        #         function values at grid points of the N_out output variables
        #     """
        #     import pyfempp
        #
        #     def get_skin_surface(mesh_fname):
        #         # load surface data from skin surface
        #         mesh = pyfempp.read_msh(mesh_fname)
        #         points = mesh.nodes.node_coord
        #         triangles = mesh.elm.node_number_list[((mesh.elm.elm_type == 2) & (mesh.elm.tag1 == 1005)), 0:3]
        #         points = np.reshape(points[triangles], (3 * triangles.shape[0], 3))
        #         skin_surface_points = pyfempp.unique_rows(points)
        #
        #         # generate Delaunay grid object of head surface
        #         skin_surface = scipy.spatial.Delaunay(skin_surface_points)
        #
        #         return skin_surface
        #
        #     def get_dispy_cluster(dispy_sched_host, func):
        #         import socket
        #         import dispy
        #         import sys
        #         import time
        #         dispy.MsgTimeout = 90
        #         dispy_schedular_ip = socket.gethostbyname(dispy_sched_host)
        #
        #         # TODO: change if logging is implemented
        #         print_out = True
        #
        #         #  ~/.local/bin/dispyscheduler.py on this machine
        #         #  ~/.local/bin/dispynode.py on any else
        #
        #         if print_out:
        #             print(("Trying to connect to dispyschedular on " + dispy_sched_host))
        #         while True:
        #             try:
        #                 cluster = dispy.SharedJobCluster(func, port=0, scheduler_node=str(dispy_schedular_ip),
        #                                                  reentrant=True)  # loglevel=dispy.logger.DEBUG,
        #                 break
        #             except socket.error:
        #                 time.sleep(1)
        #                 sys.stdout.write('.')
        #                 sys.stdout.flush()
        #
        #         assert cluster
        #
        #         return cluster
        #
        #     try:
        #         # handle input parameters
        #         i_grid = 0
        #         i_iter = 0
        #         dim = len(pdf_type)
        #         order = order_start
        #         run_subiter = True
        #
        #         if not interaction_order_max:
        #             interaction_order_max = dim
        #
        #         config_fname, subject, results_fname, _, _, _ = args
        #
        #         with open(config_fname, 'r') as f:
        #             config = yaml.load(f)
        #
        #         mesh_fn = subject.mesh[config['mesh_idx']]['fn_mesh_msh']
        #
        #         setproctitle.setproctitle("run_reg_adaptive_E_gPC_" + results_fname[-5:])
        #
        #         skin_surface = get_skin_surface(mesh_fn)
        #
        #         # TODO: encapsulate?
        #         if dispy:
        #             cluster = get_dispy_cluster(dispy_sched_host, func)
        #
        #         if fname:
        #             # if .yaml does exist: load from .yaml file
        #             if os.path.exists(fname):
        #                 print(results_fname + ": Loading reg_obj from file: " + fname)
        #                 reg_obj = read_gpc_obj(fname)
        #
        #             # if not: create reg_obj, save to .yaml file
        #             else:
        #                 # re-initialize reg object with appropriate number of grid-points
        #                 N_coeffs = calc_num_coeffs_sparse([order_start] * len(random_vars), order_start, interaction_order_max,
        #                                                   len(random_vars))
        #                 # make initial grid
        #                 grid_init = RandomGrid(pdf_type, pdf_shape, limits, np.ceil(1.2 * N_coeffs))
        #
        #                 # calculate grid
        #                 reg_obj = Reg(pdf_type,
        #                               pdf_shape,
        #                               limits,
        #                               order * np.ones(dim),
        #                               order_max=order,
        #                               interaction_order=interaction_order_max,
        #                               grid=grid_init,
        #                               random_vars=random_vars)
        #
        #                 write_gpc_obj(reg_obj, fname)
        #
        #         else:
        #             # make dummy grid
        #             grid_init = RandomGrid(pdf_type, pdf_shape, limits, 1, seed=seed)
        #
        #             # make initial regobj
        #             reg_obj = Reg(pdf_type,
        #                           pdf_shape,
        #                           limits,
        #                           order * np.ones(dim),
        #                           order_max=order,
        #                           interaction_order=interaction_order_max,
        #                           grid=grid_init,
        #                           random_vars=random_vars)
        #
        #         # run simulations on initial grid
        #         vprint("Iteration #{} (initial grid)".format(i_iter), verbose=reg_obj.verbose)
        #         vprint("=============", verbose=reg_obj.verbose)
        #
        #         # initialize list for resulting arrays
        #         results = [None for _ in range(reg_obj.grid.coords.shape[0])]
        #
        #         # iterate over grid points
        #         for index in range(reg_obj.grid.coords.shape[0]):
        #             # get input parameter for function
        #             x = [index, reg_obj.grid.coords[index, :]]
        #             # evaluate function at grid point
        #             results_func = func(x, *args)
        #
        #             # append result to solution matrix (RHS)
        #             try:
        #                 with h5py.File(results_func, 'r') as hdf:
        #                     hdf_data = hdf['/data/potential'][:]
        #             except:
        #                 print("Fail on " + results_func)
        #                 continue
        #
        #             # append to result array
        #             results.append(hdf_data.flatten())
        #         # create array
        #         results = np.vstack(results)
        #
        #         # increase grid counter by one for next iteration (to not repeat last simulation)
        #         i_grid = i_grid + 1
        #
        #         # perform leave one out cross validation
        #         reg_obj.loocv(results)
        #
        #         vprint("    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]), verbose=reg_obj.verbose)
        #
        #         # main interations (order)
        #         while (reg_obj.relerror_loocv[-1] > eps) and order < order_end:
        #
        #             i_iter = i_iter + 1
        #             order = order + 1
        #
        #             vprint("Iteration #{}".format(i_iter), verbose=reg_obj.verbose)
        #             vprint("=============", verbose=reg_obj.verbose)
        #
        #             # determine new possible polynomials
        #             poly_idx_all_new = get_multi_indices_max_order(dim, order)
        #             poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new, axis=1) == order]
        #             interaction_order_current_max = np.max(poly_idx_all_new)
        #
        #             # reset current interaction order before subiterations
        #             interaction_order_current = 1
        #
        #             # TODO: last working state
        #
        #             # subiterations (interaction orders)
        #             while (interaction_order_current <= interaction_order_current_max) and \
        #                     (interaction_order_current <= interaction_order_max) and \
        #                     run_subiter:
        #                 print("   Subiteration #{}".format(interaction_order_current))
        #                 print("   ================")
        #
        #                 interaction_order_list = np.sum(poly_idx_all_new > 0, axis=1)
        #
        #                 # filter out polynomials of interaction_order = interaction_order_count
        #                 poly_idx_added = poly_idx_all_new[interaction_order_list == interaction_order_current, :]
        #
        #                 # add polynomials to gpc expansion
        #                 reg_obj.enrich_polynomial_basis(poly_idx_added)
        #
        #                 # generate new grid-points
        #                 # reg_obj.enrich_gpc_matrix_samples(1.2)
        #
        #                 if seed:
        #                     seed += 1
        #
        #                 n_g_old = reg_obj.grid.coords.shape[0]
        #
        #                 reg_obj.enrich_gpc_matrix_samples(1.2, seed=seed)
        #
        #                 n_g_new = reg_obj.grid.coords.shape[0]
        #                 # n_g_added = n_g_new - n_g_old
        #
        #                 # check if coil position of new grid points are valid and do not lie inside head
        #                 # TODO: adapt this part to 'x' 'y' 'z' 'psi' 'theta' 'phi'...
        #                 if reg_obj.grid.coords.shape[1] >= 9:
        #                     for i in range(n_g_old, n_g_new):
        #
        #                         valid_coil_position = False
        #
        #                         while not valid_coil_position:
        #                             # get coil transformation matrix
        #                             coil_trans_mat = \
        #                                 pyfempp.calc_coil_transformation_matrix(LOC_mean=positions_mean[0:3, 3],
        #                                                                         ORI_mean=positions_mean[0:3, 0:3],
        #                                                                         LOC_var=reg_obj.grid.coords[i, 4:7],
        #                                                                         ORI_var=reg_obj.grid.coords[i, 7:10],
        #                                                                         V=v)
        #                             # get actual coordinates of magnetic dipole
        #                             dipole_coords = pyfempp.get_coil_dipole_pos(coil_fn, coil_trans_mat)
        #                             valid_coil_position = pyfempp.check_coil_position(dipole_coords, skin_surface)
        #
        #                             # replace bad sample with new one until it works (should actually never be the case)
        #                             if not valid_coil_position:
        #                                 warnings.warn(results_fname +
        #                                               ": Invalid coil position found: " + str(reg_obj.grid.coords[i]))
        #                                 reg_obj.replace_gpc_matrix_samples(idx=np.array(i), seed=seed)
        #
        #                 if do_mp:
        #                     # run repeated simulations
        #                     x = []
        #                     if print_out:
        #                         print(results_fname + \
        #                               "   Performing simulations #{} to {}".format(i_grid + 1,
        #                                                                            reg_obj.grid.coords.shape[0]))
        #                     for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
        #                         x.append([i_grid, reg_obj.grid.coords[i_grid, :]])
        #
        #                     func_part = partial(func,
        #                                         mesh_fn=mesh_fn, tensor_fn=tensor_fn,
        #                                         results_fname=results_fname,
        #                                         coil_fn=coil_fn,
        #                                         POSITIONS_mean=positions_mean,
        #                                         V=v)
        #                     p = NonDaemonicPool(n_cpu)
        #                     results_fns = np.array(p.map(func_part, x))
        #                     p.close()
        #                     p.join()
        #
        #                     # append result to solution matrix (RHS)
        #                     for hdf5_fn in results_fns:
        #                         try:
        #                             with h5py.File(hdf5_fn, 'r') as hdf, h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
        #                                 # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
        #                                 e = hdf['/data/potential'][:]
        #                         except Exception:
        #                             print("Fail on " + hdf5_fn)
        #                         results = np.vstack([results, e.flatten()])
        #                         del e
        #
        #                 elif dispy:
        #                     # compute with dispy cluster
        #                     assert cluster
        #                     if print_out:
        #                         # print("Scheduler connected. Now start dispynodes anywhere in the network")
        #                         print("   Performing simulations #{} to {}".format(i_grid + 1,
        #                                                                            reg_obj.grid.coords.shape[0]))
        #                     # build job list
        #                     jobs = []
        #                     for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
        #                         job = cluster.submit([i_grid, reg_obj.grid.coords[i_grid, :]], *(args))
        #                         job.id = i_grid
        #                         jobs.append(job)
        #
        #                     # get results from single jobs
        #                     results_fns = []
        #                     for job in jobs:
        #                         # res = job()
        #                         results_fns.append(job())
        #                         if print_out:
        #                             # print(str(job.id) + " done in " + str(job.end_time - job.start_time))
        #                             # print(job.stdout)
        #                             if job.exception is not None:
        #                                 print((job.exception))
        #                                 return
        #
        #                     for hdf5_fn in results_fns:
        #                         try:
        #                             with h5py.File(hdf5_fn, 'r') as hdf:  # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
        #                                 # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
        #                                 e = hdf['/data/potential'][:]
        #                         except Exception:
        #                             print("Fail on " + hdf5_fn)
        #                         results = np.vstack([results, e.flatten()])
        #                         del e
        #
        #                 else:  # no multiprocessing
        #                     for i_grid in range(i_grid, reg_obj.grid.coords.shape[0]):
        #                         if print_out:
        #                             print("   Performing simulation #{}".format(i_grid + 1))
        #                         # read conductivities from grid
        #                         x = [i_grid, reg_obj.grid.coords[i_grid, :]]
        #
        #                         # evaluate function at grid point
        #                         results_fn = func(x, *(args))
        #
        #                         # append result to solution matrix (RHS)
        #                         try:
        #                             with h5py.File(results_fn, 'r') as hdf:  # , h5py.File(hdf5_geo_fn, 'r') as hdf_geo:
        #                                 # e = hdf['/mesh/fields/E/value'][:][hdf_geo['/mesh/elm/elm_type'][:] == 2]
        #                                 e = hdf['/data/potential'][:]
        #                         except Exception:
        #                             print("Fail on " + results_fn)
        #                         results = np.vstack([results, e.flatten()])
        #                         del e
        #
        #                 # increase grid counter by one for next iteration (to not repeat last simulation)
        #                 i_grid = i_grid + 1
        #
        #                 # perform leave one out cross validation
        #                 reg_obj.LOOCV(results)
        #                 if print_out:
        #                     print(results_fname + "    -> relerror_LOOCV = {}".format(reg_obj.relerror_loocv[-1]))
        #
        #                 if reg_obj.relerror_loocv[-1] < eps:
        #                     run_subiter = False
        #
        #                 # increase current interaction order
        #                 interaction_order_current += 1
        #
        #         if print_out:
        #             print(results_fname + "DONE ##############################################################")
        #
        #         if dispy:
        #             try:
        #                 cluster.close()
        #             except UnboundLocalError:
        #                 pass
        #
        #         # save gPC object
        #         save_gpcobj(reg_obj, fname)
        #
        #         # save results of forward simulation
        #         np.save(os.path.splitext(fname)[0] + "_res", results)
        #
        #     except:
        #         if dispy:
        #             try:
        #                 cluster.close()
        #             except UnboundLocalError:
        #                 pass
        #         # TODO: print in log file
        #         exc_type, exc_obj, exc_tb = sys.exc_info()
        #         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #         print(exc_type, fname, exc_tb.tb_lineno)
        #         sys.exit()
        #     # return reg_obj
