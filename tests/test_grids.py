import os
import sys
import copy
import time
import h5py
import pygpc
import shutil
import unittest
import numpy as np

import matplotlib
matplotlib.use("Agg")

from scipy.integrate import odeint
from collections import OrderedDict

# disable numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# test options
folder = 'tmp'                  # output folder
plot = False                    # plot and save output
matlab = False                  # test Matlab functionality
save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
seed = 1                        # random seed for grids

# temporary folder
try:
    os.mkdir(folder)
except FileExistsError:
    pass


class TestPygpcMethods(unittest.TestCase):

    # setup method called before every test-case
    def setUp(self):
        pass

    def run(self, result=None):
        self._result = result
        self._num_expectations = 0
        super(TestPygpcMethods, self).run(result)

    def _fail(self, failure):
        try:
            raise failure
        except failure.__class__:
            self._result.addFailure(self, sys.exc_info())

    def expect_isclose(self, a, b, msg='', atol=None, rtol=None):
        if atol is None:
            atol = 1.e-8
        if rtol is None:
            rtol = 1.e-5

        if not np.isclose(a, b, atol=atol, rtol=rtol).all():
            msg = '({}) Expected {} to be close {}. '.format(self._num_expectations, a, b) + msg
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def expect_equal(self, a, b, msg=''):
        if a != b:
            msg = '({}) Expected {} to equal {}. '.format(self._num_expectations, a, b) + msg
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def expect_true(self, a, msg=''):
        if not a:
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def test_grids_001_quadrature_grids(self):
        """
        Testing Grids [TensorGrid, SparseGrid]
        """
        global folder, plot
        test_name = 'test_grids_001_quadrature_grids'
        print(test_name)

        # define testfunction
        test = pygpc.Peaks()

        # TensorGrid
        grid_tensor_1 = pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
                                         options={"grid_type": ["hermite", "jacobi"], "n_dim": [5, 10]})

        grid_tensor_2 = pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
                                         options={"grid_type": ["patterson", "fejer2"], "n_dim": [3, 10]})

        # SparseGrid
        grid_sparse = pygpc.SparseGrid(parameters_random=test.problem.parameters_random,
                                       options={"grid_type": ["jacobi", "jacobi"],
                                                "level": [3, 3],
                                                "level_max": 3,
                                                "interaction_order": 2,
                                                "order_sequence_type": "exp"})

        print("done!\n")

    def test_grids_002_random_grid(self):
        """
        Testing Grids [Random]
        """
        global folder, plot, seed
        test_name = 'test_grids_002_random_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        parameters_2 = OrderedDict()
        parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
        parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
        parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_2 = pygpc.Problem(model, parameters_2)

        n_grid = 100
        n_grid_extend = 10

        # generate grid w/o percentile constraint
        #########################################
        # initialize grid
        grid = pygpc.Random(parameters_random=problem_1.parameters_random,
                            n_grid=n_grid,
                            options={"seed": seed})
        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")

        # extend grid
        for i in range(2):
            grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
            self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
                             f"Size of random grid does not fit after extending it {i+1}. time.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i+1}. time.")

        # generate grid with percentile constraint
        ##########################################
        # initialize grid
        grid = pygpc.Random(parameters_random=problem_2.parameters_random,
                            n_grid=n_grid,
                            options={"seed": seed})

        perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

        for i_p, p in enumerate(problem_2.parameters_random):
            perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                              (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
        self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")

        # extend grid
        for i in range(2):
            grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)

            perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

            for i_p, p in enumerate(problem_2.parameters_random):
                perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                  (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

            self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
            self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                             "Size of random grid does not fit after extending it.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")

        # perform static gpc
        ###############################
        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["grid"] = pygpc.Random
        options["grid_options"] = {"seed": seed}
        options["matrix_ratio"] = None
        options["n_grid"] = 100
        options["order_start"] = 3
        options["order_end"] = 15
        options["eps"] = 0.001
        options["adaptive_sampling"] = False

        # define algorithm
        algorithm = pygpc.Static(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")

        # perform adaptive gpc
        ##############################
        options["matrix_ratio"] = 2

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")

        print("done!\n")

    def test_grids_003_LHS_grid(self):
        """
        Testing Grids [LHS]
        """
        global folder, plot, seed
        test_name = 'test_grids_003_LHS_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        parameters_2 = OrderedDict()
        parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.7)
        parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.7)
        parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_2 = pygpc.Problem(model, parameters_2)

        n_grid_extend = 10

        options_dict = {"ese": {"criterion": "ese", "seed": seed},
                        "maximin": {"criterion": "maximin", "seed": seed},
                        "corr": {"criterion": "corr", "seed": seed},
                        "standard": {"criterion": None, "seed": seed}}

        for i_c, c in enumerate(options_dict):
            print(f"- criterion: {c} -")

            # generate grid w/o percentile constraint
            #########################################
            print("- generate grid w/o percentile constraint -")
            n_grid = 20

            # initialize grid
            print("  > init")
            grid = pygpc.LHS(parameters_random=problem_1.parameters_random,
                             n_grid=n_grid,
                             options=options_dict[c])
            self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")

            # extend grid
            print("  > extend")
            for i in range(2):
                grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
                self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                                 f"Size of random grid does not fit after extending it {i + 1}. time.")
                self.expect_true(pygpc.get_different_rows_from_matrices(
                    grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                                 f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")

            # generate grid with percentile constraint
            ##########################################
            print("- generate grid with percentile constraint -")
            print("  > init")
            n_grid = 100

            # initialize grid
            grid = pygpc.LHS(parameters_random=problem_2.parameters_random,
                             n_grid=n_grid,
                             options=options_dict[c])

            perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

            for i_p, p in enumerate(problem_2.parameters_random):
                perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                  (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

            self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
            self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")

            # extend grid
            print("  > extend")
            for i in range(2):
                grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)

                perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

                for i_p, p in enumerate(problem_2.parameters_random):
                    perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                      (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

                self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
                self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                                 "Size of random grid does not fit after extending it.")
                self.expect_true(pygpc.get_different_rows_from_matrices(
                    grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                                 f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")

            # perform static gpc
            ###############################
            print("- Static gpc -")
            # gPC options
            options = dict()
            options["method"] = "reg"
            options["solver"] = "LarsLasso"
            options["settings"] = None
            options["order"] = [7, 7, 7]
            options["order_max"] = 7
            options["interaction_order"] = 2
            options["error_type"] = "nrmsd"
            options["n_samples_validation"] = 1e3
            options["n_cpu"] = 0
            options["fn_results"] = None
            options["save_session_format"] = save_session_format
            options["gradient_enhanced"] = False
            options["gradient_calculation"] = "FD_1st2nd"
            options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
            options["backend"] = "omp"
            options["grid"] = pygpc.LHS
            options["grid_options"] = options_dict[c]
            options["matrix_ratio"] = None
            options["n_grid"] = 100
            options["order_start"] = 3
            options["order_end"] = 15
            options["eps"] = 0.001
            options["adaptive_sampling"] = False

            # define algorithm
            algorithm = pygpc.Static(problem=problem_1, options=options)

            # Initialize gPC Session
            session = pygpc.Session(algorithm=algorithm)

            # run gPC algorithm
            session, coeffs, results = session.run()

            self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")

            # perform adaptive gpc
            ##############################
            print("- Adaptive gpc -")
            options["matrix_ratio"] = 2
            options["n_grid"] = None

            # define algorithm
            algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

            # Initialize gPC Session
            session = pygpc.Session(algorithm=algorithm)

            # run gPC algorithm
            session, coeffs, results = session.run()

            self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")

        print("done!\n")

    def test_grids_004_L1_grid(self):
        """
        Testing Grids [L1]
        """
        global folder, plot, seed
        test_name = 'test_grids_004_L1_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        parameters_2 = OrderedDict()
        parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
        parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
        parameters_2["x3"] = pygpc.Norm(pdf_shape=[3, 0.5], p_perc=0.5)
        problem_2 = pygpc.Problem(model, parameters_2)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order_start"] = 3
        options["order_end"] = 15
        options["order"] = [7, 7, 7]
        options["order_max"] = 7
        options["order_max_norm"] = 1
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["eps"] = 0.0075
        options["adaptive_sampling"] = False
        options["grid"] = pygpc.L1

        n_grid_extend = 10
        method_list = ["greedy", "iteration"]
        # method_list = ["iteration"]
        criterion_list = [["mc"], ["mc", "cc"], ["D"], ["D-coh"]]
        # criterion_list = [["D-coh"]]

        grid_options = {"method": None,
                        "criterion": None,
                        "n_pool": 500,
                        "n_iter": 10,
                        "seed": seed}

        for i_m, method in enumerate(method_list):
            print(f"- method: {method} -")

            for i_c, criterion in enumerate(criterion_list):
                print(f"- criterion: {criterion} -")
                grid_options["criterion"] = criterion
                grid_options["method"] = method
                grid_options["n_iter"] = 10
                options["grid_options"] = grid_options

                # generate grid w/o percentile constraint
                #########################################
                print("- generate grid w/o percentile constraint -")
                print("  > init")
                n_grid = 20
                grid_options["n_pool"] = 5 * n_grid

                # create gpc object of some order for problem_1
                gpc = pygpc.Reg(problem=problem_1,
                                order=options["order"],
                                order_max=options["order_max"],
                                order_max_norm=options["order_max_norm"],
                                interaction_order=options["interaction_order"],
                                interaction_order_current=options["interaction_order"],
                                options=options,
                                validation=None)

                # initialize grid
                grid = pygpc.L1(parameters_random=problem_1.parameters_random,
                                n_grid=n_grid,
                                options=grid_options,
                                gpc=gpc)

                self.expect_true(grid.n_grid == n_grid, f"Size of random grid does not fit after initialization. "
                                                        f"({criterion, method})")

                # extend grid
                print("  > extend")
                for i in range(2):
                    grid_pre = copy.deepcopy(grid)
                    grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
                    self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                                     f"Size of random grid does not fit after extending it {i + 1}. time. "
                                     f"({criterion, method})")
                    self.expect_true(pygpc.get_different_rows_from_matrices(
                        grid_pre.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                                     f"Extended grid points are not matching the initial grid after extending it {i + 1}. time. "
                                     f"({criterion, method})")

                # generate grid with percentile constraint
                ##########################################
                print("- generate grid with percentile constraint -")
                print("  > init")
                n_grid = 20
                grid_options["n_pool"] = 2 * n_grid
                grid_options["n_iter"] = 10

                # create gpc object of some order for problem_2
                gpc = pygpc.Reg(problem=problem_2,
                                order=options["order"],
                                order_max=options["order_max"],
                                order_max_norm=options["order_max_norm"],
                                interaction_order=options["interaction_order"],
                                interaction_order_current=options["interaction_order"],
                                options=options,
                                validation=None)

                # initialize grid
                grid = pygpc.L1(parameters_random=problem_2.parameters_random,
                                n_grid=n_grid,
                                options=grid_options,
                                gpc=gpc)

                perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

                for i_p, p in enumerate(problem_2.parameters_random):
                    perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                      (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

                self.expect_true(grid.n_grid == n_grid,
                                 f"Size of random grid does not fit after initialization. "
                                 f"({criterion, method})")
                self.expect_true(perc_check.all(),
                                 f"Grid points do not fulfill percentile constraint. "
                                 f"({criterion, method})")

                # extend grid
                print("  > extend")
                for i in range(2):
                    grid_pre = copy.deepcopy(grid)
                    grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)

                    perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

                    for i_p, p in enumerate(problem_2.parameters_random):
                        perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[
                            0]).all() and \
                                          (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

                    self.expect_true(perc_check.all(), f"Grid points do not fulfill percentile constraint. "
                                                       f"({criterion, method})")
                    self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                                     f"Size of random grid does not fit after extending it.  "
                                     f"({criterion, method})")
                    self.expect_true(pygpc.get_different_rows_from_matrices(
                        grid_pre.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                                     f"Extended grid points are matching the initial grid after extending it {i + 1}. time. "
                                     f"({criterion, method})")

                # perform static gpc
                ###############################
                print("  > Perform Static gpc")
                options["n_grid"] = None
                options["matrix_ratio"] = 1.5
                grid_options["n_pool"] = 500
                grid_options["n_iter"] = 10
                options["grid_options"] = grid_options

                # define algorithm
                algorithm = pygpc.Static(problem=problem_1, options=options)

                # Initialize gPC Session
                session = pygpc.Session(algorithm=algorithm)

                # run gPC algorithm
                session, coeffs, results = session.run()

                self.expect_true(session.gpc[0].error[0] <= options["eps"], f"Error of static gpc too high. "
                                                                            f"({criterion, method})")

                # perform adaptive gpc
                ##############################
                print("  > Perform Adaptive gpc")
                # define algorithm
                algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

                # Initialize gPC Session
                session = pygpc.Session(algorithm=algorithm)

                # run gPC algorithm
                session, coeffs, results = session.run()

                self.expect_true(session.gpc[0].error[-1] <= options["eps"], f"Error of adaptive gpc too high. "
                                                                             f"({criterion, method})")

        print("done!\n")

    def test_grids_005_FIM_grid(self):
        """
        Testing Grids [FIM]
        """
        global folder, plot, seed
        test_name = 'pygpc_test_016_FIM_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        parameters_2 = OrderedDict()
        parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
        parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
        parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_2 = pygpc.Problem(model, parameters_2)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order_start"] = 2
        options["order_end"] = 5
        options["order"] = [3, 3, 3]
        options["order_max"] = 3
        options["order_max_norm"] = 1
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["eps"] = 0.05
        options["adaptive_sampling"] = False
        options["grid"] = pygpc.FIM
        options["grid_options"] = {"seed": seed, "n_pool": 100}

        n_grid_extend = 10

        # generate grid w/o percentile constraint
        #########################################
        print("- generate grid w/o percentile constraint -")
        print("  > init")
        n_grid = 20

        # create gpc object of some order for problem_1
        gpc = pygpc.Reg(problem=problem_1,
                        order=[2, 2, 2],
                        order_max=2,
                        order_max_norm=1,
                        interaction_order=2,
                        interaction_order_current=2,
                        options=options,
                        validation=None)

        # initialize grid
        grid = pygpc.FIM(parameters_random=problem_1.parameters_random,
                         n_grid=n_grid,
                         options=options["grid_options"],
                         gpc=gpc)

        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")

        # extend grid
        print("  > extend")
        for i in range(2):
            grid_pre = copy.deepcopy(grid)
            grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
            self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
                             f"Size of random grid does not fit after extending it {i+1}. time.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid_pre.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i+1}. time.")

        # generate grid with percentile constraint
        ##########################################
        print("- generate grid with percentile constraint -")
        print("  > init")
        n_grid = 50

        # create gpc object of some order for problem_2
        gpc = pygpc.Reg(problem=problem_2,
                        order=[2, 2, 2],
                        order_max=2,
                        order_max_norm=1,
                        interaction_order=2,
                        interaction_order_current=2,
                        options=options,
                        validation=None)

        # initialize grid
        grid = pygpc.FIM(parameters_random=problem_2.parameters_random,
                         n_grid=n_grid,
                         options=options["grid_options"],
                         gpc=gpc)

        perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

        for i_p, p in enumerate(problem_2.parameters_random):
            perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                              (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

        self.expect_true(grid.n_grid == n_grid,
                         "Size of random grid does not fit after initialization.")
        self.expect_true(perc_check.all(),
                         "Grid points do not fulfill percentile constraint.")

        # extend grid
        print("  > extend")
        for i in range(2):
            grid_pre = copy.deepcopy(grid)
            grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)

            perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

            for i_p, p in enumerate(problem_2.parameters_random):
                perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                  (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

            self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
            self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                             "Size of random grid does not fit after extending it.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid_pre.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")

        # perform static gpc
        ###############################
        print("- Perform Static gpc -")
        options["n_grid"] = None
        options["matrix_ratio"] = 1.5

        # define algorithm
        algorithm = pygpc.Static(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[0] <= 0.05, "Error of static gpc too high.")

        # perform adaptive gpc
        ###############################
        print("- Perform Adaptive gpc -")
        options["grid_options"]["n_pool"] = 100

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[-1] <= 0.05, "Error of adaptive gpc too high.")

        print("done!\n")

    def test_grids_006_CO_grid(self):
        """
        Testing Grids [CO]
        """
        global folder, plot, seed
        test_name = 'pygpc_test_018_CO_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order_start"] = 2
        options["order_end"] = 5
        options["order"] = [3, 3, 3]
        options["order_max"] = 3
        options["order_max_norm"] = 1
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["eps"] = 0.01
        options["adaptive_sampling"] = False
        options["grid"] = pygpc.CO
        options["grid_options"] = {"seed": seed, "n_warmup": 10, "n_pool": 300}

        n_grid = 100
        n_grid_extend = 10

        # generate grid w/o percentile constraint
        #########################################
        print("- generate grid w/o percentile constraint -")
        print("  > init")

        # create gpc object of some order for problem_1
        gpc = pygpc.Reg(problem=problem_1,
                        order=[2, 2, 2],
                        order_max=2,
                        order_max_norm=1,
                        interaction_order=2,
                        interaction_order_current=2,
                        options=options,
                        validation=None)

        # initialize grid
        grid = pygpc.CO(parameters_random=problem_1.parameters_random,
                        n_grid=n_grid,
                        gpc=gpc,
                        options=options["grid_options"])
        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")

        # extend grid
        print("  > extend")
        for i in range(2):
            grid_pre = copy.deepcopy(grid)
            grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
            self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
                             f"Size of random grid does not fit after extending it {i+1}. time.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid_pre.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i+1}. time.")

        # perform static gpc
        ###############################
        print("- Perform Static gpc -")

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["grid"] = pygpc.CO
        options["grid_options"] = {"seed": seed, "n_warmup": 10, "n_pool": 300}
        options["matrix_ratio"] = None
        options["n_grid"] = 150
        options["order_start"] = 3
        options["order_end"] = 11
        options["eps"] = 0.01
        options["adaptive_sampling"] = False

        # define algorithm
        algorithm = pygpc.Static(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[0] <= options["eps"], f"Error of static gpc too high "
                                                                    f"({session.gpc[0].error[0]} > {options['eps']}).")

        # perform adaptive gpc
        ##############################
        print("- Perform Adaptive gpc -")

        options["matrix_ratio"] = 2

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[-1] <= options["eps"], "Error of adaptive gpc too high.")

        print("done!\n")

    def test_grids_007_GP_grid(self):
        """
        Testing Grids [GP]
        """
        global folder, plot, seed
        test_name = 'pygpc_test_007_GP_grid'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters_1 = OrderedDict()
        parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters_1["x2"] = 1.
        parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_1 = pygpc.Problem(model, parameters_1)

        parameters_2 = OrderedDict()
        parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
        parameters_2["x2"] = 1.
        parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem_2 = pygpc.Problem(model, parameters_2)

        n_grid = 4
        n_grid_extend = 2

        # generate grid w/o percentile constraint
        #########################################
        # initialize grid
        grid = pygpc.GP(parameters_random=problem_1.parameters_random,
                            n_grid=n_grid,
                            options={"seed": seed, "n_pool": 1000})
        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")

        # extend grid
        for i in range(2):
            grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
            self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
                             f"Size of random grid does not fit after extending it {i+1}. time.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i+1}. time.")

        # generate grid with percentile constraint
        ##########################################
        # initialize grid
        grid = pygpc.Random(parameters_random=problem_2.parameters_random,
                            n_grid=n_grid,
                            options={"seed": seed, "n_pool": 1000})

        perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

        for i_p, p in enumerate(problem_2.parameters_random):
            perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                              (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

        self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
        self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")

        # extend grid
        for i in range(2):
            grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)

            perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)

            for i_p, p in enumerate(problem_2.parameters_random):
                perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
                                  (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()

            self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
            self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
                             "Size of random grid does not fit after extending it.")
            self.expect_true(pygpc.get_different_rows_from_matrices(
                grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
                             f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")

        # perform static gpc
        ###############################
        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = None
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["grid"] = pygpc.GP
        options["grid_options"] = {"seed": seed, "n_pool": 1000}
        options["matrix_ratio"] = None
        options["n_grid"] = 50
        options["order_start"] = 3
        options["order_end"] = 15
        options["eps"] = 0.0075
        options["adaptive_sampling"] = False

        # define algorithm
        algorithm = pygpc.Static(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[0] <= 0.0075, "Error of static gpc too high.")

        # perform adaptive gpc
        ##############################
        options["matrix_ratio"] = 2

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        self.expect_true(session.gpc[0].error[-1] <= 0.0075, "Error of adaptive gpc too high.")

        print("done!\n")

    def test_grids_008_seed_grids_reproducibility(self):
        """
        Test reproducibility of grids when seeding
        """
        global folder, plot, matlab, save_session_format
        test_name = 'pygpc_test_019_seed_grids_reproducibility'
        print(test_name)

        # define testfunction
        model = pygpc.testfunctions.Peaks()

        # define problems
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem = pygpc.Problem(model, parameters)

        gpc = pygpc.Reg(problem=problem, order_max=2)

        # Random
        print("Testing reproducibility of Random grid ...")
        grid = [0 for _ in range(2)]
        for i in range(2):
            # initialize grid
            grid[i] = pygpc.Random(parameters_random=problem.parameters_random,
                                   n_grid=10,
                                   options={"seed": 1})

            # extend grid
            grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)

        # compare
        self.expect_true(np.isclose(grid[0].coords_norm, grid[1].coords_norm).all(),
                         "Random grid is not reproducible when seeding")

        # LHS
        print("Testing reproducibility of LHS grids ...")
        criterion_list = [None, "maximin", "ese"]

        for criterion in criterion_list:
            print(f"\t > criterion: {criterion}")
            grid = [0 for _ in range(2)]

            for i in range(2):
                # initialize grid
                grid[i] = pygpc.LHS(parameters_random=problem.parameters_random,
                                    n_grid=10,
                                    options={"criterion": criterion,
                                             "seed": 1})

                # extend grid
                grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)

            # compare
            self.expect_true(np.isclose(grid[0].coords_norm, grid[1].coords_norm).all(),
                             f"LHS ({criterion}) grid is not reproducible when seeding")

        # L1
        print("Testing reproducibility of L1 grids ...")
        criterion_list = [["mc"], ["tmc", "cc"], ["D"], ["D-coh"]]
        method_list = ["greedy", "iter"]

        for criterion in criterion_list:
            print(f"\t > criterion: {criterion}")
            for method in method_list:
                print(f"\t\t > method: {method}")
                grid = [0 for _ in range(2)]

                for i in range(2):
                    # initialize grid
                    grid[i] = pygpc.L1(parameters_random=problem.parameters_random,
                                       n_grid=10,
                                       options={"criterion": criterion,
                                                "method": method,
                                                "seed": 1,
                                                "n_pool": 100,
                                                "n_iter": 100},
                                       gpc=gpc)

                    # extend grid
                    grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)

                # compare
                self.expect_true((np.isclose(grid[0].coords_norm, grid[1].coords_norm)).all(),
                                 f"L1 ({criterion}, {method}) grid is not reproducible when seeding")

        # FIM
        print("Testing reproducibility of FIM grid ...")
        grid = [0 for _ in range(2)]
        for i in range(2):
            # initialize grid
            grid[i] = pygpc.FIM(parameters_random=problem.parameters_random,
                                n_grid=10,
                                options={"seed": 1},
                                gpc=gpc)

            # extend grid
            grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)

        # compare
        self.expect_true(np.isclose(grid[0].coords_norm, grid[1].coords_norm).all(),
                         "FIM grid is not reproducible when seeding")

        # CO
        print("Testing reproducibility of CO grid ...")
        grid = [0 for _ in range(2)]
        for i in range(2):
            # initialize grid
            grid[i] = pygpc.CO(parameters_random=problem.parameters_random,
                               n_grid=10,
                               options={"seed": 1, "n_warmup": 10},
                               gpc=gpc)

            # extend grid
            grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)

        # compare
        self.expect_true(np.isclose(grid[0].coords_norm, grid[1].coords_norm).all(),
                         "CO grid is not reproducible when seeding")


if __name__ == '__main__':
    unittest.main()
