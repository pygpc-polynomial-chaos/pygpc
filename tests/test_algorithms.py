import os
import sys
import copy
import time
import h5py
import pygpc
import shutil
import unittest
import numpy as np

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

    def test_000_Static_gpc_quad(self):
        """
        Algorithm: Static
        Method: Quadrature
        Solver: NumInt
        Grid: TensorGrid
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_000_Static_gpc_quad'
        print(test_name)

        # define model
        model = pygpc.testfunctions.Peaks()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
        parameters["x2"] = 1.25
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "quad"
        options["solver"] = "NumInt"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["backend"] = "omp"
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # generate grid
        grid = pygpc.TensorGrid(parameters_random=problem.parameters_random,
                                options={"grid_type": ["jacobi", "jacobi"], "n_dim": [9, 9]})

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard")

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=[0],
                                    fn_out=options["fn_results"],
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      n_cpu=session.n_cpu,
                                      output_idx=[0],
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_001_Static_gpc(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_001_Static_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.Peaks()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
        parameters["x2"] = 1.25
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["matrix_ratio"] = 0.7
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["backend"] = "omp"
        options["grid"] = pygpc.Random
        options["grid_options"] = {"seed": seed}
        options["adaptive_sampling"] = True

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard")

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"],
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=session.n_cpu,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_002_MEStatic_gpc(self):
        """
        Algorithm: MEStatic
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_002_MEStatic_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.SurfaceCoverageSpecies()

        # define problem
        parameters = OrderedDict()
        parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])
        parameters["alpha"] = 1.
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["n_grid"] = 500
        options["matrix_ratio"] = None
        options["n_cpu"] = 0
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["grid"] = pygpc.Random
        options["grid_options"] = {"seed": seed}

        # define algorithm
        algorithm = pygpc.MEStatic(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling")

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_003_StaticProjection_gpc(self):
        """
        Algorithm: StaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_003_StaticProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.GenzOscillatory()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [10]
        options["order_max"] = 10
        options["interaction_order"] = 1
        options["n_cpu"] = 0
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["eps"] = 1e-3
        options["error_norm"] = "relative"
        options["matrix_ratio"] = 2
        options["qoi"] = 0
        options["n_grid"] = 5
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["grid"] = pygpc.LHS_L1
        options["grid_options"] = {"criterion": ["tmc", "cc"]}

        # define algorithm
        algorithm = pygpc.StaticProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_004_MEStaticProjection_gpc(self):
        """
        Algorithm: MEStaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_004_MEStaticProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [3, 3]
        options["order_max"] = 3
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["n_cpu"] = 0
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["n_grid"] = 2000
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["grid"] = pygpc.Random
        options["grid_options"] = {"seed": 1}

        # define algorithm
        algorithm = pygpc.MEStaticProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"],
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(5e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_005_RegAdaptive_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_005_RegAdaptive_gpc'
        print(test_name)

        # Model
        model = pygpc.testfunctions.Ishigami()

        # Problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
        parameters["x3"] = 0.
        parameters["a"] = 7.
        parameters["b"] = 0.1

        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["order_start"] = 8
        options["order_end"] = 20
        options["solver"] = "LarsLasso"
        options["interaction_order"] = 2
        options["order_max_norm"] = 1.0
        options["n_cpu"] = 0
        options["adaptive_sampling"] = False
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["eps"] = 0.0075
        # options["grid"] = pygpc.LHS
        # options["grid_options"] = {"criterion": "ese", "seed": seed}

        options["grid"] = pygpc.L1
        options["grid_options"] = {"criterion": ["mc"],
                                   "method": "iter",
                                   "n_iter": 1000,
                                   "seed": seed}

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_006_RegAdaptive_anisotropic_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_006_RegAdaptiveAnisotropic_gpc'
        print(test_name)

        # Model
        model = pygpc.testfunctions.Ishigami()

        # Problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
        parameters["x3"] = 1.  # pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
        parameters["a"] = 7.
        parameters["b"] = 0.1

        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["order_start"] = 0
        options["order_end"] = 20
        options["solver"] = "Moore-Penrose"
        options["interaction_order"] = 2
        options["order_max_norm"] = 1.0
        options["n_cpu"] = 0
        options["adaptive_sampling"] = False
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["eps"] = 0.0075
        options["basis_increment_strategy"] = "anisotropic"
        options["matrix_ratio"] = 2

        options["grid"] = pygpc.Random
        options["grid_options"] = {"seed": seed}

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=1e3)

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_007_RegAdaptiveProjection_gpc(self):
        """
        Algorithm: RegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_006_RegAdaptiveProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.GenzOscillatory()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["order_start"] = 2
        options["order_end"] = 12
        options["interaction_order"] = 2
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["seed"] = 1
        options["matrix_ratio"] = 10
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["adaptive_sampling"] = False
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["n_grid_gradient"] = 5
        options["qoi"] = 0
        options["error_type"] = "nrmsd"
        options["eps"] = 1e-3
        options["grid"] = pygpc.L1
        options["grid_options"] = {"method": "greedy",
                                   "criterion": ["mc"],
                                   "n_pool": 1000,
                                   "seed": seed}

        # define algorithm
        algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None, #options["fn_results"]
                                    folder=None, #"gpc_vs_original_plot"
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=[0],
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_008_MERegAdaptiveProjection_gpc(self):
        """
        Algorithm: MERegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_007_MERegAdaptiveProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order_start"] = 3
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["n_cpu"] = 0
        options["projection"] = True
        options["adaptive_sampling"] = True
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
        options["error_type"] = "nrmsd"
        options["error_norm"] = "absolute"
        options["n_samples_validations"] = "absolute"
        options["qoi"] = 0
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["n_samples_discontinuity"] = 12
        options["eps"] = 0.75
        options["n_grid_init"] = 100
        options["backend"] = "omp"
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=[0],
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"],
                                      folder="gpc_vs_original_mc",
                                      plot=plot)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=[0, 1],
                                    fn_out=options["fn_results"],
                                    folder="gpc_vs_original_plot",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_020_clustering_3_domains(self):
        """
        Algorithm: MERegAdaptiveprojection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_022_clustering_3_domains'
        print(test_name)

        # define model
        model = pygpc.testfunctions.Cluster3Simple()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso" #"Moore-Penrose"
        options["settings"] = None
        options["order_start"] = 1
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["matrix_ratio"] = 1
        options["projection"] = False
        options["n_cpu"] = 0
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "FD_fwd"
        options["error_type"] = "loocv"
        options["error_norm"] = "absolute" # "relative"
        options["qoi"] = 0 # "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 3,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["n_samples_discontinuity"] = 50
        options["adaptive_sampling"] = False
        options["eps"] = 0.01
        options["n_grid_init"] = 500
        options["backend"] = "omp"
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=[0],
                                    fn_out=options["fn_results"],
                                    folder="validate_gpc_plot",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"],
                                      folder="validate_gpc_mc",
                                      plot=plot)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=int(1e3))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=problem.parameters_random,
                               fn_out=options["fn_results"] + ".txt")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_024_Static_IO_gpc(self):
        """
        Algorithm: Static_IO
        Method: Regression
        Solver: LarsLasso
        Grid: Custom (Random)
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_026_Static_IO_gpc'
        print(test_name)

        # define input data
        np.random.seed(1)
        n_grid = 100
        x1 = np.random.rand(n_grid) * 0.8 + 1.2
        x2 = 1.25
        x3 = np.random.rand(n_grid) * 0.6

        # define random variables
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])

        # generate grid from it for gpc
        grid = pygpc.RandomGrid(parameters_random=parameters, coords=np.vstack((x1,x3)).T)

        # get output data (Peaks function)
        results = (3.0 * (1 - x1) ** 2. * np.exp(-(x1 ** 2) - (x3 + 1) ** 2)
                   - 10.0 * (x1 / 5.0 - x1 ** 3 - x3 ** 5)
                   * np.exp(-x1 ** 2 - x3 ** 2) - 1.0 / 3
                   * np.exp(-(x1 + 1) ** 2 - x3 ** 2)) + x2
        results = results[:, np.newaxis]

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["error_type"] = "loocv"
        options["n_samples_validation"] = None
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["backend"] = "omp"
        options["verbose"] = True

        # define algorithm
        algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard")

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=parameters,
                               fn_out=options["fn_results"] + ".txt")

        self.expect_true(session.gpc[0].error[0] < 0.001,
                         f'gPC test failed with LOOCV error = {session.gpc[0].error[0]}')

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_025_MEStatic_IO_gpc(self):
        """
        Algorithm: MEStatic_IO
        Method: Regression
        Solver: LarsLasso
        Grid: Custom (Random)
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_027_MEStatic_IO_gpc'
        print(test_name)

        # define input data
        np.random.seed(1)
        n_grid = 3000
        rho_0 = np.random.rand(n_grid)
        beta = np.random.rand(n_grid) * 20.
        alpha = 1.

        # define random variables
        parameters = OrderedDict()
        parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])

        # generate grid from it for gpc
        grid = pygpc.RandomGrid(parameters_random=parameters, coords=np.vstack((rho_0,beta)).T)

        # get output data (SurfaceCoverageSpecies function)
        # System of 1st order DEQ
        def deq(rho, t, alpha, beta, gamma):
            return alpha * (1. - rho) - gamma * rho - beta * (rho - 1) ** 2 * rho

        # Constants
        gamma = 0.01

        # Simulation parameters
        dt = 0.01
        t_end = 1.
        t = np.arange(0, t_end, dt)

        # Solve
        results = odeint(deq, rho_0, t, args=(alpha, beta, gamma))[-1][:, np.newaxis]

        # gPC options
        options = dict()
        options["solver"] = "LarsLasso"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["matrix_ratio"] = None
        options["n_cpu"] = 0
        options["error_type"] = "loocv"
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = save_session_format
        options["verbose"] = True

        # define algorithm
        algorithm = pygpc.MEStatic_IO(parameters=parameters, options=options, grid=grid, results=results)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # read session
        session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard",
                                     n_samples=int(1e4))

        pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                               parameters_random=parameters,
                               fn_out=options["fn_results"] + ".txt")

        self.expect_true(session.gpc[0].error[0] < 0.075,
                         f'gPC test failed with LOOCV error = {session.gpc[0].error[0]}')

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")


if __name__ == '__main__':
    unittest.main()
