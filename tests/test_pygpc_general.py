import numpy as np
import unittest
import shutil
import pygpc
import time
import h5py
import sys
import os
import numpy as np
from collections import OrderedDict

# disable numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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

    # def test_000_Static_gpc_quad(self):
    #     """
    #     Algorithm: Static
    #     Method: Quadrature
    #     Solver: NumInt
    #     Grid: TensorGrid
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_000_Static_gpc_quad'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "quad"
    #     options["solver"] = "NumInt"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["backend"] = "omp"
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     # generate grid
    #     grid = pygpc.TensorGrid(parameters_random=problem.parameters_random,
    #                             options={"grid_type": ["jacobi", "jacobi"], "n_dim": [9, 9]})
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="standard",
    #                                  n_samples=1e3)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=[0],
    #                                 fn_out=options["fn_results"],
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   n_cpu=session.n_cpu,
    #                                   output_idx=[0],
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #     print("> Checking file consistency...")
    #
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_001_Static_gpc(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_001_Static_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 0.7
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = {"seed": seed}
    #     options["adaptive_sampling"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="standard",
    #                                  n_samples=1e3)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=options["fn_results"],
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=session.n_cpu,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")

    # def test_002_MEStatic_gpc(self):
    #     """
    #     Algorithm: MEStatic
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_002_MEStatic_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.SurfaceCoverageSpecies()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])
    #     parameters["alpha"] = 1.
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["n_grid"] = 500
    #     options["matrix_ratio"] = None
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["qoi"] = "all"
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 2,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = {"seed": seed}
    #
    #     # define algorithm
    #     algorithm = pygpc.MEStatic(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=None,
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e3)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=False,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_003_StaticProjection_gpc(self):
    #     """
    #     Algorithm: StaticProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_003_StaticProjection_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzOscillatory()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order"] = [10]
    #     options["order_max"] = 10
    #     options["interaction_order"] = 1
    #     options["n_cpu"] = 0
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["eps"] = 1e-3
    #     options["error_norm"] = "relative"
    #     options["matrix_ratio"] = 2
    #     options["qoi"] = 0
    #     options["n_grid"] = 5
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["grid"] = pygpc.LHS_L1
    #     options["grid_options"] = {"criterion": ["tmc", "cc"]}
    #
    #     # define algorithm
    #     algorithm = pygpc.StaticProjection(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=None,
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e3)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=False,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_004_MEStaticProjection_gpc(self):
    #     """
    #     Algorithm: MEStaticProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_004_MEStaticProjection_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order"] = [3, 3]
    #     options["order_max"] = 3
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["n_grid"] = 100
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["qoi"] = "all"
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 2,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     # define algorithm
    #     algorithm = pygpc.MEStaticProjection(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC session
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=options["fn_results"],
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e3)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(5e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=True,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_005_RegAdaptive_gpc(self):
    #     """
    #     Algorithm: RegAdaptive
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_005_RegAdaptive_gpc'
    #     print(test_name)
    #
    #     # Model
    #     model = pygpc.testfunctions.Ishigami()
    #
    #     # Problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    #     parameters["x3"] = 0.
    #     parameters["a"] = 7.
    #     parameters["b"] = 0.1
    #
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 8
    #     options["order_end"] = 20
    #     options["solver"] = "LarsLasso"
    #     options["interaction_order"] = 2
    #     options["order_max_norm"] = 1.0
    #     options["n_cpu"] = 0
    #     options["adaptive_sampling"] = False
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["eps"] = 0.0075
    #     # options["grid"] = pygpc.LHS
    #     # options["grid_options"] = {"criterion": "ese", "seed": seed}
    #
    #     options["grid"] = pygpc.L1
    #     options["grid_options"] = {"criterion": ["mc"],
    #                                "method": "iter",
    #                                "n_iter": 1000,
    #                                "seed": seed}
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC session
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=None,
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e3)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=True,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")

    # def test_006_RegAdaptiveProjection_gpc(self):
    #     """
    #     Algorithm: RegAdaptiveProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_006_RegAdaptiveProjection_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzOscillatory()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 12
    #     options["interaction_order"] = 2
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 10
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["adaptive_sampling"] = False
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["n_grid_gradient"] = 5
    #     options["qoi"] = 0
    #     options["error_type"] = "nrmsd"
    #     options["eps"] = 1e-3
    #     options["grid"] = pygpc.L1
    #     options["grid_options"] = {"method": "iter",
    #                                "criterion": ["mc"],
    #                                "n_iter": 1000,
    #                                "seed": seed}
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC session
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=None, #options["fn_results"]
    #                                 folder=None, #"gpc_vs_original_plot"
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e3)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=False,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")

    # def test_007_MERegAdaptiveProjection_gpc(self):
    #     """
    #     Algorithm: MERegAdaptiveProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_007_MERegAdaptiveProjection_gpc'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 3
    #     options["order_end"] = 15
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["projection"] = True
    #     options["adaptive_sampling"] = True
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #     options["error_type"] = "nrmsd"
    #     options["error_norm"] = "absolute"
    #     options["n_samples_validations"] = "absolute"
    #     options["qoi"] = 0
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 2,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["n_samples_discontinuity"] = 12
    #     options["eps"] = 0.75
    #     options["n_grid_init"] = 100
    #     options["backend"] = "omp"
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     # define algorithm
    #     algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC session
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=[0],
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=True,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=[0, 1],
    #                                 fn_out=options["fn_results"],
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e4)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_008_testfunctions(self):
    #     """
    #     Testing testfunctions (multi-threading and inherited parallelization)
    #     """
    #     test_name = 'pygpc_test_008_testfunctions'
    #     print(test_name)
    #
    #     tests = []
    #     tests.append(pygpc.Ackley())
    #     tests.append(pygpc.BukinFunctionNumber6())
    #     tests.append(pygpc.CrossinTrayFunction())
    #     tests.append(pygpc.BohachevskyFunction1())
    #     tests.append(pygpc.PermFunction())
    #     tests.append(pygpc.SixHumpCamelFunction())
    #     tests.append(pygpc.RotatedHyperEllipsoid())
    #     tests.append(pygpc.SumOfDifferentPowersFunction())
    #     tests.append(pygpc.ZakharovFunction())
    #     tests.append(pygpc.DropWaveFunction())
    #     tests.append(pygpc.DixonPriceFunction())
    #     tests.append(pygpc.RosenbrockFunction())
    #     tests.append(pygpc.MichalewiczFunction())
    #     tests.append(pygpc.DeJongFunctionFive())
    #     tests.append(pygpc.MatyasFunction())
    #     tests.append(pygpc.GramacyLeeFunction())
    #     tests.append(pygpc.SchafferFunction4())
    #     tests.append(pygpc.SphereFunction())
    #     tests.append(pygpc.McCormickFunction())
    #     tests.append(pygpc.BoothFunction())
    #     tests.append(pygpc.Peaks())
    #     tests.append(pygpc.Franke())
    #     tests.append(pygpc.Lim2002())
    #     tests.append(pygpc.Ishigami())
    #     tests.append(pygpc.ManufactureDecay())
    #     tests.append(pygpc.GenzContinuous())
    #     tests.append(pygpc.GenzCornerPeak())
    #     tests.append(pygpc.GenzOscillatory())
    #     tests.append(pygpc.GenzProductPeak())
    #     tests.append(pygpc.Ridge())
    #     tests.append(pygpc.OakleyOhagan2004())
    #     tests.append(pygpc.Welch1992())
    #     tests.append(pygpc.HyperbolicTangent())
    #     tests.append(pygpc.MovingParticleFrictionForce())
    #     tests.append(pygpc.SurfaceCoverageSpecies())
    #
    #     for n_cpu in [4, 0]:
    #         if n_cpu != 0:
    #             print("Running testfunctions using multi-threading with {} cores...".format(n_cpu))
    #         else:
    #             print("Running testfunctions using inherited function parallelization...")
    #
    #         com = pygpc.Computation(n_cpu=n_cpu)
    #
    #         for t in tests:
    #             grid = pygpc.Random(parameters_random=t.problem.parameters_random,
    #                                 n_grid=10,
    #                                 options={"seed": 1})
    #
    #             res = com.run(model=t.problem.model,
    #                           problem=t.problem,
    #                           coords=grid.coords,
    #                           coords_norm=grid.coords_norm,
    #                           i_iter=None,
    #                           i_subiter=None,
    #                           fn_results=None,
    #                           print_func_time=False)
    #
    #         com.close()
    #
    #         print("done!\n")
    #
    # def test_009_RandomParameters(self):
    #     """
    #     Testing RandomParameters
    #     """
    #     global folder, plot
    #     test_name = 'pygpc_test_009_RandomParameters'
    #     print(test_name)
    #
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[5, 5], pdf_limits=[0, 1])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[0, 1])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[2, 5], pdf_limits=[0, 1])
    #     parameters["x5"] = pygpc.Beta(pdf_shape=[0.75, 0.75], pdf_limits=[0, 1])
    #     parameters["x6"] = pygpc.Norm(pdf_shape=[5, 1])
    #
    #     if plot:
    #         import matplotlib.pyplot as plt
    #         fig = plt.figure()
    #         ax = parameters["x1"].plot_pdf()
    #         ax = parameters["x2"].plot_pdf()
    #         ax = parameters["x3"].plot_pdf()
    #         ax = parameters["x4"].plot_pdf()
    #         ax = parameters["x5"].plot_pdf()
    #         ax = parameters["x6"].plot_pdf()
    #         ax.legend(["x1", "x2", "x3", "x4", "x5", "x6"])
    #         ax.savefig(os.path.join(folder, test_name) + ".png")
    #
    #         print("done!\n")
    #
    # def test_010_quadrature_grids(self):
    #     """
    #     Testing Grids [TensorGrid, SparseGrid]
    #     """
    #     global folder, plot
    #     test_name = 'pygpc_test_010_quadrature_grids'
    #     print(test_name)
    #
    #     # define testfunction
    #     test = pygpc.Peaks()
    #
    #     # TensorGrid
    #     grid_tensor_1 = pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
    #                                      options={"grid_type": ["hermite", "jacobi"], "n_dim": [5, 10]})
    #
    #     grid_tensor_2 = pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
    #                                      options={"grid_type": ["patterson", "fejer2"], "n_dim": [3, 10]})
    #
    #     # SparseGrid
    #     grid_sparse = pygpc.SparseGrid(parameters_random=test.problem.parameters_random,
    #                                    options={"grid_type": ["jacobi", "jacobi"],
    #                                             "level": [3, 3],
    #                                             "level_max": 3,
    #                                             "interaction_order": 2,
    #                                             "order_sequence_type": "exp"})
    #
    #     print("done!\n")
    #
    # def test_011_random_grid(self):
    #     """
    #     Testing Grids [Random]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_011_random_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     n_grid = 100
    #     n_grid_extend = 10
    #
    #     # generate grid w/o percentile constraint
    #     #########################################
    #     # initialize grid
    #     grid = pygpc.Random(parameters_random=problem_1.parameters_random,
    #                         n_grid=n_grid,
    #                         options={"seed": seed})
    #     self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #
    #     # extend grid
    #     for i in range(2):
    #         grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
    #         self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
    #                          f"Size of random grid does not fit after extending it {i+1}. time.")
    #         self.expect_true(pygpc.get_different_rows_from_matrices(
    #             grid.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                          f"Extended grid points are matching the initial grid after extending it {i+1}. time.")
    #
    #     # generate grid with percentile constraint
    #     ##########################################
    #     # initialize grid
    #     grid = pygpc.Random(parameters_random=problem_2.parameters_random,
    #                         n_grid=n_grid,
    #                         options={"seed": seed})
    #
    #     perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #     for i_p, p in enumerate(problem_2.parameters_random):
    #         perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                           (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #     self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #     self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #
    #     # extend grid
    #     for i in range(2):
    #         grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #         perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #         for i_p, p in enumerate(problem_2.parameters_random):
    #             perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                               (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #         self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #         self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                          "Size of random grid does not fit after extending it.")
    #         self.expect_true(pygpc.get_different_rows_from_matrices(
    #             grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                          f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #     # perform static gpc
    #     ###############################
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order"] = [9, 9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = {"seed": seed}
    #     options["matrix_ratio"] = None
    #     options["n_grid"] = 100
    #     options["order_start"] = 3
    #     options["order_end"] = 15
    #     options["eps"] = 0.001
    #     options["adaptive_sampling"] = False
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")
    #
    #     # perform adaptive gpc
    #     ##############################
    #     options["matrix_ratio"] = 2
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")
    #
    #     print("done!\n")
    #
    # def test_012_LHS_grid(self):
    #     """
    #     Testing Grids [LHS]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_012_LHS_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.7)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.7)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     n_grid_extend = 10
    #
    #     options_dict = {"ese": {"criterion": "ese", "seed": seed},
    #                     "maximin": {"criterion": "maximin", "seed": seed},
    #                     "corr": {"criterion": "corr", "seed": seed},
    #                     "standard": {"criterion": None, "seed": seed}}
    #
    #     for i_c, c in enumerate(options_dict):
    #         print(f"- criterion: {c} -")
    #
    #         # generate grid w/o percentile constraint
    #         #########################################
    #         print("- generate grid w/o percentile constraint -")
    #         n_grid = 20
    #
    #         # initialize grid
    #         print("  > init")
    #         grid = pygpc.LHS(parameters_random=problem_1.parameters_random,
    #                          n_grid=n_grid,
    #                          options=options_dict[c])
    #         self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              f"Size of random grid does not fit after extending it {i + 1}. time.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # generate grid with percentile constraint
    #         ##########################################
    #         print("- generate grid with percentile constraint -")
    #         print("  > init")
    #         n_grid = 100
    #
    #         # initialize grid
    #         grid = pygpc.LHS(parameters_random=problem_2.parameters_random,
    #                          n_grid=n_grid,
    #                          options=options_dict[c])
    #
    #         perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #         for i_p, p in enumerate(problem_2.parameters_random):
    #             perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                               (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #         self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #         self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #             perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #             for i_p, p in enumerate(problem_2.parameters_random):
    #                 perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                                   (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #             self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              "Size of random grid does not fit after extending it.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # perform static gpc
    #         ###############################
    #         print("- Static gpc -")
    #         # gPC options
    #         options = dict()
    #         options["method"] = "reg"
    #         options["solver"] = "LarsLasso"
    #         options["settings"] = None
    #         options["order"] = [7, 7, 7]
    #         options["order_max"] = 7
    #         options["interaction_order"] = 2
    #         options["error_type"] = "nrmsd"
    #         options["n_samples_validation"] = 1e3
    #         options["n_cpu"] = 0
    #         options["fn_results"] = None
    #         options["save_session_format"] = save_session_format
    #         options["gradient_enhanced"] = False
    #         options["gradient_calculation"] = "FD_1st2nd"
    #         options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #         options["backend"] = "omp"
    #         options["grid"] = pygpc.LHS
    #         options["grid_options"] = options_dict[c]
    #         options["matrix_ratio"] = None
    #         options["n_grid"] = 100
    #         options["order_start"] = 3
    #         options["order_end"] = 15
    #         options["eps"] = 0.001
    #         options["adaptive_sampling"] = False
    #
    #         # define algorithm
    #         algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")
    #
    #         # perform adaptive gpc
    #         ##############################
    #         print("- Adaptive gpc -")
    #         options["matrix_ratio"] = 2
    #         options["n_grid"] = None
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")
    #
    #     print("done!\n")
    #
    # def test_013_L1_grid(self):
    #     """
    #     Testing Grids [L1]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_013_L1_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 3
    #     options["order_end"] = 15
    #     options["order"] = [7, 7, 7]
    #     options["order_max"] = 7
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["eps"] = 0.001
    #     options["adaptive_sampling"] = False
    #     options["grid"] = pygpc.L1
    #
    #     n_grid_extend = 10
    #     method_list = ["greedy", "iteration"]
    #     # method_list = ["iteration"]
    #     criterion_list = [["mc"], ["tmc", "cc"], ["D"], ["D-coh"]]
    #     # criterion_list = [["D"]]
    #
    #     grid_options = {"method": None,
    #                     "criterion": None,
    #                     "n_pool": 1000,
    #                     "n_iter": 1000,
    #                     "seed": seed}
    #
    #     for i_m, method in enumerate(method_list):
    #         print(f"- method: {method} -")
    #
    #         for i_c, criterion in enumerate(criterion_list):
    #             print(f"- criterion: {criterion} -")
    #             grid_options["criterion"] = criterion
    #             grid_options["method"] = method
    #             grid_options["n_iter"] = 100
    #             options["grid_options"] = grid_options
    #
    #             # generate grid w/o percentile constraint
    #             #########################################
    #             print("- generate grid w/o percentile constraint -")
    #             print("  > init")
    #             n_grid = 20
    #             grid_options["n_pool"] = 5 * n_grid
    #
    #             # create gpc object of some order for problem_1
    #             gpc = pygpc.Reg(problem=problem_1,
    #                             order=options["order"],
    #                             order_max=options["order_max"],
    #                             order_max_norm=options["order_max_norm"],
    #                             interaction_order=options["interaction_order"],
    #                             interaction_order_current=options["interaction_order"],
    #                             options=options,
    #                             validation=None)
    #
    #             # initialize grid
    #             grid = pygpc.L1(parameters_random=problem_1.parameters_random,
    #                             n_grid=n_grid,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #             self.expect_true(grid.n_grid == n_grid, f"Size of random grid does not fit after initialization. "
    #                                                     f"({criterion, method})")
    #
    #             # extend grid
    #             print("  > extend")
    #             for i in range(2):
    #                 grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #                 self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                                  f"Size of random grid does not fit after extending it {i + 1}. time. "
    #                                  f"({criterion, method})")
    #                 self.expect_true(pygpc.get_different_rows_from_matrices(
    #                     grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                                  f"Extended grid points are matching the initial grid after extending it {i + 1}. time. "
    #                                  f"({criterion, method})")
    #
    #             # generate grid with percentile constraint
    #             ##########################################
    #             print("- generate grid with percentile constraint -")
    #             print("  > init")
    #             n_grid = 100
    #             grid_options["n_pool"] = 2 * n_grid
    #
    #             # create gpc object of some order for problem_2
    #             gpc = pygpc.Reg(problem=problem_2,
    #                             order=options["order"],
    #                             order_max=options["order_max"],
    #                             order_max_norm=options["order_max_norm"],
    #                             interaction_order=options["interaction_order"],
    #                             interaction_order_current=options["interaction_order"],
    #                             options=options,
    #                             validation=None)
    #
    #             # initialize grid
    #             grid = pygpc.L1(parameters_random=problem_2.parameters_random,
    #                             n_grid=n_grid,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #             perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #             for i_p, p in enumerate(problem_2.parameters_random):
    #                 perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                                   (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #             self.expect_true(grid.n_grid == n_grid,
    #                              f"Size of random grid does not fit after initialization. "
    #                              f"({criterion, method})")
    #             self.expect_true(perc_check.all(),
    #                              f"Grid points do not fulfill percentile constraint. "
    #                              f"({criterion, method})")
    #
    #             # extend grid
    #             print("  > extend")
    #             for i in range(2):
    #                 grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #                 perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #                 for i_p, p in enumerate(problem_2.parameters_random):
    #                     perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[
    #                         0]).all() and \
    #                                       (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #                 self.expect_true(perc_check.all(), f"Grid points do not fulfill percentile constraint. "
    #                                                    f"({criterion, method})")
    #                 self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                                  f"Size of random grid does not fit after extending it.  "
    #                                  f"({criterion, method})")
    #                 self.expect_true(pygpc.get_different_rows_from_matrices(
    #                     grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                                  f"Extended grid points are matching the initial grid after extending it {i + 1}. time. "
    #                                  f"({criterion, method})")
    #
    #             # perform static gpc
    #             ###############################
    #             print("  > Perform Static gpc")
    #             options["n_grid"] = None
    #             options["matrix_ratio"] = 1.5
    #             grid_options["n_pool"] = 500
    #             grid_options["n_iter"] = 500
    #             options["grid_options"] = grid_options
    #
    #             # define algorithm
    #             algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #             # Initialize gPC Session
    #             session = pygpc.Session(algorithm=algorithm)
    #
    #             # run gPC algorithm
    #             session, coeffs, results = session.run()
    #
    #             self.expect_true(session.gpc[0].error[0] <= 0.001, f"Error of static gpc too high. "
    #                                                                f"({criterion, method})")
    #
    #             # perform adaptive gpc
    #             ##############################
    #             print("  > Perform Adaptive gpc")
    #             # define algorithm
    #             algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #             # Initialize gPC Session
    #             session = pygpc.Session(algorithm=algorithm)
    #
    #             # run gPC algorithm
    #             session, coeffs, results = session.run()
    #
    #             self.expect_true(session.gpc[0].error[-1] <= 0.001, f"Error of adaptive gpc too high. "
    #                                                                 f"({criterion, method})")
    #
    #     print("done!\n")
    #
    # def test_014_L1_LHS_grid(self):
    #     """
    #     Testing Grids [L1_LHS]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_014_L1_LHS_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.8)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.8)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 3
    #     options["order_end"] = 15
    #     options["order"] = [7, 7, 7]
    #     options["order_max"] = 7
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["eps"] = 0.001
    #     options["adaptive_sampling"] = False
    #     options["grid"] = pygpc.L1_LHS
    #
    #     n_grid_extend = 10
    #     weights_list = [[0, 1], [0.5, 0.5], [1, 0]]
    #
    #     grid_options = {"weights": None,
    #                     "method": "iteration",
    #                     "criterion": ["mc"],
    #                     "n_iter": 100,
    #                     "seed": seed}
    #
    #     for i_w, weights in enumerate(weights_list):
    #         grid_options["weights"] = weights
    #         options["grid_options"] = grid_options
    #         options["matrix_ratio"] = None
    #
    #         # generate grid w/o percentile constraint
    #         #########################################
    #         print("- generate grid w/o percentile constraint -")
    #         n_grid = 20
    #         # create gpc object of some order for problem_1
    #         gpc = pygpc.Reg(problem=problem_1,
    #                         order=options["order"],
    #                         order_max=options["order_max"],
    #                         order_max_norm=options["order_max_norm"],
    #                         interaction_order=options["interaction_order"],
    #                         interaction_order_current=options["interaction_order"],
    #                         options=options,
    #                         validation=None)
    #
    #         # initialize grid
    #         print("  > init")
    #         grid = pygpc.L1_LHS(parameters_random=problem_1.parameters_random,
    #                             n_grid=n_grid,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #         self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              f"Size of random grid does not fit after extending it {i + 1}. time.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # generate grid with percentile constraint
    #         ##########################################
    #         print("- generate grid with percentile constraint -")
    #         n_grid = 100
    #         # create gpc object of some order for problem_2
    #         gpc = pygpc.Reg(problem=problem_2,
    #                         order=options["order"],
    #                         order_max=options["order_max"],
    #                         order_max_norm=options["order_max_norm"],
    #                         interaction_order=options["interaction_order"],
    #                         interaction_order_current=options["interaction_order"],
    #                         options=options,
    #                         validation=None)
    #
    #         # initialize grid
    #         print("  > init")
    #         grid = pygpc.L1_LHS(parameters_random=problem_2.parameters_random,
    #                             n_grid=n_grid,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #         perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #         for i_p, p in enumerate(problem_2.parameters_random):
    #             perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                               (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #         self.expect_true(grid.n_grid == n_grid,
    #                          "Size of random grid does not fit after initialization.")
    #         self.expect_true(perc_check.all(),
    #                          "Grid points do not fulfill percentile constraint.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #             perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #             for i_p, p in enumerate(problem_2.parameters_random):
    #                 perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                                   (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #             self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              "Size of random grid does not fit after extending it.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # perform static gpc
    #         ###############################
    #         print("- Perform Static gpc -")
    #         options["n_grid"] = None
    #         options["matrix_ratio"] = 2
    #         options["grid_options"] = grid_options
    #
    #         # define algorithm
    #         algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")
    #
    #         # perform adaptive gpc
    #         ##############################
    #         print("- Perform Adaptive gpc -")
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")
    #
    #     print("done!\n")
    #
    # def test_015_LHS_L1_grid(self):
    #     """
    #     Testing Grids [LHS_L1]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_015_LHS_L1_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.8)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.8)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 3
    #     options["order_end"] = 15
    #     options["order"] = [7, 7, 7]
    #     options["order_max"] = 7
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["eps"] = 0.001
    #     options["adaptive_sampling"] = False
    #     options["grid"] = pygpc.LHS_L1
    #
    #     n_grid_extend = 10
    #     weights_list = [[0, 1], [0.5, 0.5], [1, 0]]
    #
    #     grid_options = {"weights": None,
    #                     "method": "iteration",
    #                     "criterion": ["mc"],
    #                     "n_iter": 100,
    #                     "seed": seed}
    #
    #     for i_w, weights in enumerate(weights_list):
    #         grid_options["weights"] = weights
    #         options["grid_options"] = grid_options
    #
    #         # generate grid w/o percentile constraint
    #         #########################################
    #         print(" - generate grid w/o percentile constraint -")
    #         n_grid = 20
    #
    #         # create gpc object of some order for problem_1
    #         gpc = pygpc.Reg(problem=problem_1,
    #                         order=options["order"],
    #                         order_max=options["order_max"],
    #                         order_max_norm=options["order_max_norm"],
    #                         interaction_order=options["interaction_order"],
    #                         interaction_order_current=options["interaction_order"],
    #                         options=options,
    #                         validation=None)
    #
    #         # initialize grid
    #         print("  > init")
    #         grid = pygpc.LHS_L1(parameters_random=problem_1.parameters_random,
    #                             n_grid=20,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #         self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              f"Size of random grid does not fit after extending it {i + 1}. time.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # generate grid with percentile constraint
    #         ##########################################
    #         print(" - generate grid with percentile constraint -")
    #         # create gpc object of some order for problem_2
    #         n_grid = 100
    #         gpc = pygpc.Reg(problem=problem_2,
    #                         order=options["order"],
    #                         order_max=options["order_max"],
    #                         order_max_norm=options["order_max_norm"],
    #                         interaction_order=options["interaction_order"],
    #                         interaction_order_current=options["interaction_order"],
    #                         options=options,
    #                         validation=None)
    #
    #         # initialize grid
    #         print("  > init")
    #         grid = pygpc.LHS_L1(parameters_random=problem_2.parameters_random,
    #                             n_grid=n_grid,
    #                             options=grid_options,
    #                             gpc=gpc)
    #
    #         perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #         for i_p, p in enumerate(problem_2.parameters_random):
    #             perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                               (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #         self.expect_true(grid.n_grid == n_grid,
    #                          "Size of random grid does not fit after initialization.")
    #         self.expect_true(perc_check.all(),
    #                          "Grid points do not fulfill percentile constraint.")
    #
    #         # extend grid
    #         print("  > extend")
    #         for i in range(2):
    #             grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #             perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #             for i_p, p in enumerate(problem_2.parameters_random):
    #                 perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                                   (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #             self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #             self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                              "Size of random grid does not fit after extending it.")
    #             self.expect_true(pygpc.get_different_rows_from_matrices(
    #                 grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                              f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #         # perform static gpc
    #         ###############################
    #         print("- Perform Static gpc -")
    #         options["n_grid"] = None
    #         options["matrix_ratio"] = 2
    #         options["grid_options"] = grid_options
    #
    #         # define algorithm
    #         algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[0] <= 0.001, "Error of static gpc too high.")
    #
    #         # perform adaptive gpc
    #         ##############################
    #         print("- Perform Adaptive gpc -")
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC algorithm
    #         session, coeffs, results = session.run()
    #
    #         self.expect_true(session.gpc[0].error[-1] <= 0.001, "Error of adaptive gpc too high.")
    #
    #     print("done!\n")
    #
    # def test_016_FIM_grid(self):
    #     """
    #     Testing Grids [FIM]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_016_FIM_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     parameters_2 = OrderedDict()
    #     parameters_2["x1"] = pygpc.Norm(pdf_shape=[0, 1], p_perc=0.5)
    #     parameters_2["x2"] = pygpc.Norm(pdf_shape=[1, 2], p_perc=0.5)
    #     parameters_2["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_2 = pygpc.Problem(model, parameters_2)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 2
    #     options["order_end"] = 5
    #     options["order"] = [3, 3, 3]
    #     options["order_max"] = 3
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["eps"] = 0.05
    #     options["adaptive_sampling"] = False
    #     options["grid"] = pygpc.FIM
    #     options["grid_options"] = {"seed": seed, "n_pool": 10}
    #
    #     n_grid_extend = 10
    #
    #     # generate grid w/o percentile constraint
    #     #########################################
    #     print("- generate grid w/o percentile constraint -")
    #     print("  > init")
    #     n_grid = 20
    #
    #     # create gpc object of some order for problem_1
    #     gpc = pygpc.Reg(problem=problem_1,
    #                     order=[2, 2, 2],
    #                     order_max=2,
    #                     order_max_norm=1,
    #                     interaction_order=2,
    #                     interaction_order_current=2,
    #                     options=options,
    #                     validation=None)
    #
    #     # initialize grid
    #     grid = pygpc.FIM(parameters_random=problem_1.parameters_random,
    #                      n_grid=n_grid,
    #                      options=options["grid_options"],
    #                      gpc=gpc)
    #
    #     self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
    #
    #     # extend grid
    #     print("  > extend")
    #     for i in range(2):
    #         grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
    #         self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
    #                          f"Size of random grid does not fit after extending it {i+1}. time.")
    #         self.expect_true(pygpc.get_different_rows_from_matrices(
    #             grid.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                          f"Extended grid points are matching the initial grid after extending it {i+1}. time.")
    #
    #     # generate grid with percentile constraint
    #     ##########################################
    #     print("- generate grid with percentile constraint -")
    #     print("  > init")
    #     n_grid = 50
    #
    #     # create gpc object of some order for problem_2
    #     gpc = pygpc.Reg(problem=problem_2,
    #                     order=[2, 2, 2],
    #                     order_max=2,
    #                     order_max_norm=1,
    #                     interaction_order=2,
    #                     interaction_order_current=2,
    #                     options=options,
    #                     validation=None)
    #
    #     # initialize grid
    #     grid = pygpc.FIM(parameters_random=problem_2.parameters_random,
    #                      n_grid=n_grid,
    #                      options=options["grid_options"],
    #                      gpc=gpc)
    #
    #     perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #     for i_p, p in enumerate(problem_2.parameters_random):
    #         perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                           (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #     self.expect_true(grid.n_grid == n_grid,
    #                      "Size of random grid does not fit after initialization.")
    #     self.expect_true(perc_check.all(),
    #                      "Grid points do not fulfill percentile constraint.")
    #
    #     # extend grid
    #     print("  > extend")
    #     for i in range(2):
    #         grid.extend_random_grid(n_grid_new=n_grid + (i + 1) * n_grid_extend)
    #
    #         perc_check = np.zeros(len(problem_2.parameters_random)).astype(bool)
    #
    #         for i_p, p in enumerate(problem_2.parameters_random):
    #             perc_check[i_p] = (grid.coords[:, i_p] >= problem_2.parameters_random[p].pdf_limits[0]).all() and \
    #                               (grid.coords[:, i_p] <= problem_2.parameters_random[p].pdf_limits[1]).all()
    #
    #         self.expect_true(perc_check.all(), "Grid points do not fulfill percentile constraint.")
    #         self.expect_true(grid.n_grid == n_grid + (i + 1) * n_grid_extend,
    #                          "Size of random grid does not fit after extending it.")
    #         self.expect_true(pygpc.get_different_rows_from_matrices(
    #             grid.coords_norm[0:n_grid + i * n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
    #                          f"Extended grid points are matching the initial grid after extending it {i + 1}. time.")
    #
    #     # perform static gpc
    #     ###############################
    #     print("- Perform Static gpc -")
    #     options["n_grid"] = None
    #     options["matrix_ratio"] = 1.5
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem_1, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     self.expect_true(session.gpc[0].error[0] <= 0.05, "Error of static gpc too high.")
    #
    #     # perform adaptive gpc
    #     ###############################
    #     print("- Perform Adaptive gpc -")
    #     options["grid_options"]["n_pool"] = 10
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     self.expect_true(session.gpc[0].error[-1] <= 0.05, "Error of adaptive gpc too high.")
    #
    #     print("done!\n")

    # def test_018_CO_grid(self):
    #     """
    #     Testing Grids [CO]
    #     """
    #     global folder, plot, seed
    #     test_name = 'pygpc_test_018_CO_grid'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters_1 = OrderedDict()
    #     parameters_1["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters_1["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem_1 = pygpc.Problem(model, parameters_1)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = None
    #     options["order_start"] = 2
    #     options["order_end"] = 5
    #     options["order"] = [3, 3, 3]
    #     options["order_max"] = 3
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["eps"] = 0.05
    #     options["adaptive_sampling"] = False
    #     options["grid"] = pygpc.CO
    #     options["grid_options"] = {"seed": seed, "n_warmup": 10}

        # n_grid = 100
        # n_grid_extend = 10

        # # generate grid w/o percentile constraint
        # #########################################
        # print("- generate grid w/o percentile constraint -")
        # print("  > init")
        #
        # # create gpc object of some order for problem_1
        # gpc = pygpc.Reg(problem=problem_1,
        #                 order=[2, 2, 2],
        #                 order_max=2,
        #                 order_max_norm=1,
        #                 interaction_order=2,
        #                 interaction_order_current=2,
        #                 options=options,
        #                 validation=None)
        #
        # # initialize grid
        # grid = pygpc.CO(parameters_random=problem_1.parameters_random,
        #                 n_grid=n_grid,
        #                 gpc=gpc,
        #                 options={"seed": seed, "n_warmup": 10})
        # self.expect_true(grid.n_grid == n_grid, "Size of random grid does not fit after initialization.")
        #
        # # extend grid
        # print("  > extend")
        # for i in range(2):
        #     grid.extend_random_grid(n_grid_new=n_grid + (i+1)*n_grid_extend)
        #     self.expect_true(grid.n_grid == n_grid + (i+1)*n_grid_extend,
        #                      f"Size of random grid does not fit after extending it {i+1}. time.")
        #     self.expect_true(pygpc.get_different_rows_from_matrices(
        #         grid.coords_norm[0:n_grid + i*n_grid_extend, :], grid.coords_norm).shape[0] == n_grid_extend,
        #                      f"Extended grid points are matching the initial grid after extending it {i+1}. time.")

        # # perform static gpc
        # ###############################
        # print("- Perform Static gpc -")
        #
        # # gPC options
        # options = dict()
        # options["method"] = "reg"
        # options["solver"] = "LarsLasso"
        # options["settings"] = None
        # options["order"] = [9, 9, 9]
        # options["order_max"] = 9
        # options["interaction_order"] = 2
        # options["error_type"] = "nrmsd"
        # options["n_samples_validation"] = 1e3
        # options["n_cpu"] = 0
        # options["fn_results"] = None
        # options["save_session_format"] = save_session_format
        # options["gradient_enhanced"] = False
        # options["gradient_calculation"] = "FD_1st2nd"
        # options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        # options["backend"] = "omp"
        # options["grid"] = pygpc.CO
        # options["grid_options"] = {"seed": seed, "n_warmup": 10}
        # options["matrix_ratio"] = None
        # options["n_grid"] = 100
        # options["order_start"] = 3
        # options["order_end"] = 11
        # options["eps"] = 0.001
        # options["adaptive_sampling"] = False
        #
        # # define algorithm
        # algorithm = pygpc.Static(problem=problem_1, options=options)
        #
        # # Initialize gPC Session
        # session = pygpc.Session(algorithm=algorithm)
        #
        # # run gPC algorithm
        # session, coeffs, results = session.run()
        #
        # self.expect_true(session.gpc[0].error[0] <= 0.01, "Error of static gpc too high.")
        #
        # # perform adaptive gpc
        # ##############################
        # print("- Perform Adaptive gpc -")
        #
        # options["matrix_ratio"] = 2
        #
        # # define algorithm
        # algorithm = pygpc.RegAdaptive(problem=problem_1, options=options)
        #
        # # Initialize gPC Session
        # session = pygpc.Session(algorithm=algorithm)
        #
        # # run gPC algorithm
        # session, coeffs, results = session.run()
        #
        # self.expect_true(session.gpc[0].error[-1] <= 0.01, "Error of adaptive gpc too high.")

        # print("done!\n")

    # def test_019_seed_grids_reproducibility(self):
    #     """
    #     Test reproducibility of grids when seeding
    #     """
    #     global folder, plot, matlab, save_session_format
    #     test_name = 'pygpc_test_019_seed_grids_reproducibility'
    #     print(test_name)
    #
    #     # define testfunction
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problems
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     gpc = pygpc.Reg(problem=problem, order_max=2)
    #
    #     # Random
    #     print("Testing reproducibility of Random grid ...")
    #     grid = [0 for _ in range(2)]
    #     for i in range(2):
    #         # initialize grid
    #         grid[i] = pygpc.Random(parameters_random=problem.parameters_random,
    #                                n_grid=10,
    #                                options={"seed": 1})
    #
    #         # extend grid
    #         grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #     # compare
    #     self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                      "Random grid is not reproducible when seeding")
    #
    #     # LHS
    #     print("Testing reproducibility of LHS grids ...")
    #     criterion_list = [None, "maximin", "ese"]
    #
    #     for criterion in criterion_list:
    #         print(f"\t > criterion: {criterion}")
    #         grid = [0 for _ in range(2)]
    #
    #         for i in range(2):
    #             # initialize grid
    #             grid[i] = pygpc.LHS(parameters_random=problem.parameters_random,
    #                                 n_grid=10,
    #                                 options={"criterion": criterion,
    #                                          "seed": 1})
    #
    #             # extend grid
    #             grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #         # compare
    #         self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                          f"LHS ({criterion}) grid is not reproducible when seeding")
    #
    #     # L1
    #     print("Testing reproducibility of L1 grids ...")
    #     criterion_list = [["mc"], ["tmc", "cc"], ["D"], ["D-coh"]]
    #     method_list = ["greedy", "iter"]
    #
    #     for criterion in criterion_list:
    #         print(f"\t > criterion: {criterion}")
    #         for method in method_list:
    #             print(f"\t\t > method: {method}")
    #             grid = [0 for _ in range(2)]
    #
    #             for i in range(2):
    #                 # initialize grid
    #                 grid[i] = pygpc.L1(parameters_random=problem.parameters_random,
    #                                    n_grid=10,
    #                                    options={"criterion": criterion,
    #                                             "method": method,
    #                                             "seed": 1,
    #                                             "n_pool": 100,
    #                                             "n_iter": 100},
    #                                    gpc=gpc)
    #
    #                 # extend grid
    #                 grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #             # compare
    #             self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                              f"L1 ({criterion}, {method}) grid is not reproducible when seeding")
    #
    #     # L1_LHS
    #     print("Testing reproducibility of L1-LHS grids ...")
    #     criterion_list = [["mc"], ["tmc", "cc"]]
    #     method_list = ["greedy", "iter"]
    #
    #     for criterion in criterion_list:
    #         print(f"\t > criterion: {criterion}")
    #         for method in method_list:
    #             print(f"\t\t > method: {method}")
    #             grid = [0 for _ in range(2)]
    #
    #             for i in range(2):
    #                 # initialize grid
    #                 grid[i] = pygpc.L1_LHS(parameters_random=problem.parameters_random,
    #                                        n_grid=10,
    #                                        options={"weights": [0.5, 0.5],
    #                                                 "criterion": criterion,
    #                                                 "method": method,
    #                                                 "seed": 1,
    #                                                 "n_pool": 100,
    #                                                 "n_iter": 100},
    #                                        gpc=gpc)
    #
    #                 # extend grid
    #                 grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #             # compare
    #             self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                              f"L1_LHS ({criterion}, {method}) grid is not reproducible when seeding")
    #
    #     # LHS_L1
    #     print("Testing reproducibility of LHS-L1 grids ...")
    #     criterion_list = [["mc"], ["tmc", "cc"]]
    #     method_list = ["greedy", "iter"]
    #
    #     for criterion in criterion_list:
    #         print(f"\t > criterion: {criterion}")
    #         for method in method_list:
    #             print(f"\t\t > method: {method}")
    #             grid = [0 for _ in range(2)]
    #
    #             for i in range(2):
    #                 # initialize grid
    #                 grid[i] = pygpc.LHS_L1(parameters_random=problem.parameters_random,
    #                                        n_grid=10,
    #                                        options={"weights": [0.5, 0.5],
    #                                                 "criterion": criterion,
    #                                                 "method": method,
    #                                                 "seed": 1,
    #                                                 "n_pool": 100,
    #                                                 "n_iter": 100},
    #                                        gpc=gpc)
    #
    #                 # extend grid
    #                 grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #             # compare
    #             self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                              f"LHS_L1 ({criterion}, {method}) grid is not reproducible when seeding")
    #
    #     # FIM
    #     print("Testing reproducibility of FIM grid ...")
    #     grid = [0 for _ in range(2)]
    #     for i in range(2):
    #         # initialize grid
    #         grid[i] = pygpc.FIM(parameters_random=problem.parameters_random,
    #                             n_grid=10,
    #                             options={"seed": 1},
    #                             gpc=gpc)
    #
    #         # extend grid
    #         grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #     # compare
    #     self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                      "FIM grid is not reproducible when seeding")
    #
    #     # CO
    #     print("Testing reproducibility of CO grid ...")
    #     grid = [0 for _ in range(2)]
    #     for i in range(2):
    #         # initialize grid
    #         grid[i] = pygpc.CO(parameters_random=problem.parameters_random,
    #                            n_grid=10,
    #                            options={"seed": 1, "n_warmup": 10},
    #                            gpc=gpc)
    #
    #         # extend grid
    #         grid[i].extend_random_grid(n_grid_new=grid[i].n_grid + 5)
    #
    #     # compare
    #     self.expect_true((grid[0].coords_norm == grid[1].coords_norm).all(),
    #                      "CO grid is not reproducible when seeding")

    # def test_020_Matlab_gpc(self):
    #     """
    #     Algorithm: RegAdaptive
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, matlab, save_session_format
    #     test_name = 'pygpc_test_020_Matlab_gpc'
    #     print(test_name)
    #
    #     if matlab:
    #         import matlab.engine
    #         from templates.MyModel_matlab import  MyModel_matlab
    #         # define model
    #         model = MyModel_matlab(fun_path=os.path.join(pygpc.__path__[0], "testfunctions"))
    #
    #         # define problem (the parameter names have to be the same as in the model)
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    #         parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    #         parameters["x3"] = 0.
    #         parameters["a"] = 7.
    #         parameters["b"] = 0.1
    #
    #         problem = pygpc.Problem(model, parameters)
    #
    #         # gPC options
    #         options = dict()
    #         options["order_start"] = 5
    #         options["order_end"] = 20
    #         options["solver"] = "LarsLasso"
    #         options["interaction_order"] = 2
    #         options["order_max_norm"] = 0.7
    #         options["n_cpu"] = 0
    #         options["adaptive_sampling"] = True
    #         options["gradient_enhanced"] = True
    #         options["gradient_calculation"] = "FD_fwd"
    #         options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["save_session_format"] = save_session_format
    #         options["eps"] = 0.0075
    #         options["matlab_model"] = True
    #         options["grid"] = pygpc.Random
    #         options["grid_options"] = None
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #         # Initialize gPC Session
    #         session = pygpc.Session(algorithm=algorithm)
    #
    #         # run gPC session
    #         session, coeffs, results = session.run()
    #
    #         if plot:
    #             # Validate gPC vs original model function (2D-surface)
    #             pygpc.validate_gpc_plot(session=session,
    #                                     coeffs=coeffs,
    #                                     random_vars=list(problem.parameters_random.keys()),
    #                                     n_grid=[51, 51],
    #                                     output_idx=0,
    #                                     fn_out=options["fn_results"] + "_val",
    #                                     n_cpu=options["n_cpu"])
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True,
    #                                      algorithm="sampling",
    #                                      n_samples=1e3)
    #
    #         # Validate gPC vs original model function (Monte Carlo)
    #         nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       n_cpu=options["n_cpu"],
    #                                       smooth_pdf=True,
    #                                       fn_out=options["fn_results"] + "_pdf",
    #                                       plot=plot)
    #
    #         print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #         # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #         print("> Checking file consistency...")
    #         files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #         self.expect_true(files_consistent, error_msg)
    #
    #         print("done!\n")
    #
    #     else:
    #         print("Skipping Matlab test...")
    #
    # def test_021_random_vars_postprocessing(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_021_random_vars_postprocessing_sobol'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #
    #     parameters = OrderedDict()
    #     # parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[1.25, 1.72])
    #     parameters["x1"] = pygpc.Gamma(pdf_shape=[3., 10., 1.25], p_perc=0.98)
    #     parameters["x2"] = pygpc.Norm(pdf_shape=[1, 1], p_perc=0.98)
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0.6, 1.4])
    #     # parameters["x3"] = pygpc.Norm(pdf_shape=[1., 0.25], p_perc=0.95)
    #     # parameters["x2"] = 1.
    #
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [4, 4, 4]
    #     options["order_max"] = 4
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["error_type"] = "loocv"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["gradient_enhanced"] = True
    #     options["backend"] = "omp"
    #     options["grid_options"] = {"seed": seed}
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.Random(parameters_random=problem.parameters_random,
    #                         n_grid=options["matrix_ratio"] * n_coeffs,
    #                         options=options["grid_options"])
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # Determine Sobol indices using standard approach (gPC coefficients)
    #     sobol_standard, sobol_idx_standard, sobol_idx_bool_standard = session.gpc[0].get_sobol_indices(coeffs=coeffs,
    #                                                                                                    algorithm="standard")
    #
    #     sobol_sampling, sobol_idx_sampling, sobol_idx_bool_sampling = session.gpc[0].get_sobol_indices(coeffs=coeffs,
    #                                                                                                    algorithm="sampling",
    #                                                                                                    n_samples=3e4)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=options["fn_results"],
    #                                   folder="validate_gpc_mc",
    #                                   plot=plot,
    #                                   n_cpu=session.n_cpu)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     for i in range(sobol_standard.shape[0]):
    #         self.expect_true(np.max(np.abs(sobol_standard[i, :]-sobol_sampling[i, :])) < 0.1,
    #                          msg="Sobol index: {}".format(str(sobol_idx_sampling[3])))
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=["x2", "x3"],
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=options["fn_results"],
    #                                 folder="validate_gpc_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     print("done!\n")
    #
    # def test_022_clustering_3_domains(self):
    #     """
    #     Algorithm: MERegAdaptiveprojection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: Random
    #     """
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_022_clustering_3_domains'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Cluster3Simple()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "LarsLasso" #"Moore-Penrose"
    #     options["settings"] = None
    #     options["order_start"] = 1
    #     options["order_end"] = 15
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 1
    #     options["projection"] = False
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = False
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["error_type"] = "loocv"
    #     options["error_norm"] = "absolute" # "relative"
    #     options["qoi"] = 0 # "all"
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 3,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["n_samples_discontinuity"] = 50
    #     options["adaptive_sampling"] = False
    #     options["eps"] = 0.01
    #     options["n_grid_init"] = 500
    #     options["backend"] = "omp"
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = save_session_format
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     # define algorithm
    #     algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC session
    #     session, coeffs, results = session.run()
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=[0],
    #                                 fn_out=options["fn_results"],
    #                                 folder="validate_gpc_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   smooth_pdf=True,
    #                                   fn_out=options["fn_results"],
    #                                   folder="validate_gpc_mc",
    #                                   plot=plot)
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling",
    #                                  n_samples=1e4)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_023_backends(self):
    #     """
    #     Test the different backends ["python", "cpu", "omp", "cuda"]
    #     """
    #
    #     global folder, gpu
    #     test_name = 'pygpc_test_023_backends'
    #     print(test_name)
    #
    #     backends = ["python", "cpu", "omp", "cuda"]
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # define test grid
    #     grid = pygpc.Random(parameters_random=problem.parameters_random,
    #                         n_grid=100,
    #                         options={"seed": 1})
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = None
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_fwd"
    #     options["gradient_calculation_options"] = {"dx": 0.5, "distance_weight": -2}
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     gpc_matrix = dict()
    #     gpc_matrix_gradient = dict()
    #     pce_matrix = dict()
    #
    #     print("Constructing gPC matrices with different backends...")
    #     for b in backends:
    #         try:
    #             options["backend"] = b
    #
    #             # setup gPC
    #             gpc = pygpc.Reg(problem=problem,
    #                             order=[8, 8],
    #                             order_max=8,
    #                             order_max_norm=0.8,
    #                             interaction_order=2,
    #                             interaction_order_current=2,
    #                             options=options,
    #                             validation=None)
    #
    #             gpc.grid = grid
    #
    #             # init gPC matrices
    #             start = time.time()
    #             gpc.init_gpc_matrix(gradient_idx=np.arange(grid.coords.shape[0]))
    #             stop = time.time()
    #
    #             print(b, "Time create_gpc_matrix: ", stop-start)
    #
    #             # perform polynomial chaos expansion
    #             coeffs = np.ones([len(gpc.basis.b), 2])
    #             start = time.time()
    #             pce = gpc.get_approximation(coeffs, gpc.grid.coords_norm)
    #             stop = time.time()
    #
    #             print(b, "Time get_approximation: ", stop-start)
    #
    #             gpc_matrix[b] = gpc.gpc_matrix
    #             gpc_matrix_gradient[b] = gpc.gpc_matrix_gradient
    #             pce_matrix[b] = pce
    #
    #         except NotImplementedError:
    #             backends.remove(b)
    #             warnings.warn("Skipping {} (not installed)...".format(b))
    #
    #     for b_ref in backends:
    #         for b_compare in backends:
    #             if b_compare != b_ref:
    #                 self.expect_isclose(gpc_matrix[b_ref], gpc_matrix[b_compare], atol=1e-6,
    #                                     msg="gpc matrices between "+b_ref+" and "+b_compare+" are not equal")
    #
    #                 self.expect_isclose(gpc_matrix_gradient[b_ref], gpc_matrix_gradient[b_compare], atol=1e-6,
    #                                     msg="gpc matrices between "+b_ref+" and "+b_compare+" are not equal")
    #
    #                 self.expect_isclose(pce_matrix[b_ref], pce_matrix[b_compare], atol=1e-6,
    #                                     msg="pce matrices between "+b_ref+" and "+b_compare+" are not equal")
    #
    #     print("done!\n")
    #
    # def test_024_save_and_load_session(self):
    #     """
    #     Save and load a gPC Session
    #     """
    #
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_024_save_and_load_session'
    #     print(test_name)
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 20
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["save_session_format"] = ".hdf5"
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "FD_1st2nd"
    #     options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    #     options["backend"] = "omp"
    #     options["grid"] = pygpc.Random
    #     options["grid_options"] = None
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options)
    #
    #     # Initialize gPC Session
    #     session = pygpc.Session(algorithm=algorithm)
    #
    #     # run gPC algorithm
    #     session, coeffs, results = session.run()
    #
    #     # read session
    #     session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="standard",
    #                                  n_samples=1e3)
    #
    #     if plot:
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(session=session,
    #                                 coeffs=coeffs,
    #                                 random_vars=list(problem.parameters_random.keys()),
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=options["fn_results"],
    #                                 folder="gpc_vs_original_plot",
    #                                 n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(session=session,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=session.n_cpu,
    #                                   fn_out=options["fn_results"],
    #                                   folder="gpc_vs_original_mc",
    #                                   plot=plot)
    #
    #     print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
    #     # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
    #
    #     print("> Checking file consistency...")
    #     files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
    #     self.expect_true(files_consistent, error_msg)
    #
    #     print("done!\n")
    #
    # def test_025_gradient_estimation_methods(self):
    #     """
    #     Test gradient estimation methods
    #     """
    #
    #     global folder, plot, save_session_format
    #     test_name = 'pygpc_test_025_gradient_estimation_methods'
    #     print(test_name)
    #
    #     methods_options = dict()
    #     methods = ["FD_fwd", "FD_1st", "FD_2nd", "FD_1st2nd"]
    #     methods_options["FD_fwd"] = {"dx": 0.001, "distance_weight": -2}
    #     methods_options["FD_1st"] = {"dx": 0.1, "distance_weight": -2}
    #     methods_options["FD_2nd"] = {"dx": 0.1, "distance_weight": -2}
    #     methods_options["FD_1st2nd"] = {"dx": 0.1, "distance_weight": -2}
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks()
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 0.5
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # define grid
    #     n_grid = 1000
    #     grid = pygpc.Random(parameters_random=problem.parameters_random,
    #                         n_grid=n_grid,
    #                         options={"seed": 1})
    #
    #     # create grid points for finite difference approximation
    #     grid.create_gradient_grid(delta=1e-3)
    #
    #     # evaluate model function
    #     com = pygpc.Computation(n_cpu=0, matlab_model=False)
    #
    #     # [n_grid x n_out]
    #     res = com.run(model=model,
    #                   problem=problem,
    #                   coords=grid.coords,
    #                   coords_norm=grid.coords_norm,
    #                   i_iter=None,
    #                   i_subiter=None,
    #                   fn_results=None,
    #                   print_func_time=False)
    #
    #     grad_res = dict()
    #     gradient_idx = dict()
    #     for m in methods:
    #         # [n_grid x n_out x dim]
    #         grad_res[m], gradient_idx[m] = pygpc.get_gradient(model=model,
    #                                                           problem=problem,
    #                                                           grid=grid,
    #                                                           results=res,
    #                                                           com=com,
    #                                                           method=m,
    #                                                           gradient_results_present=None,
    #                                                           gradient_idx_skip=None,
    #                                                           i_iter=None,
    #                                                           i_subiter=None,
    #                                                           print_func_time=False,
    #                                                           dx=methods_options[m]["dx"],
    #                                                           distance_weight=methods_options[m]["distance_weight"])
    #
    #         if m != "FD_fwd":
    #             nrmsd = pygpc.nrmsd(grad_res[m][:, 0, :], grad_res["FD_fwd"][gradient_idx[m], 0, :])
    #             self.expect_true((nrmsd < 0.05).all(),
    #                              msg="gPC test failed during gradient estimation: {} error too large".format(m))
    #
    #     print("done!\n")


if __name__ == '__main__':
    unittest.main()
