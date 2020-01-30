import numpy as np
import unittest
import shutil
import pygpc
import time
import h5py
import sys
import os
from collections import OrderedDict

# disable numpy warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# test options
folder = 'tmp'      # output folder
plot = False        # plot and save output
matlab = False      # test Matlab functionality

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

    def expect_true(self, a, msg):
        if not a:
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def test_0_Static_gpc_quad(self):
        """
        Algorithm: Static
        Method: Quadrature
        Solver: NumInt
        Grid: TensorGrid
        """
        global folder, plot
        test_name = 'pygpc_test_0_Static_gpc_quad'
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
        options["backend"] = "omp"
        # options["backend"] = "cuda"
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

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard",
                                     n_samples=1e3)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=[0, 1],
                                    fn_out=options["fn_results"] + "_val",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=[0, 1],
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_1_Static_gpc(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_1_Static_gpc'
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
        options["matrix_ratio"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True
        options["backend"] = "omp"
        # options["backend"] = "cuda"
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)

        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=options["matrix_ratio"] * n_coeffs,
                            seed=1)

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard",
                                     n_samples=1e3)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)

        print("done!\n")

    def test_2_MEStatic_gpc(self):
        """
        Algorithm: MEStatic
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_2_MEStatic_gpc'
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
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["n_cpu"] = 0
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "standard_forward"
        options["error_type"] = "loocv"
        options["qoi"] = "all"
        options["n_grid_gradient"] = 5
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # generate grid
        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=200,  # options["matrix_ratio"] * n_coeffs
                            seed=1)

        # define algorithm
        algorithm = pygpc.MEStatic(problem=problem, options=options, grid=grid)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
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
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_3_StaticProjection_gpc(self):
        """
        Algorithm: StaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_3_StaticProjection_gpc'
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
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [10]
        options["order_max"] = 10
        options["interaction_order"] = 1
        options["n_cpu"] = 0
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["error_norm"] = "relative"
        options["matrix_ratio"] = 2
        options["qoi"] = 0
        options["n_grid_gradient"] = 50
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.StaticProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
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
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_4_MEStaticProjection_gpc(self):
        """
        Algorithm: MEStaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_4_MEStaticProjection_gpc'
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
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [3, 3]
        options["order_max"] = 3
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["n_cpu"] = 0
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "standard_forward"
        options["n_grid_gradient"] = 200
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.MEStaticProjection(problem=problem, options=options)

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
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
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
                                      n_samples=int(5e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_5_RegAdaptive_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_5_RegAdaptive_gpc'
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
        options["order_start"] = 5
        options["order_end"] = 20
        options["solver"] = "LarsLasso"
        options["interaction_order"] = 2
        options["order_max_norm"] = 0.7
        options["n_cpu"] = 0
        options["adaptive_sampling"] = True
        options["gradient_enhanced"] = True
        options["fn_results"] = os.path.join(folder, test_name)
        options["eps"] = 0.0075
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem, options=options)

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
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
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
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_6_RegAdaptiveProjection_gpc(self):
        """
        Algorithm: RegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_6_RegAdaptiveProjection_gpc'
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
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["seed"] = 1
        options["matrix_ratio"] = 2
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_calculation"] = "standard_forward"
        options["n_grid_gradient"] = 5
        options["qoi"] = 0
        options["error_type"] = "loocv"
        options["eps"] = 1e-3
        options["eps_lambda_gradient"] = 0.95
        options["gradient_enhanced"] = True
        options["adaptive_sampling"] = False
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)

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
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
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
                                      smooth_pdf=False,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_7_MERegAdaptiveProjection_gpc(self):
        """
        Algorithm: MERegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_7_MERegAdaptiveProjection_gpc'
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
        options["solver"] = "LarsLasso" #"Moore-Penrose"
        options["settings"] = None
        options["order_start"] = 3
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["projection"] = False
        options["n_cpu"] = 0
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "standard_forward"
        options["error_type"] = "loocv"
        options["error_norm"] = "absolute" # "relative"
        options["qoi"] = 0 # "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["n_samples_discontinuity"] = 10
        options["adaptive_sampling"] = False
        options["eps"] = 0.75
        options["n_grid_init"] = 20
        options["backend"] = "omp"
        # options["backend"] = "cuda"
        options["fn_results"] = os.path.join(folder, test_name)
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        # define algorithm
        algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC session
        session, coeffs, results = session.run()

        # Validate gPC vs original model function (Monte Carlo)
        # plot = True
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=[0],
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=[0, 1],
                                    fn_out=options["fn_results"] + "_val",
                                    n_cpu=options["n_cpu"])

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=1e4)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_8_testfunctions(self):
        """
        Testing testfunctions (multi-threading and inherited parallelization)
        """
        test_name = 'pygpc_test_9_testfunctions'
        print(test_name)

        tests = []
        tests.append(pygpc.Ackley())
        tests.append(pygpc.BukinFunctionNumber6())
        tests.append(pygpc.CrossinTrayFunction())
        tests.append(pygpc.BohachevskyFunction1())
        tests.append(pygpc.PermFunction())
        tests.append(pygpc.SixHumpCamelFunction())
        tests.append(pygpc.RotatedHyperEllipsoid())
        tests.append(pygpc.SumOfDifferentPowersFunction())
        tests.append(pygpc.ZakharovFunction())
        tests.append(pygpc.DropWaveFunction())
        tests.append(pygpc.DixonPriceFunction())
        tests.append(pygpc.RosenbrockFunction())
        tests.append(pygpc.MichalewiczFunction())
        tests.append(pygpc.DeJongFunctionFive())
        tests.append(pygpc.MatyasFunction())
        tests.append(pygpc.GramacyLeeFunction())
        tests.append(pygpc.SchafferFunction4())
        tests.append(pygpc.SphereFunction())
        tests.append(pygpc.McCormickFunction())
        tests.append(pygpc.BoothFunction())
        tests.append(pygpc.Peaks())
        tests.append(pygpc.Franke())
        tests.append(pygpc.Lim2002())
        tests.append(pygpc.Ishigami())
        tests.append(pygpc.ManufactureDecay())
        tests.append(pygpc.GenzContinuous())
        tests.append(pygpc.GenzCornerPeak())
        tests.append(pygpc.GenzOscillatory())
        tests.append(pygpc.GenzProductPeak())
        tests.append(pygpc.Ridge())
        tests.append(pygpc.OakleyOhagan2004())
        tests.append(pygpc.Welch1992())
        tests.append(pygpc.HyperbolicTangent())
        tests.append(pygpc.MovingParticleFrictionForce())
        tests.append(pygpc.SurfaceCoverageSpecies())

        for n_cpu in [4, 0]:
            if n_cpu != 0:
                print("Running testfunctions using multi-threading with {} cores...".format(n_cpu))
            else:
                print("Running testfunctions using inherited function parallelization...")

            com = pygpc.Computation(n_cpu=n_cpu)

            for t in tests:
                grid = pygpc.Random(parameters_random=t.problem.parameters_random,
                                    n_grid=10,
                                    seed=1)

                res = com.run(model=t.problem.model,
                              problem=t.problem,
                              coords=grid.coords,
                              coords_norm=grid.coords_norm,
                              i_iter=None,
                              i_subiter=None,
                              fn_results=None,
                              print_func_time=False)

            com.close()

            print("done!\n")

    def test_9_RandomParameters(self):
        """
        Testing RandomParameters
        """
        global folder, plot
        test_name = 'pygpc_test_10_RandomParameters'
        print(test_name)

        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x2"] = pygpc.Beta(pdf_shape=[5, 5], pdf_limits=[0, 1])
        parameters["x3"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[0, 1])
        parameters["x4"] = pygpc.Beta(pdf_shape=[2, 5], pdf_limits=[0, 1])
        parameters["x5"] = pygpc.Beta(pdf_shape=[0.75, 0.75], pdf_limits=[0, 1])
        parameters["x6"] = pygpc.Norm(pdf_shape=[5, 1])

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = parameters["x1"].plot_pdf()
            ax = parameters["x2"].plot_pdf()
            ax = parameters["x3"].plot_pdf()
            ax = parameters["x4"].plot_pdf()
            ax = parameters["x5"].plot_pdf()
            ax = parameters["x6"].plot_pdf()
            ax.legend(["x1", "x2", "x3", "x4", "x5", "x6"])
            ax.savefig(os.path.join(folder, test_name) + ".png")

            print("done!\n")

    def test_10_Grids(self):
        """
        Testing Grids
        """
        global folder, plot
        test_name = 'pygpc_test_11_Grids'
        print(test_name)

        test = pygpc.Peaks()

        grids = []
        fn_out = []
        grids.append(pygpc.Random(parameters_random=test.problem.parameters_random,
                                  n_grid=100,
                                  seed=1))
        fn_out.append(test_name + "_Random")
        grids.append(pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
                                      options={"grid_type": ["hermite", "jacobi"], "n_dim": [5, 10]}))
        fn_out.append(test_name + "_TensorGrid_1")
        grids.append(pygpc.TensorGrid(parameters_random=test.problem.parameters_random,
                                      options={"grid_type": ["patterson", "fejer2"], "n_dim": [3, 10]}))
        fn_out.append(test_name + "_TensorGrid_2")
        grids.append(pygpc.SparseGrid(parameters_random=test.problem.parameters_random,
                                      options={"grid_type": ["jacobi", "jacobi"],
                                               "level": [3, 3],
                                               "level_max": 3,
                                               "interaction_order": 2,
                                               "order_sequence_type": "exp"}))
        fn_out.append(test_name + "_SparseGrid")

        if plot:
            for i, g in enumerate(grids):
                pygpc.plot_2d_grid(coords=g.coords_norm, weights=g.weights, fn_plot=os.path.join(folder, fn_out[i]))

        print("done!\n")

    def test_11_Matlab_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, matlab
        test_name = 'pygpc_test_12_Matlab_gpc'
        print(test_name)

        if matlab:
            import matlab.engine
            from templates.MyModel_matlab import  MyModel_matlab
            # define model
            model = MyModel_matlab(fun_path=os.path.join(pygpc.__path__[0], "testfunctions"))

            # define problem (the parameter names have to be the same as in the model)
            parameters = OrderedDict()
            parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
            parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
            parameters["x3"] = 0.
            parameters["a"] = 7.
            parameters["b"] = 0.1

            problem = pygpc.Problem(model, parameters)

            # gPC options
            options = dict()
            options["order_start"] = 5
            options["order_end"] = 20
            options["solver"] = "LarsLasso"
            options["interaction_order"] = 2
            options["order_max_norm"] = 0.7
            options["n_cpu"] = 0
            options["adaptive_sampling"] = True
            options["gradient_enhanced"] = True
            options["fn_results"] = os.path.join(folder, test_name)
            options["eps"] = 0.0075
            options["matlab_model"] = True
            options["grid"] = pygpc.Random
            options["grid_options"] = None

            # define algorithm
            algorithm = pygpc.RegAdaptive(problem=problem, options=options)

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
                                        output_idx=0,
                                        fn_out=options["fn_results"] + "_val",
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
                                          fn_out=options["fn_results"] + "_pdf",
                                          plot=plot)

            files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

            print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
            # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
            print("> Checking file consistency...")
            self.expect_true(files_consistent, error_msg)
            print("done!\n")

        else:
            print("Skipping Matlab test...")

    def test_12_random_vars_postprocessing(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_13_random_vars_postprocessing_sobol'
        print(test_name)

        # define model
        model = pygpc.testfunctions.Peaks()

        # define problem

        parameters = OrderedDict()
        # parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[1.25, 1.72])
        parameters["x1"] = pygpc.Gamma(pdf_shape=[3., 10., 1.25], p_perc=0.98)
        parameters["x2"] = pygpc.Norm(pdf_shape=[1, 1], p_perc=0.98)
        parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0.6, 1.4])
        # parameters["x3"] = pygpc.Norm(pdf_shape=[1., 0.25], p_perc=0.95)
        # parameters["x2"] = 1.

        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [4, 4, 4]
        options["order_max"] = 4
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["error_type"] = "loocv"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True
        options["backend"] = "omp"
        # options["backend"] = "cuda"

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)

        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=options["matrix_ratio"] * n_coeffs,
                            seed=1)

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # Initialize gPC Session
        session = pygpc.Session(algorithm=algorithm)

        # run gPC algorithm
        session, coeffs, results = session.run()

        # Determine Sobol indices using standard approach (gPC coefficients)
        sobol_standard, sobol_idx_standard, sobol_idx_bool_standard = session.gpc[0].get_sobol_indices(coeffs=coeffs,
                                                                                                       algorithm="standard")

        sobol_sampling, sobol_idx_sampling, sobol_idx_bool_sampling = session.gpc[0].get_sobol_indices(coeffs=coeffs,
                                                                                                       algorithm="sampling",
                                                                                                       n_samples=3e4)

        # grid = pygpc.Random(parameters_random=session.parameters_random,
        #                     n_grid=int(5e5),
        #                     seed=None)
        #
        # com = pygpc.Computation(n_cpu=0, matlab_model=session.matlab_model)
        # y_orig = com.run(model=session.model,
        #                  problem=session.problem,
        #                  coords=grid.coords,
        #                  coords_norm=grid.coords_norm,
        #                  i_iter=None,
        #                  i_subiter=None,
        #                  fn_results=None,
        #                  print_func_time=False)
        # y_gpc = session.gpc[0].get_approximation(coeffs=coeffs, x=grid.coords_norm)
        #
        # mean_gpc_coeffs = session.gpc[0].get_mean(coeffs=coeffs)
        # mean_gpc_sampling = session.gpc[0].get_mean(samples=y_gpc)
        # mean_orig = np.mean(y_orig, axis=0)
        #
        # std_gpc_coeffs = session.gpc[0].get_std(coeffs=coeffs)
        # std_gpc_sampling = session.gpc[0].get_std(samples=y_gpc)
        # std_orig = np.std(y_orig, axis=0)

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot,
                                      n_cpu=session.n_cpu)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)

        for i in range(sobol_standard.shape[0]):
            self.expect_true(np.max(np.abs(sobol_standard[i, :]-sobol_sampling[i, :])) < 0.1,
                             msg="Sobol index: {}".format(str(sobol_idx_sampling[3])))

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(session=session,
                                    coeffs=coeffs,
                                    random_vars=["x2", "x3"],
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=options["fn_results"] + "_val",
                                    n_cpu=options["n_cpu"])

        print("done!\n")

    def test_13_clustering_3_domains(self):
        """
        Algorithm: MERegAdaptiveprojection
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot
        test_name = 'pygpc_test_14_clustering_3_domains'
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
        options["gradient_calculation"] = "standard_forward"
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
        options["n_grid_init"] = 50
        options["backend"] = "omp"
        # options["backend"] = "cuda"
        options["fn_results"] = os.path.join(folder, test_name)
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
                                    fn_out=options["fn_results"] + "_val",
                                    n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=options["fn_results"] + "_pdf",
                                      plot=plot)

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="sampling",
                                     n_samples=1e4)

        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("> Checking file consistency...")
        self.expect_true(files_consistent, error_msg)
        print("done!\n")

    def test_14_backends(self):
        """
        Test the different backends ["python", "cpu", "omp", "gpu"]
        """

        global folder, gpu
        test_name = 'pygpc_test_14_backends'
        print(test_name)

        backends = ["python", "cpu", "omp", "cuda"]

        # define model
        model = pygpc.testfunctions.Peaks()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
        parameters["x2"] = 1.25
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem = pygpc.Problem(model, parameters)

        # define test grid
        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=100,
                            seed=1)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [9, 9]
        options["order_max"] = 9
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        gpc_matrix = dict()
        gpc_matrix_gradient = dict()
        pce_matrix = dict()

        print("Constructing gPC matrices with different backends:")
        for b in backends:
            try:
                options["backend"] = b

                # setup gPC
                gpc = pygpc.Reg(problem=problem,
                                order=[8, 8],
                                order_max=8,
                                order_max_norm=0.8,
                                interaction_order=2,
                                interaction_order_current=2,
                                options=options,
                                validation=None)

                gpc.grid = grid

                # init gPC matrices
                start = time.time()
                gpc.init_gpc_matrix()
                stop = time.time()

                print(b, "create gpc matrix: ", stop-start)

                # perform polynomial chaos expansion
                coeffs = np.ones([len(gpc.basis.b), 2])
                start = time.time()
                pce = gpc.get_approximation(coeffs, gpc.grid.coords_norm)
                stop = time.time()

                print(b, "polynomial chaos expansion: ", stop-start)

                gpc_matrix[b] = gpc.gpc_matrix
                gpc_matrix_gradient[b] = gpc.gpc_matrix_gradient
                pce_matrix[b] = pce

            except NotImplementedError:
                backends.remove(b)

        for b_ref in backends:
            for b_compare in backends:
                if b_compare != b_ref:
                    self.expect_isclose(gpc_matrix[b_ref], gpc_matrix[b_compare], atol=1e-6,
                                        msg="gpc matrices between "+b_ref+" and "+b_compare+" are not equal")

                    self.expect_isclose(gpc_matrix_gradient[b_ref], gpc_matrix_gradient[b_compare], atol=1e-6,
                                        msg="gpc matrices between "+b_ref+" and "+b_compare+" are not equal")

                    self.expect_isclose(pce_matrix[b_ref], pce_matrix[b_compare], atol=1e-6,
                                        msg="pce matrices between "+b_ref+" and "+b_compare+" are not equal")

        print("done!\n")

    def test_15_save_and_load_session(self):
        """
        Save and load a gPC Session
        """
        global folder, plot
        test_name = 'pygpc_test_15_save_and_load_session'
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

        # save session
        pygpc.write_session_hdf5(obj=session,
                                 fname=options["fn_results"] + "_session.hdf5")

        # load session
        # session_hdf5 = pygpc.read_gpc_hdf5(fname=options["fn_results"] + "_session.hdf5")

        # compare session
        pass


if __name__ == '__main__':
    unittest.main()
