# -*- coding: utf-8 -*-
import unittest
import pygpc
from collections import OrderedDict
import numpy as np
import sys
import os
import shutil

# temporary folder
try:
    os.mkdir('./tmp')
except FileExistsError:
    pass

folder = './tmp'
plot = False


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

    def expect_equal(self, a, b, msg=''):
        if a != b:
            msg = '({}) Expected {} to equal {}. '.format(self._num_expectations, a, b) + msg
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def expect_true(self, a, msg):
        if not a:
            self._fail(self.failureException(msg))
        self._num_expectations += 1

    def test_1_MEStatic_gpc(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_1_Static_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.Peaks

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
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True
        options["GPU"] = False

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)

        grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                                options={"n_grid": options["matrix_ratio"] * n_coeffs, "seed": 1})

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        # Post-process gPC
        pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=True,
                                     algorithm="standard",
                                     n_samples=1e3)

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

        # remove temporary directory
        shutil.rmtree('./tmp')

    def test_2_MEStatic_gpc(self):
        """
        Algorithm: MEStatic
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_2_MEStatic_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.SurfaceCoverageSpecies

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
        options["n_samples_validation"] = 1e4
        options["qoi"] = "all"
        options["n_grid_gradient"] = 5
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)

        # generate grid
        grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                                options={"n_grid": 200, "seed": 1})  # options["matrix_ratio"] * n_coeffs

        # define algorithm
        algorithm = pygpc.MEStatic(problem=problem, options=options, grid=grid)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

    def test_3_StaticProjection_gpc(self):
        """
        Algorithm: StaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_3_StaticProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.GenzOscillatory

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
        options["error_norm"] = "relative"
        options["matrix_ratio"] = 2
        options["qoi"] = 0
        options["n_grid_gradient"] = 50
        options["fn_results"] = os.path.join(folder, test_name)
        options["gradient_enhanced"] = True

        # define algorithm
        algorithm = pygpc.StaticProjection(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

    def test_4_MEStaticProjection_gpc(self):
        """
        Algorithm: MEStaticProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_4_MEStaticProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay

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
        options["n_samples_validation"] = 1e4
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["fn_results"] = os.path.join(folder, test_name)

        # define algorithm
        algorithm = pygpc.MEStaticProjection(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(5e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

    def test_5_RegAdaptive_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_5_RegAdaptive_gpc'
        print(test_name)

        # Model
        model = pygpc.testfunctions.Ishigami

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

        # define algorithm
        algorithm = pygpc.RegAdaptive(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

    def test_6_RegAdaptiveProjection_gpc(self):
        """
        Algorithm: RegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_6_RegAdaptiveProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.GenzOscillatory

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

        # define algorithm
        algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=False,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")

    def test_7_MERegAdaptiveProjection_gpc(self):
        """
        Algorithm: MERegAdaptiveProjection
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder, plot
        test_name = 'pygpc_test_7_MERegAdaptiveProjection_gpc'
        print(test_name)

        # define model
        model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay

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
        options["order_start"] = 0
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["projection"] = True
        options["n_cpu"] = 0
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "standard_forward"
        options["error_type"] = "loocv"
        options["n_samples_validation"] = 1e4
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["n_samples_discontinuity"] = 10
        options["adaptive_sampling"] = True
        options["eps"] = 0.01
        options["n_grid_init"] = 10
        options["GPU"] = False
        options["fn_results"] = os.path.join(folder, test_name)

        # define algorithm
        algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        if plot:
            # Validate gPC vs original model function (2D-surface)
            pygpc.validate_gpc_plot(gpc=gpc,
                                    coeffs=coeffs,
                                    random_vars=list(problem.parameters_random.keys()),
                                    n_grid=[51, 51],
                                    output_idx=0,
                                    fn_out=None,
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
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      smooth_pdf=True,
                                      fn_out=None,
                                      plot=plot)

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))
        print("done!\n")


if __name__ == '__main__':
    unittest.main()
