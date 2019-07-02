# -*- coding: utf-8 -*-
"""
Unittest class of pygpc
@author: Konstantin Weise
"""
import unittest
import pygpc
from collections import OrderedDict
import numpy as np
import sys
import os

# temporary output folder
# folder = "/home/kporzig/tmp"
folder = "/NOBACKUP2/tmp"


# first test fixture (class)
class TestpygpcMethods(unittest.TestCase):

    # setup method called before every test-case
    def setUp(self):
        pass

    def run(self, result=None):
        self._result = result
        self._num_expectations = 0
        super(TestpygpcMethods, self).run(result)

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

    # def test_1_static_gpc_reg_mp_randomgrid(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_1_static_reg_mp_randomgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Norm(pdf_shape=[0.1, 0.15])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["error_type"] = "nrmsd"
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_enhanced"] = True
    #     options["GPU"] = False
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": options["matrix_ratio"] * n_coeffs, "seed": 1})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="standard")
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (2D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x3", "x1"],
    #                             n_grid=[10, 25],
    #                             output_idx=[0],
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x3"],
    #                             n_grid=[125],
    #                             output_idx=[0],
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'))
    #
    #     self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_2_static_gpc_reg_omp_randomgrid(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #
    #     global folder
    #     test_name = r'pygpc_test_2_static_reg_omp_randomgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Norm(pdf_shape=[0.1, 0.15])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "OMP"
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # number of gPC coefficients
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     # we assume sparsity of 50%
    #     sparsity = 0.5
    #     options["settings"] = {"n_coeffs_sparse": np.ceil(sparsity*n_coeffs)}
    #
    #     # generate grid
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": options["settings"]["n_coeffs_sparse"]*np.log10(n_coeffs),
    #                                      "seed": 1})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name) + '_validation_mc')
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")
    #
    # def test_3_static_gpc_quad_numint_tensorgrid(self):
    #     """
    #     Algorithm: Static
    #     Method: Quadrature
    #     Solver: NumInt
    #     Grid: TensorGrid
    #     """
    #     global folder
    #     test_name = r'pygpc_test_3_static_quad_numint_tensorgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Norm(pdf_shape=[0.1, 0.15])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "quad"
    #     options["solver"] = "NumInt"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     grid = pygpc.TensorGrid(parameters_random=problem.parameters_random,
    #                             options={"grid_type": ["jacobi", "hermite"], "n_dim": options["order"]})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords,
    #                        weights=grid.weights*1e3,
    #                        fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name) + '_validation_mc')
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")
    #
    # def test_4_static_gpc_quad_numint_sparsegrid(self):
    #     """
    #     Algorithm: Static
    #     Method: Quadrature
    #     Solver: NumInt
    #     Grid: SparseGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_4_static_quad_numint_sparsegrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "quad"
    #     options["solver"] = "NumInt"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     grid = pygpc.SparseGrid(parameters_random=problem.parameters_random,
    #                             options={"grid_type": ["jacobi", "jacobi"],
    #                                      "level": [3, 3],
    #                                      "level_max": 3,
    #                                      "interaction_order": options["interaction_order"],
    #                                      "order_sequence_type": "exp"})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords,
    #                        weights=grid.weights*5e2,
    #                        fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")
    #
    # def test_5_adaptive_gpc_reg_mp_randomgrid(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_5_adaptive_gpc_reg_mp_randomgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 1
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 1
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["print_func_time"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=gpc.grid.coords,
    #                        fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling")
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # Validate gPC vs original model function (2D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x3", "x1"],
    #                             n_grid=[10, 25],
    #                             output_idx=[0, 1],
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")
    #
    # def test_6_adaptive_gpc_reg_omp_randomgrid(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_6_adaptive_gpc_reg_omp_randomgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "OMP"
    #     options["settings"] = {"sparsity": 0.5}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=gpc.grid.coords,
    #                        fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")
    #
    # def test_7_adaptive_gpc_reg_mp_randomgrid_parallel_function_evaluation(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     Info: The function evaluates the grid-points in parallel
    #     """
    #     global folder
    #     test_name = 'pygpc_test_7_adaptive_gpc_reg_mp_randomgrid_parallel_function_eval'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 1
    #     options["order_max_norm"] = 0.9
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1.5
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["print_func_time"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=gpc.grid.coords,
    #                        fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # Validate gPC vs original model function (2D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x3", "x1"],
    #                             n_grid=[10, 25],
    #                             output_idx=[0, 1],
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_8_static_gpc_reg_mp_randomgrid_HyperbolicTangent(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_8_static_reg_mp_randomgrid_HyperbolicTangent'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.HyperbolicTangent
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[31, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_9_static_gpc_reg_mp_randomgrid_MovingParticleFrictionForce(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_9_static_reg_mp_randomgrid_MovingParticleFrictionForce'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.MovingParticleFrictionForce
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [20]
    #     options["order_max"] = 20
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    #
    # def test_10_static_gpc_reg_mp_randomgrid_SurfaceCoverageSpecies(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_10_static_reg_mp_randomgrid_SurfaceCoverageSpecies'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.SurfaceCoverageSpecies
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["rho_0"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["beta"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 20.])
    #     parameters["alpha"] = 1.
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 8
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e3),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["rho_0"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["rho_0", "beta"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_11_static_gpc_reg_mp_randomgrid_Franke(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_11_static_reg_mp_randomgrid_Franke'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Franke
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [11, 11]
    #     options["order_max"] = 11
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_12_static_gpc_reg_mp_randomgrid_ManufactureDecay(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_12_static_reg_mp_randomgrid_ManufactureDecay'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Peaks
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[5, 5], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[4, 2], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 3
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "standard_forward"
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e4
    #     options["n_grid_gradient"] = 4
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": options["matrix_ratio"] * n_coeffs, "seed": 1})
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # test sobol calculation
    #     sobol_ref, sobol_idx_ref, sobol_idx_bool_ref = gpc.get_sobol_indices(coeffs=coeffs, algorithm="standard", n_samples=1e4)
    #     sobol, sobol_idx, sobol_idx_bool = gpc.get_sobol_indices(coeffs=coeffs, algorithm="sampling", n_samples=5e4)
    #
    #     var = gpc.get_standard_deviation(coeffs)**2
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
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # # Validate gPC vs original model function (1D-slice)
    #     # pygpc.validate_gpc_plot(gpc=gpc,
    #     #                         coeffs=coeffs,
    #     #                         random_vars=["x1"],
    #     #                         n_grid=[51],
    #     #                         output_idx=0,
    #     #                         fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #     #                         n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[101, 101],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_13_static_gpc_reg_mp_randomgrid_GenzContinuous(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_13_static_reg_mp_randomgrid_GenzContinuous'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzContinuous
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_14_static_gpc_reg_mp_randomgrid_GenzCornerPeak(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_14_static_reg_mp_randomgrid_GenzCornerPeak'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzCornerPeak
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_15_static_gpc_reg_mp_randomgrid_GenzDiscontinuous(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_15_static_reg_mp_randomgrid_GenzDiscontinuous'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzDiscontinuous
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_16_static_gpc_reg_mp_randomgrid_GenzGaussianPeak(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_16_static_reg_mp_randomgrid_GenzGaussianPeak'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzGaussianPeak
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_17_staticprojection_gpc_reg_mp_randomgrid_GenzOscillatory(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_17_staticprojection_reg_mp_randomgrid_GenzOscillatory'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzOscillatory
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [10]
    #     options["order_max"] = 10
    #     options["interaction_order"] = 1
    #     options["n_cpu"] = 0
    #     options["error_type"] = "nrmsd"
    #     options["error_norm"] = "relative"
    #     options["matrix_ratio"] = 2
    #     options["qoi"] = 1
    #     options["n_grid_gradient"] = 5
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_enhanced"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.StaticProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True,
    #                                  algorithm="sampling")
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_18_static_gpc_reg_mp_randomgrid_GenzProductPeak(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_18_static_reg_mp_randomgrid_GenzProductPeak'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzProductPeak
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            order_glob_max_norm=options["order_max_norm"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_19_static_gpc_reg_mp_randomgrid_Ridge(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_19_static_reg_mp_randomgrid_Ridge'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ridge
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-4., 4.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-4., 4.])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-4., 4.])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-4., 4.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [12,12,12,12]
    #     options["order_max"] = 12
    #     options["order_max_norm"] = 1
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["error_norm"] = "relative"
    #     options["error_type"] = "nrmsd"
    #     options["gradient_enhanced"] = True
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            order_glob_max_norm=options["order_max_norm"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     # pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_20_static_gpc_reg_omp_randomgrid_Ishigami_2D(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_20_RegAdaptive_reg_omp_randomgrid_Ishigami_2D'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ishigami
    #
    #     # define parameters
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x3"] = 0.
    #     parameters["a"] = 7.
    #     parameters["b"] = 0.1
    #     problem = pygpc.Problem(model, parameters)
    #     problem.create_validation_set(n_samples=int(1e4), n_cpu=0)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 1
    #     options["order_end"] = 14
    #     options["interaction_order"] = 2
    #     options["solver"] = "OMP"
    #     options["settings"] = {"sparsity": 0.5}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["print_func_time"] = True
    #     options["error_norm"] = "relative"
    #     options["error_type"] = "loocv"
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_21_static_gpc_reg_mp_randomgrid_Ishigami_2D(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_21_RegAdaptive_reg_mp_randomgrid_Ishigami_2D'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ishigami
    #
    #     # define parameters
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x3"] = 0.
    #     parameters["a"] = 7.
    #     parameters["b"] = 0.1
    #     problem = pygpc.Problem(model, parameters)
    #     problem.create_validation_set(n_samples=int(1e4), n_cpu=0)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 1
    #     options["order_end"] = 14
    #     options["interaction_order"] = 2
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["print_func_time"] = True
    #     options["error_norm"] = "relative"
    #     options["error_type"] = "nrmsd"
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


    # def test_22_static_gpc_reg_mp_randomgrid_SphereFun(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_22_static_reg_mp_randomgrid_SphereFun'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.SphereFun
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["method"] = "reg"
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [7, 7]
    #     options["order_max"] = 7
    #     options["interaction_order"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 2 * n_coeffs, "seed": 1})
    #
    #     # plot grid
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.StaticProjection(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function (Monte Carlo)
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     # Validate gPC vs original model function (1D-slice)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1"],
    #                             n_grid=[51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_1d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x2"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_23_regadaptiveprojection_gpc_reg_mp_randomgrid_GenzOscillatory(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_23_regadaptiveprojection_gpc_reg_mp_randomgrid_GenzOscillatory'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.GenzOscillatory
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 15
    #     options["interaction_order"] = 2
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_calculation"] = "standard_forward"
    #     options["n_grid_gradient"] = 5
    #     options["qoi"] = 1
    #     options["error_type"] = "loocv"
    #     options["eps_lambda_gradient"] = 0.95
    #     options["gradient_enhanced"] = True
    #     options["adaptive_sampling"] = False
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=["x1", "x3"],
    #                             n_grid=[51, 51],
    #                             output_idx=0,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_24_regadaptiveprojection_gpc_reg_mp_randomgrid_Welch1992(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_24_regadaptiveprojection_gpc_reg_mp_randomgrid_Welch1992'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Welch1992
    #
    #     # define parameters
    #     parameters = OrderedDict()
    #
    #     for i in range(20):
    #         parameters["x{}".format(i + 1)] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-0.5, 0.5])
    #
    #     # define problem
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 3
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_calculation"] = "standard_forward"
    #     options["n_grid_gradient"] = 25
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.95
    #     options["gradient_enhanced"] = True
    #     options["adaptive_sampling"] = False
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_25_regadaptiveprojection_gpc_reg_mp_randomgrid_Ridge(self):
    #     """
    #     Algorithm: RegAdaptiveProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_25_regadaptiveprojection_gpc_reg_mp_randomgrid_Ridge'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ridge
    #
    #     # define parameters
    #     parameters = OrderedDict()
    #
    #     for i in range(10):
    #         parameters["x{}".format(i + 1)] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-4, 4])
    #
    #     # define problem
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 5
    #     options["order_end"] = 10
    #     options["interaction_order"] = 3
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_calculation"] = "standard_forward"
    #     options["n_grid_gradient"] = 100
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.95
    #     options["gradient_enhanced"] = True
    #     options["adaptive_sampling"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_26_regadaptiveprojection_gpc_reg_mp_randomgrid_OakleyOhagan2004(self):
    #     """
    #     Algorithm: RegAdaptiveProjection
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_26_regadaptiveprojection_gpc_reg_mp_randomgrid_OakleyOhagan2004'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.OakleyOhagan2004
    #
    #     # define parameters
    #     parameters = OrderedDict()
    #
    #     for i in range(15):
    #         parameters["x{}".format(i + 1)] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-1, 1])
    #
    #     # define problem
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 3
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 3
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["gradient_calculation"] = "standard_forward"
    #     options["n_grid_gradient"] = 100
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.05
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_27_regadaptivesampling_gpc_reg_larslasso_randomgrid_Peaks(self):
    #         """
    #         Algorithm: Adaptive
    #         Method: Regression
    #         Solver: OMP
    #         Grid: RandomGrid
    #         """
    #         global folder
    #         test_name = 'pygpc_test_27_regadaptivesampling_gpc_reg_larslasso_randomgrid_Peaks'
    #         print(test_name)
    #
    #         # define model
    #         model = pygpc.testfunctions.Peaks
    #
    #         # define problem
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #         parameters["x2"] = 1.25
    #         parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #         problem = pygpc.Problem(model, parameters)
    #
    #         # gPC options
    #         options = dict()
    #         options["order_start"] = 2
    #         options["order_end"] = 15
    #         options["interaction_order"] = 2
    #         options["solver"] = "LarsLasso"  # "OMP"
    #         options["settings"] = {"alpha": 1e-5}  # {"sparsity": 0.25}
    #         options["seed"] = 1
    #         options["matrix_ratio"] = 1
    #         options["n_cpu"] = 0
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["adaptive_sampling"] = True
    #         options["n_grid_gradient"] = 25
    #         options["qoi"] = 0
    #         options["eps_lambda_gradient"] = 0.1
    #         options["gradient_calculation"] = "standard_forward"
    #         options["error_type"] = "nrmsd"
    #         options["error_norm"] = "relative"
    #         options["gradient_enhanced"] = True
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #         # run gPC algorithm
    #         gpc, coeffs, results = algorithm.run()
    #
    #         # plot grid
    #         pygpc.plot_2d_grid(coords=gpc.grid.coords,
    #                            fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True)
    #
    #         # Validate gPC vs original model function
    #         nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #         # Validate gPC vs original model function (2D-surface)
    #         pygpc.validate_gpc_plot(gpc=gpc,
    #                                 coeffs=coeffs,
    #                                 random_vars=["x1", "x3"],
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                                 n_cpu=options["n_cpu"])
    #
    #         print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #         self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #         print("done!\n")

    # def test_28_regadaptivesampling_gpc_reg_larslasso_randomgrid_SphereFun(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_28_regadaptivesampling_gpc_reg_larslasso_randomgrid_SphereFun'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.SphereFun
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-1, 1])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = {"alpha": 1e-5}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 0.2
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["adaptive_sampling"] = True
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_29_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_Ridge(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_29_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_Ridge'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ridge
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-4, 4])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-4, 4])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-4, 4])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-4, 4])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "LarsLasso"
    #     options["settings"] = {"alpha": 1e-5}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["adaptive_sampling"] = False
    #     options["n_grid_gradient"] = 3
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.1
    #     options["gradient_calculation"] = "standard_forward"
    #     options["error_type"] = "loocv"
    #     options["error_norm"] = "relative"
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_30_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_Ishigami(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_30_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_Ishigami'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.Ishigami
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #     parameters["a"] = 7.
    #     parameters["b"] = 0.1
    #     problem = pygpc.Problem(model, parameters)
    #     problem.create_validation_set(n_samples=1e4, n_cpu=0)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 4
    #     options["order_end"] = 15
    #     options["interaction_order"] = 3
    #     options["solver"] = "LarsLasso"  # "OMP"
    #     options["settings"] = {"alpha": 1e-5}  # {"sparsity": 0.25}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["adaptive_sampling"] = True
    #     options["n_grid_gradient"] = 25
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.1
    #     options["gradient_calculation"] = "standard_forward"
    #     options["error_type"] = "loocv"
    #     options["error_norm"] = "relative"
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_31_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_ManufactureDecay(self):
    #     """
    #     Algorithm: Adaptive
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_31_regadaptiveprojectionsampling_gpc_reg_larslasso_randomgrid_ManufactureDecay'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.ManufactureDecay
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x3"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     parameters["x4"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 15
    #     options["interaction_order"] = 2
    #     options["solver"] = "LarsLasso"  # "OMP"
    #     options["settings"] = {"alpha": 1e-5}  # {"sparsity": 0.25}
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1
    #     options["n_cpu"] = 0
    #     options["fn_results"] = os.path.join(folder, test_name)
    #     options["adaptive_sampling"] = True
    #     options["n_grid_gradient"] = 5
    #     options["qoi"] = 0
    #     options["eps_lambda_gradient"] = 0.1
    #     options["gradient_calculation"] = "standard_forward"
    #     options["error_type"] = "loocv"
    #     options["error_norm"] = "relative"
    #
    #     # define algorithm
    #     algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Post-process gPC
    #     pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                  output_idx=None,
    #                                  calc_sobol=True,
    #                                  calc_global_sens=True,
    #                                  calc_pdf=True)
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_32_regadaptivesampling_gpc_reg_larslasso_randomgrid_Lim2002(self):
    #         """
    #         Algorithm: Adaptive
    #         Method: Regression
    #         Solver: OMP
    #         Grid: RandomGrid
    #         """
    #         global folder
    #         test_name = 'pygpc_test_32_regadaptivesampling_gpc_reg_larslasso_randomgrid_Lim2002'
    #         print(test_name)
    #
    #         # define model
    #         model = pygpc.testfunctions.Lim2002
    #
    #         # define problem
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         problem = pygpc.Problem(model, parameters)
    #         problem.create_validation_set(n_samples=int(1e4), n_cpu=0)
    #
    #         # gPC options
    #         options = dict()
    #         options["order_start"] = 2
    #         options["order_end"] = 15
    #         options["interaction_order"] = 2
    #         options["solver"] = "LarsLasso"  # "OMP"
    #         options["settings"] = {"alpha": 1e-5}  # {"sparsity": 0.25}
    #         options["seed"] = 1
    #         options["matrix_ratio"] = 1.5
    #         options["n_cpu"] = 0
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["adaptive_sampling"] = False
    #         options["n_grid_gradient"] = 5
    #         options["qoi"] = 0
    #         options["eps_lambda_gradient"] = 0.95
    #         options["gradient_calculation"] = "standard_forward"
    #         options["error_type"] = "loocv"
    #         options["error_norm"] = "relative"
    #         options["gradient_enhanced"] = True
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #         # run gPC algorithm
    #         gpc, coeffs, results = algorithm.run()
    #
    #         # plot grid
    #         pygpc.plot_2d_grid(coords=gpc.grid.coords,
    #                            fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True)
    #
    #         # Validate gPC vs original model function
    #         nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #         print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #         self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #         print("done!\n")

    # def test_33_regadaptivesampling_gpc_reg_larslasso_randomgrid_Franke(self):
    #         """
    #         Algorithm: Adaptive
    #         Method: Regression
    #         Solver: OMP
    #         Grid: RandomGrid
    #         """
    #         global folder
    #         test_name = 'pygpc_test_33_regadaptivesampling_gpc_reg_larslasso_randomgrid_Franke'
    #         print(test_name)
    #
    #         # define model
    #         model = pygpc.testfunctions.Franke
    #
    #         # define problem
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         problem = pygpc.Problem(model, parameters)
    #         problem.create_validation_set(n_samples=int(1e4), n_cpu=0)
    #
    #         # gPC options
    #         options = dict()
    #         options["order_start"] = 2
    #         options["order_end"] = 15
    #         options["interaction_order"] = 2
    #         options["solver"] = "LarsLasso"  # "OMP"
    #         options["settings"] = {"alpha": 1e-5}  # {"sparsity": 0.25}
    #         options["seed"] = 1
    #         options["matrix_ratio"] = 1.5
    #         options["n_cpu"] = 0
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["adaptive_sampling"] = False
    #         options["n_grid_gradient"] = 5
    #         options["qoi"] = 0
    #         options["eps_lambda_gradient"] = 0.95
    #         options["gradient_calculation"] = "standard_forward"
    #         options["error_type"] = "loocv"
    #         options["error_norm"] = "relative"
    #         options["gradient_enhanced"] = True
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)
    #
    #         # run gPC algorithm
    #         gpc, coeffs, results = algorithm.run()
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True)
    #
    #         # Validate gPC vs original model function
    #         nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #         print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #         self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #         print("done!\n")

    # def test_34_StaticGradientEnhanced_gpc_reg_mp_randomgrid_Lim2002(self):
    #         """
    #         Algorithm: Static (gradient enhanced)
    #         Method: Regression
    #         Solver: Moore-Penrose
    #         Grid: RandomGrid
    #         """
    #         global folder
    #         test_name = 'pygpc_test_1_static_reg_mp_randomgrid'
    #         print(test_name)
    #
    #         # define model
    #         model = pygpc.testfunctions.Lim2002
    #
    #         # define problem
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    #         problem = pygpc.Problem(model, parameters)
    #
    #         # gPC options
    #         options = dict()
    #         options["method"] = "reg"
    #         options["solver"] = "Moore-Penrose"
    #         options["settings"] = None
    #         options["order"] = [5, 5]
    #         options["order_max"] = 5
    #         options["interaction_order"] = 2
    #         options["matrix_ratio"] = 2
    #         options["n_cpu"] = 0
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["gradient_enhanced"] = True
    #         options["gradient_calculation"] = "standard_forward"
    #
    #         # generate grid
    #         n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                                order_glob_max=options["order_max"],
    #                                                order_inter_max=options["interaction_order"],
    #                                                dim=problem.dim)
    #
    #         grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                                 options={"n_grid": 1.5 * n_coeffs, "seed": 1})
    #
    #         # define algorithm
    #         algorithm = pygpc.StaticProjection(problem=problem, options=options)
    #
    #         # run gPC algorithm
    #         gpc, coeffs, results = algorithm.run()
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True)
    #
    #         # Validate gPC vs original model function (Monte Carlo)
    #         nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #         print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #         # Validate gPC vs original model function (2D-slice)
    #         pygpc.validate_gpc_plot(gpc=gpc,
    #                                 coeffs=coeffs,
    #                                 random_vars=["x3", "x1"],
    #                                 n_grid=[10, 25],
    #                                 output_idx=[0, 1],
    #                                 fn_out=os.path.join(folder, test_name + '_validation_2d'))
    #
    #         # Validate gPC vs original model function (1D-slice)
    #         pygpc.validate_gpc_plot(gpc=gpc,
    #                                 coeffs=coeffs,
    #                                 random_vars=["x3"],
    #                                 n_grid=[125],
    #                                 output_idx=[0, 1],
    #                                 fn_out=os.path.join(folder, test_name + '_validation_1d'))
    #
    #         self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #         print("done!\n")

    # def test_35_RegAdaptiveGradientEnhanced_reg_mp_randomgrid_Ishigami_2D(self):
    #         """
    #         Algorithm: RegAdaptive (gradient enhanced)
    #         Method: Regression
    #         Solver: Moore-Penrose
    #         Grid: RandomGrid
    #         """
    #         global folder
    #         test_name = 'pygpc_test_35_RegAdaptiveGradientEnhanced_reg_mp_randomgrid_Ishigami_2D'
    #         print(test_name)
    #
    #         # define model
    #         model = pygpc.testfunctions.Ishigami
    #
    #         # define parameters
    #         parameters = OrderedDict()
    #         parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #         parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
    #         parameters["x3"] = 0.
    #         parameters["a"] = 7.
    #         parameters["b"] = 0.1
    #         problem = pygpc.Problem(model, parameters)
    #         problem.create_validation_set(n_samples=int(1e4), n_cpu=0)
    #
    #         # gPC options
    #         options = dict()
    #         options["order_start"] = 1
    #         options["order_end"] = 14
    #         options["interaction_order"] = 2
    #         options["solver"] = "LarsLasso"
    #         options["settings"] = None
    #         options["seed"] = 1
    #         options["matrix_ratio"] = 2
    #         options["n_cpu"] = 0
    #         options["fn_results"] = os.path.join(folder, test_name)
    #         options["print_func_time"] = True
    #         options["error_norm"] = "relative"
    #         options["error_type"] = "nrmsd"
    #         options["adaptive_sampling"] = True
    #         options["gradient_enhanced"] = True
    #         options["gradient_calculation"] = "standard_forward"
    #
    #         # define algorithm
    #         algorithm = pygpc.RegAdaptive(problem=problem, options=options)
    #
    #         # run gPC algorithm
    #         gpc, coeffs, results = algorithm.run()
    #
    #         # Post-process gPC
    #         pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
    #                                      output_idx=None,
    #                                      calc_sobol=True,
    #                                      calc_global_sens=True,
    #                                      calc_pdf=True)
    #
    #         # Validate gPC vs original model function (Monte Carlo)
    #         nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                       coeffs=coeffs,
    #                                       n_samples=int(1e4),
    #                                       output_idx=0,
    #                                       n_cpu=options["n_cpu"],
    #                                       fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #         print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #         # Validate gPC vs original model function (1D-slice)
    #         pygpc.validate_gpc_plot(gpc=gpc,
    #                                 coeffs=coeffs,
    #                                 random_vars=["x1", "x2"],
    #                                 n_grid=[51, 51],
    #                                 output_idx=0,
    #                                 fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                                 n_cpu=options["n_cpu"])
    #
    #         # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #         print("done!\n")

    # def test_36_static_megpc_reg_mp_randomgrid_BinaryDiscontinuousSphere(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_36_static_megpc_reg_mp_randomgrid_SurfaceCoverageSpecies'
    #     print(test_name)
    #
    #     # define model
    #     # model = pygpc.testfunctions.GenzDiscontinuous
    #     model = pygpc.testfunctions.SurfaceCoverageSpecies
    #     # model = pygpc.testfunctions.HyperbolicTangent
    #     # model = pygpc.testfunctions.BinaryDiscontinuousSphere
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [9, 9]
    #     options["order_max"] = 9
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "standard_forward"
    #     options["error_type"] = "loocv"
    #     options["n_samples_validation"] = 1e4
    #     options["qoi"] = "all"
    #     options["n_grid_gradient"] = 5
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 2,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #
    #     grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
    #                             options={"n_grid": 200, "seed": 1})  # options["matrix_ratio"] * n_coeffs
    #
    #     # define algorithm
    #     algorithm = pygpc.MEStatic(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     # pygpc.validate_gpc_plot(gpc=gpc,
    #     #                         coeffs=coeffs,
    #     #                         random_vars=list(problem.parameters_random.keys()),
    #     #                         n_grid=[51, 51],
    #     #                         output_idx=1,
    #     #                         fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #     #                         n_cpu=options["n_cpu"])
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
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #     print("done!\n")

    # def test_37_staticprojection_megpc_reg_mp_randomgrid_DiscontinuousRidgeManufactureDecay(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: Moore-Penrose
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_37_staticprojection_megpc_reg_mp_randomgrid_DiscontinuousRidgeManufactureDecay'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay
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
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["order"] = [3, 3]
    #     options["order_max"] = 3
    #     options["interaction_order"] = 2
    #     options["matrix_ratio"] = 2
    #     options["n_cpu"] = 0
    #     options["gradient_enhanced"] = True
    #     options["gradient_calculation"] = "standard_forward"
    #     options["n_grid_gradient"] = 50
    #     options["error_type"] = "nrmsd"
    #     options["n_samples_validation"] = 1e4
    #     options["qoi"] = "all"
    #     options["classifier"] = "learning"
    #     options["classifier_options"] = {"clusterer": "KMeans",
    #                                      "n_clusters": 2,
    #                                      "classifier": "MLPClassifier",
    #                                      "classifier_solver": "lbfgs"}
    #     options["fn_results"] = os.path.join(folder, test_name)
    #
    #     # define algorithm
    #     algorithm = pygpc.MEStaticProjection(problem=problem, options=options)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Validate gPC vs original model function (2D-surface)
    #     pygpc.validate_gpc_plot(gpc=gpc,
    #                             coeffs=coeffs,
    #                             random_vars=list(problem.parameters_random.keys()),
    #                             n_grid=[101, 101],
    #                             output_idx=1,
    #                             fn_out=os.path.join(folder, test_name + '_validation_2d'),
    #                             n_cpu=options["n_cpu"])
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
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e5),
    #                                   output_idx=0,
    #                                   n_cpu=options["n_cpu"],
    #                                   fn_out=os.path.join(folder, test_name + '_validation_mc'))
    #
    #     # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #     print("done!\n")

    def test_38_MERegAdaptive_DiscontinuousRidgeManufactureDecay(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder
        test_name = 'pygpc_test_38_MERegAdaptive_DiscontinuousRidgeManufactureDecay'
        print(test_name)

        # define model
        # model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay
        model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay
        # model = pygpc.testfunctions.BinaryDiscontinuousSphere

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order_start"] = 2
        options["order_end"] = 15
        options["interaction_order"] = 2
        options["matrix_ratio"] = 2
        options["projection"] = True
        options["n_cpu"] = 0
        options["gradient_enhanced"] = False
        options["gradient_calculation"] = "standard_forward"
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e4
        options["qoi"] = "all"
        options["classifier"] = "learning"
        options["classifier_options"] = {"clusterer": "KMeans",
                                         "n_clusters": 2,
                                         "classifier": "MLPClassifier",
                                         "classifier_solver": "lbfgs"}
        options["n_samples_discontinuity"] = 5
        options["adaptive_sampling"] = False
        options["eps"] = 0.01
        options["n_grid_init"] = 20
        options["GPU"] = True
        options["fn_results"] = os.path.join(folder, test_name)

        # define algorithm
        algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        # Validate gPC vs original model function (2D-surface)
        pygpc.validate_gpc_plot(gpc=gpc,
                                coeffs=coeffs,
                                random_vars=list(problem.parameters_random.keys()),
                                n_grid=[101, 101],
                                output_idx=0,
                                fn_out=os.path.join(folder, test_name + '_validation_2d'),
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
                                      n_samples=int(1e5),
                                      output_idx=0,
                                      n_cpu=options["n_cpu"],
                                      fn_out=os.path.join(folder, test_name + '_validation_mc'))

        # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
        print("done!\n")


if __name__ == '__main__':
    unittest.main()
