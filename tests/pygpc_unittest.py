# -*- coding: utf-8 -*-
"""
Unittest class of pygpc
@author: Konstantin Weise
"""
import unittest
import pygpc
from collections import OrderedDict
import numpy as np
from scipy.interpolate import griddata
import sys
import os

# temporary output folder
folder = '/home/kporzig/tmp'
# folder = '/NOBACKUP2/tmp'


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

    def test_1_static_gpc_reg_mp_randomgrid(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: RandomGrid
        """
        global folder
        test_name = 'pygpc_test_1_static_reg_mp_randomgrid'
        print(test_name)

        # define model
        model = pygpc.testfunctions.PeaksSingle

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
        parameters["x2"] = 1.25
        parameters["x3"] = pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "Moore-Penrose"
        options["settings"] = None
        options["order"] = [7, 7]
        options["order_max"] = 7
        options["interaction_order"] = 2
        options["n_cpu"] = 8
        options["fn_results"] = os.path.join(folder, test_name)

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)
        grid = pygpc.RandomGrid(problem=problem,
                                parameters={"n_grid": 1.5 * n_coeffs, "seed": 1})

        pygpc.plot_2d_grid(coords=grid.coords,
                           fn_plot=os.path.join(folder, test_name + '_grid'))

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        # # Validate gPC vs original model function
        # nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
        #                               coeffs=coeffs,
        #                               n_samples=int(1e4),
        #                               output_idx=0,
        #                               fn_pdf=os.path.join(folder, test_name))

        # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))

        # Validate gPC vs original model function
        pygpc.validate_gpc_plot(gpc=gpc,
                              coeffs=coeffs,
                              random_vars=["x1", "x3"],
                              n_grid=[25, 10],
                              output_idx=0,
                              fn_out=os.path.join(folder, test_name + '_2d'))

        pygpc.validate_gpc_plot(gpc=gpc,
                              coeffs=coeffs,
                              random_vars=["x1"],
                              n_grid=[25, 10],
                              output_idx=0,
                              fn_out=os.path.join(folder, test_name + '_2d'))

        # self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))

        print("done!\n")
    #
    # def test_2_static_gpc_reg_omp_randomgrid(self):
    #     """
    #     Algorithm: Static
    #     Method: Regression
    #     Solver: OMP
    #     Grid: RandomGrid
    #     """
    #     global folder
    #     test_name = 'pygpc_test_2_static_reg_omp_randomgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
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
    #     grid = pygpc.RandomGrid(problem=problem,
    #                             parameters={"n_grid": options["settings"]["n_coeffs_sparse"]*np.log10(n_coeffs),
    #                                         "seed": 1})
    #
    #     pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_pdf=os.path.join(folder, test_name))
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
    #     test_name = 'pygpc_test_3_static_quad_numint_tensorgrid'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
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
    #     grid = pygpc.TensorGrid(problem=problem,
    #                             parameters={"grid_type": ["jacobi", "hermite"], "n_dim": options["order"]})
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
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_pdf=os.path.join(folder, test_name))
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
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
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
    #     grid = pygpc.SparseGrid(problem=problem,
    #                             parameters={"grid_type": ["jacobi", "jacobi"],
    #                                         "level": [3, 3],
    #                                         "level_max": 3,
    #                                         "interaction_order": options["interaction_order"],
    #                                         "order_sequence_type": "exp"})
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
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_pdf=os.path.join(folder, test_name))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

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
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     problem = pygpc.Problem(model, parameters)
    #
    #     # gPC options
    #     options = dict()
    #     options["order_start"] = 2
    #     options["order_end"] = 10
    #     options["interaction_order"] = 2
    #     options["solver"] = "Moore-Penrose"
    #     options["settings"] = None
    #     options["seed"] = 1
    #     options["matrix_ratio"] = 1.5
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
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_pdf=os.path.join(folder, test_name))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")

    # def test_6_adaptive_gpc_reg_mp_randomgrid(self):
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
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    #     parameters["x2"] = 1.25
    #     parameters["x3"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
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
    #     # Validate gPC vs original model function
    #     nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
    #                                   coeffs=coeffs,
    #                                   n_samples=int(1e4),
    #                                   output_idx=0,
    #                                   fn_pdf=os.path.join(folder, test_name))
    #
    #     print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
    #
    #     self.expect_true(np.max(nrmsd) < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))
    #
    #     print("done!\n")


if __name__ == '__main__':
    unittest.main()
