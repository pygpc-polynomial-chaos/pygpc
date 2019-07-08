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

# temporary folder
os.mkdir('./tmp')
folder = './tmp'


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
        options["order"] = [7, 7]
        options["order_max"] = 7
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

        pygpc.plot_2d_grid(coords=grid.coords, fn_plot=os.path.join(folder, test_name + '_grid'))

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
                                     algorithm="standard")

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      fn_out=os.path.join(folder, test_name + '_validation_mc'))

        print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))

        # Validate gPC vs original model function (2D-slice)
        pygpc.validate_gpc_plot(gpc=gpc,
                                coeffs=coeffs,
                                random_vars=["x3", "x1"],
                                n_grid=[25, 25],
                                output_idx=[0],
                                fn_out=os.path.join(folder, test_name + '_validation_2d'),
                                n_cpu=options["n_cpu"])

        # Validate gPC vs original model function (1D-slice)
        pygpc.validate_gpc_plot(gpc=gpc,
                                coeffs=coeffs,
                                random_vars=["x3"],
                                n_grid=[125],
                                output_idx=[0],
                                fn_out=os.path.join(folder, test_name + '_validation_1d'))

        self.expect_true(np.max(nrmsd) < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)))

        print("done!\n")

        # remove temporary directory
        os.rmdir('./tmp')


if __name__ == '__main__':
    unittest.main()
