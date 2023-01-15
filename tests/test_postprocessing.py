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

    def test_postprocessing_001_random_vars_postprocessing(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'pygpc_test_021_random_vars_postprocessing_sobol'
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
        options["save_session_format"] = save_session_format
        options["gradient_enhanced"] = True
        options["backend"] = "omp"
        options["grid_options"] = {"seed": seed}

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)

        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=options["matrix_ratio"] * n_coeffs,
                            options=options["grid_options"])

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

        # Validate gPC vs original model function (Monte Carlo)
        nrmsd = pygpc.validate_gpc_mc(session=session,
                                      coeffs=coeffs,
                                      n_samples=int(1e4),
                                      output_idx=0,
                                      fn_out=options["fn_results"],
                                      folder="validate_gpc_mc",
                                      plot=plot,
                                      n_cpu=session.n_cpu)

        print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
        # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

        print("> Checking file consistency...")
        files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
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
                                    fn_out=options["fn_results"],
                                    folder="validate_gpc_plot",
                                    n_cpu=options["n_cpu"])

        print("done!\n")


if __name__ == '__main__':
    unittest.main()
