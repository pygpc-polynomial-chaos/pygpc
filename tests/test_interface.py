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

    def test_interface_001_Matlab_gpc(self):
        """
        Algorithm: RegAdaptive
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, matlab, save_session_format
        test_name = 'test_interface_001_Matlab_gpc'
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
            options["gradient_calculation"] = "FD_fwd"
            options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
            options["fn_results"] = os.path.join(folder, test_name)
            options["save_session_format"] = save_session_format
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
                                          fn_out=options["fn_results"] + "_pdf",
                                          plot=plot)

            print("> Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)*100))
            # self.expect_true(np.max(nrmsd) < 0.1, 'gPC test failed with NRMSD error = {:1.2f}%'.format(np.max(nrmsd)*100))

            print("> Checking file consistency...")
            files_consistent, error_msg = pygpc.check_file_consistency(options["fn_results"] + ".hdf5")
            self.expect_true(files_consistent, error_msg)

            print("done!\n")

        else:
            print("Skipping Matlab test...")

    def test_interface_002_save_and_load_session(self):
        """
        Save and load a gPC Session
        """

        global folder, plot, save_session_format
        test_name = 'test_interface_002_save_and_load_session'
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
        options["matrix_ratio"] = 20
        options["error_type"] = "nrmsd"
        options["n_samples_validation"] = 1e3
        options["n_cpu"] = 0
        options["fn_results"] = os.path.join(folder, test_name)
        options["save_session_format"] = ".hdf5"
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_1st2nd"
        options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
        options["backend"] = "omp"
        options["grid"] = pygpc.Random
        options["grid_options"] = None

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


if __name__ == '__main__':
    unittest.main()
