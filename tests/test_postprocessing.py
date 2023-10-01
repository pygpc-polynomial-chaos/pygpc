import os
import sys
import copy
import time
import h5py
import pygpc
import shutil
import unittest
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib
# matplotlib.use("Qt5Agg")

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
        test_name = 'test_postprocessing_001_random_vars_postprocessing'
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

    def test_postprocessing_002_plot_functions(self):
        """
        Algorithm: Static
        Method: Regression
        Solver: Moore-Penrose
        Grid: Random
        """
        global folder, plot, save_session_format
        test_name = 'test_postprocessing_002_plot_functions'
        print(test_name)

        # %%
        # At first, we are loading the model:
        model = pygpc.testfunctions.Lorenz_System()

        # %%
        # In the next step, we are defining the random variables (ensure that you are using an OrderedDict! Otherwise,
        # the parameter can be mixed up during postprocessing because Python reorders the parameters in standard dictionaries!).
        # Further details on how to define random variables can be found in the tutorial :ref:`How to define a gPC problem`.
        parameters = OrderedDict()
        parameters["sigma"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[10 - 1, 10 + 1])
        parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[28 - 10, 28 + 10])
        parameters["rho"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[(8 / 3) - 1, (8 / 3) + 1])

        # %%
        # To complete the parameter definition, we will also define the deterministic parameters, which are assumed to be
        # constant during the uncertainty and sensitivity analysis:
        parameters["x_0"] = 1.0  # initial value for x
        parameters["y_0"] = 1.0  # initial value for y
        parameters["z_0"] = 1.0  # initial value for z
        parameters["t_end"] = 5.0  # end time of simulation
        parameters["step_size"] = 0.05  # step size for differential equation integration

        # %%
        # With the model and the parameters dictionary, the pygpc problem can be defined:
        problem = pygpc.Problem(model, parameters)

        # %%
        # Now we are ready to define the gPC options, like expansion orders, error types, gPC matrix properties etc.:
        fn_results = "tmp/example_lorenz"
        options = dict()
        options["order_start"] = 6
        options["order_end"] = 20
        options["solver"] = "Moore-Penrose"
        options["interaction_order"] = 2
        options["order_max_norm"] = 0.7
        options["n_cpu"] = 0
        options["error_type"] = 'nrmsd'
        options["error_norm"] = 'absolute'
        options["n_samples_validation"] = 1000
        options["matrix_ratio"] = 5
        options["fn_results"] = fn_results
        options["eps"] = 0.01
        options["grid_options"] = {"seed": 1}

        # %%
        # Now we chose the algorithm to conduct the gPC expansion and initialize the gPC Session:
        algorithm = pygpc.RegAdaptive(problem=problem, options=options)
        session = pygpc.Session(algorithm=algorithm)

        # %%
        # Finally, we are ready to run the gPC. An .hdf5 results file will be created as specified in the options["fn_results"]
        # field from the gPC options dictionary.
        session, coeffs, results = session.run()

        # %%
        # Postprocessing
        # ^^^^^^^^^^^^^^
        # Postprocess gPC and add sensitivity coefficients to results .hdf5 file
        pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                                     output_idx=None,
                                     calc_sobol=True,
                                     calc_global_sens=True,
                                     calc_pdf=False)

        # extract sensitivity coefficients from results .hdf5 file
        sobol, gsens = pygpc.get_sens_summary(fn_gpc=fn_results,
                                              parameters_random=session.parameters_random,
                                              fn_out=fn_results + "_sens_summary.txt")

        # plot time course of mean together with probability density, sobol sensitivity coefficients and global derivatives
        t = np.arange(0.0, parameters["t_end"], parameters["step_size"])
        pygpc.plot_sens_summary(session=session,
                                coeffs=coeffs,
                                sobol=sobol,
                                gsens=gsens,
                                plot_pdf_over_output_idx=True,
                                qois=t,
                                mean=pygpc.SGPC.get_mean(coeffs),
                                std=pygpc.SGPC.get_std(coeffs),
                                x_label="t in s",
                                y_label="x(t)",
                                zlim=[0, 0.4],
                                fn_plot=fn_results + "_sens_summary_test_1.pdf")

        # plot time course of mean together with std, sobol sensitivity coefficients and global derivatives
        pygpc.plot_sens_summary(session=session,
                                coeffs=coeffs,
                                sobol=sobol,
                                gsens=gsens,
                                plot_pdf_over_output_idx=False,
                                qois=t,
                                mean=pygpc.SGPC.get_mean(coeffs),
                                std=pygpc.SGPC.get_std(coeffs),
                                x_label="t in s",
                                y_label="x(t)",
                                fn_plot=fn_results + "_sens_summary_test_2.png")

        # plot sensitivities at one time point with donut plot
        pygpc.plot_sens_summary(sobol=sobol,
                                gsens=gsens,
                                output_idx=50,
                                fn_plot=fn_results + "_sens_summary_test_3.png")

        # plot probability density of output over time (qoi)
        pygpc.plot_gpc(session=session,
                       coeffs=coeffs,
                       output_idx="all",
                       zlim=[0, 0.4],
                       plot_pdf_over_output_idx=True,
                       qois=t,
                       x_label="t in s",
                       y_label="x(t)",
                       fn_plot=fn_results + "_plot_gpc_test_1.png")


if __name__ == '__main__':
    unittest.main()
