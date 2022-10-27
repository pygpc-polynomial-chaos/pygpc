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

    def test_009_testfunctions(self):
        """
        Testing testfunctions (multi-threading and inherited parallelization)
        """
        test_name = 'pygpc_test_008_testfunctions'
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
                                    options={"seed": 1})

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

    def test_010_RandomParameters(self):
        """
        Testing RandomParameters
        """
        global folder, plot
        test_name = 'pygpc_test_009_RandomParameters'
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

    def test_021_backends(self):
        """
        Test the different backends ["python", "cpu", "omp", "cuda"]
        """

        global folder, gpu
        test_name = 'pygpc_test_023_backends'
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
                            options={"seed": 1})

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
        options["fn_results"] = None
        options["gradient_enhanced"] = True
        options["gradient_calculation"] = "FD_fwd"
        options["gradient_calculation_options"] = {"dx": 0.5, "distance_weight": -2}
        options["grid"] = pygpc.Random
        options["grid_options"] = None

        gpc_matrix = dict()
        gpc_matrix_gradient = dict()
        pce_matrix = dict()

        print("Constructing gPC matrices with different backends...")
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
                gpc.init_gpc_matrix(gradient_idx=np.arange(grid.coords.shape[0]))
                stop = time.time()

                print(b, "Time create_gpc_matrix: ", stop-start)

                # perform polynomial chaos expansion
                coeffs = np.ones([len(gpc.basis.b), 2])
                start = time.time()
                pce = gpc.get_approximation(coeffs, gpc.grid.coords_norm)
                stop = time.time()

                print(b, "Time get_approximation: ", stop-start)

                gpc_matrix[b] = gpc.gpc_matrix
                gpc_matrix_gradient[b] = gpc.gpc_matrix_gradient
                pce_matrix[b] = pce

            except NotImplementedError:
                backends.remove(b)
                warnings.warn("Skipping {} (not installed)...".format(b))

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

    def test_023_gradient_estimation_methods(self):
        """
        Test gradient estimation methods
        """

        global folder, plot, save_session_format
        test_name = 'pygpc_test_025_gradient_estimation_methods'
        print(test_name)

        methods_options = dict()
        methods = ["FD_fwd", "FD_1st", "FD_2nd", "FD_1st2nd"]
        methods_options["FD_fwd"] = {"dx": 0.001, "distance_weight": -2}
        methods_options["FD_1st"] = {"dx": 0.1, "distance_weight": -2}
        methods_options["FD_2nd"] = {"dx": 0.1, "distance_weight": -2}
        methods_options["FD_1st2nd"] = {"dx": 0.1, "distance_weight": -2}

        # define model
        model = pygpc.testfunctions.Peaks()

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
        parameters["x2"] = 0.5
        parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
        problem = pygpc.Problem(model, parameters)

        # define grid
        n_grid = 1000
        grid = pygpc.Random(parameters_random=problem.parameters_random,
                            n_grid=n_grid,
                            options={"seed": 1})

        # create grid points for finite difference approximation
        grid.create_gradient_grid(delta=1e-3)

        # evaluate model function
        com = pygpc.Computation(n_cpu=0, matlab_model=False)

        # [n_grid x n_out]
        res = com.run(model=model,
                      problem=problem,
                      coords=grid.coords,
                      coords_norm=grid.coords_norm,
                      i_iter=None,
                      i_subiter=None,
                      fn_results=None,
                      print_func_time=False)

        grad_res = dict()
        gradient_idx = dict()
        for m in methods:
            # [n_grid x n_out x dim]
            grad_res[m], gradient_idx[m] = pygpc.get_gradient(model=model,
                                                              problem=problem,
                                                              grid=grid,
                                                              results=res,
                                                              com=com,
                                                              method=m,
                                                              gradient_results_present=None,
                                                              gradient_idx_skip=None,
                                                              i_iter=None,
                                                              i_subiter=None,
                                                              print_func_time=False,
                                                              dx=methods_options[m]["dx"],
                                                              distance_weight=methods_options[m]["distance_weight"])

            if m != "FD_fwd":
                nrmsd = pygpc.nrmsd(grad_res[m][:, 0, :], grad_res["FD_fwd"][gradient_idx[m], 0, :])
                self.expect_true((nrmsd < 0.05).all(),
                                 msg="gPC test failed during gradient estimation: {} error too large".format(m))

        print("done!\n")


if __name__ == '__main__':
    unittest.main()
