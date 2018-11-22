# -*- coding: utf-8 -*-
"""
Unittest class of pygpc
@author: Konstantin Weise
"""
import unittest
import pygpc
from collections import OrderedDict
import numpy as np
import scipy.stats
from scipy.interpolate import griddata
import sys
import matplotlib.pyplot as plt

# first test fixture (class)
class TestpygpcMethods(unittest.TestCase):

    # setup method called before every testcase
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

    # def test_1_static_gpc_reg_MP(self):
    #     test_name = 'Test #1 (Method: Regression, Solver: Moore-Penrose, Grid: RandomGrid)'
    #     print(test_name)
    #
    #     # define model
    #     model = pygpc.testfunctions.PeaksSingle
    #
    #     # define problem
    #     parameters = OrderedDict()
    #     parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    #     parameters["x2"] = pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
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
    #     options["fn_results"] = '/NOBACKUP2/tmp/test'
    #
    #     # generate grid
    #     n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
    #                                            order_glob_max=options["order_max"],
    #                                            order_inter_max=options["interaction_order"],
    #                                            dim=problem.dim)
    #     grid = pygpc.RandomGrid(problem=problem,
    #                             parameters={"n_grid": 1.5 * n_coeffs, "seed": None})
    #
    #     # define algorithm
    #     algorithm = pygpc.Static(problem=problem, options=options, grid=grid)
    #
    #     # run gPC algorithm
    #     gpc, coeffs, results = algorithm.run()
    #
    #     # Apply Monte Carlo method on original model function and gPC approximation for comparison
    #     com = pygpc.Computation(n_cpu=options["n_cpu"])
    #     grid_mc = pygpc.RandomGrid(problem=problem, parameters={"n_grid": 1E4, "seed": None})
    #     y_gpc = gpc.get_approximation(coeffs, grid_mc.coords_norm, output_idx=None)
    #     y_orig = com.run(gpc, grid_mc.coords)
    #     nrmsd = pygpc.get_normalized_rms_deviation(y_gpc, y_orig)[0]
    #
    #     print("\t > NRMSD (gpc vs original): {:.2}%".format(nrmsd))
    #     # Calculating output PDFs
    #     kde_gpc = scipy.stats.gaussian_kde(y_gpc, bw_method=0.15 / y_gpc.std(ddof=1))
    #     pdf_x_gpc = np.linspace(y_gpc.min(), y_gpc.max(), 100)
    #     pdf_y_gpc = kde_gpc(pdf_x_gpc)
    #     kde_orig = scipy.stats.gaussian_kde(y_orig.transpose(), bw_method=0.15 / y_orig.std(ddof=1))
    #     pdf_x_orig = np.linspace(y_orig.min(), y_orig.max(), 100)
    #     pdf_y_orig = kde_orig(pdf_x_orig)
    #
    #     # plot pdfs
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     plt.plot(pdf_x_gpc, pdf_y_gpc, pdf_x_orig, pdf_y_orig)
    #     plt.legend(['gpc', 'original'])
    #     plt.grid()
    #     plt.title(test_name, fontsize=10)
    #     plt.xlabel('y', fontsize=12)
    #     plt.ylabel('p(y)', fontsize=12)
    #     ax.text(0.05, 0.95, r'$error=%.2f$' % (nrmsd, ) + "%",
    #             transform=ax.transAxes, fontsize=12, verticalalignment='top',
    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    #
    #     self.expect_true(nrmsd < 1.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(nrmsd))
    #
    #     print("done!\n")

    def test_2_static_gpc_reg_omp(self):
        test_name = 'Test #2 (Method: Regression, Solver: OMP, Grid: RandomGrid)'
        print(test_name)

        # define model
        model = pygpc.testfunctions.PeaksSingle

        # define problem
        parameters = OrderedDict()
        parameters["x1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
        parameters["x2"] = pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
        problem = pygpc.Problem(model, parameters)

        # gPC options
        options = dict()
        options["method"] = "reg"
        options["solver"] = "OMP"
        options["order"] = [7, 7]
        options["order_max"] = 7
        options["interaction_order"] = 2
        options["n_cpu"] = 8
        options["fn_results"] = '/NOBACKUP2/tmp/test'

        # generate grid
        n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                               order_glob_max=options["order_max"],
                                               order_inter_max=options["interaction_order"],
                                               dim=problem.dim)

        # we assume sparsity of 50%
        sparsity = 0.5
        options["settings"] = {"n_coeffs_sparse": np.ceil(sparsity*n_coeffs)}

        grid = pygpc.RandomGrid(problem=problem,
                                parameters={"n_grid": options["settings"]["n_coeffs_sparse"]*np.log10(n_coeffs),
                                            "seed": None})

        # define algorithm
        algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

        # run gPC algorithm
        gpc, coeffs, results = algorithm.run()

        # Apply Monte Carlo method on original model function and gPC approximation for comparison
        com = pygpc.Computation(n_cpu=options["n_cpu"])
        grid_mc = pygpc.RandomGrid(problem=problem, parameters={"n_grid": 1E4, "seed": None})
        y_gpc = gpc.get_approximation(coeffs, grid_mc.coords_norm, output_idx=None)
        y_orig = com.run(gpc, grid_mc.coords)
        nrmsd = pygpc.get_normalized_rms_deviation(y_gpc.flatten(), y_orig)[0]

        print("\t > NRMSD (gpc vs original): {:.2}%".format(nrmsd))

        # Calculating output PDFs
        kde_gpc = scipy.stats.gaussian_kde(y_gpc.flatten(), bw_method=0.15 / y_gpc.std(ddof=1))
        pdf_x_gpc = np.linspace(y_gpc.min(), y_gpc.max(), 100)
        pdf_y_gpc = kde_gpc(pdf_x_gpc)
        kde_orig = scipy.stats.gaussian_kde(y_orig.flatten(), bw_method=0.15 / y_orig.std(ddof=1))
        pdf_x_orig = np.linspace(y_orig.min(), y_orig.max(), 100)
        pdf_y_orig = kde_orig(pdf_x_orig)

        # plot pdfs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(pdf_x_gpc, pdf_y_gpc, pdf_x_orig, pdf_y_orig)
        plt.legend(['gpc', 'original'])
        plt.grid()
        plt.title(test_name, fontsize=10)
        plt.xlabel('y', fontsize=12)
        plt.ylabel('p(y)', fontsize=12)
        ax.text(0.05, 0.95, r'$error=%.2f$' % (nrmsd, ) + "%",
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.expect_true(nrmsd < 5.0, 'gPC test failed with NRMSD error = {:1.2f}%'.format(nrmsd))

        print("done!\n")



    # def test_1_regular_gpc(self):
    #
    #     print("1. Testing regular gPC")
    #
    #     # set simulation parameters
    #     random_vars = ['x1', 'x2']      # label of random variables
    #     DIM = 2                         # number of random variables
    #     testfun = "peaks"               # test function
    #     pdftype = ["beta", "beta"]      # Type of input PDFs in each dimension
    #     p = [1, 3]                      # first shape parameter of beta distribution (also often: alpha)
    #     q = [1, 6]                      # second shape parameter of beta distribution (also often: beta)
    #     pdfshape = [p, q]
    #     a = [0.5, -0.5]                 # lower bounds of random variables
    #     b = [1.5, 1]                    # upper bounds of random variables
    #     limits = [a, b]
    #     order = [10, 10]                # expansion order in each dimension
    #     order_max = 10                  # maximum order in all dimensions
    #     interaction_order = 2           # interaction order between variables
    #
    #     # random grid parameters:
    #     N_rand = int(1.5 * pygpc.calc_Nc(DIM, order[0]))  # number of grid points
    #
    #     # Sparse grid parameters:
    #     gridtype_sparse = ["jacobi", "jacobi"]  # type of quadrature rule in each dimension
    #     level = [4, 4]                          # level of sparse grid in each dimension
    #     level_max = max(level) + DIM - 1        # maximum level in all dimensions
    #     order_sequence_type = 'exp'             # exponential: order = 2**level + 1
    #
    #     # Tensored grid parameters:
    #     N_tens = [order[0], order[1]]           # number of grid points in each dimension
    #     gridtype_tens = ["jacobi", "jacobi"]    # type of quadrature rule in each dimension
    #
    #     # Monte Carlo simulations (bruteforce for comparison)
    #     N_mc = int(1E5)                         # number of random samples
    #
    #     # generate grids for computations
    #     grid_rand = pygpc.randomgrid(pdftype, pdfshape, limits, N_rand)
    #     grid_mc = pygpc.randomgrid(pdftype, pdfshape, limits, N_mc)
    #
    #     grid_SG = pygpc.grid.sparsegrid(pdftype, gridtype_sparse, pdfshape, limits, level, level_max, interaction_order,
    #                                     order_sequence_type)
    #     grid_tens = pygpc.grid.tensgrid(pdftype, gridtype_tens, pdfshape, limits, N_tens)
    #
    #     # % generate gpc objects
    #     gpc_reg = pygpc.reg(pdftype, pdfshape, limits, order, order_max, interaction_order, grid_rand, random_vars)
    #     gpc_tens = pygpc.quad(pdftype, pdfshape, limits, order, order_max, interaction_order, grid_tens, random_vars)
    #     gpc_SG = pygpc.quad(pdftype, pdfshape, limits, order, order_max, interaction_order, grid_SG, random_vars)
    #
    #     # % evaluate model function on different grids
    #     data_rand = pygpc.tf.peaks(grid_rand.coords)
    #     data_mc = pygpc.tf.peaks(grid_mc.coords)
    #     data_tens = pygpc.tf.peaks(grid_tens.coords)
    #     data_SG = pygpc.tf.peaks(grid_SG.coords)
    #     data_mc = pygpc.tf.peaks(grid_mc.coords)
    #
    #     # % determine gpc coefficients
    #     coeffs_reg = gpc_reg.expand(data_rand)
    #     coeffs_tens = gpc_tens.expand(data_tens)
    #     coeffs_SG = gpc_SG.expand(data_SG)
    #
    #     # perform postprocessing
    #     print("Calculating mean ...")
    #     out_mean_reg = gpc_reg.mean(coeffs_reg)
    #     out_mean_tens = gpc_tens.mean(coeffs_tens)
    #     out_mean_SG = gpc_SG.mean(coeffs_SG)
    #     out_mean_mc = np.mean(data_mc)
    #
    #     print("Calculating standard deviation ...")
    #     out_std_reg = gpc_reg.std(coeffs_reg)
    #     out_std_tens = gpc_tens.std(coeffs_tens)
    #     out_std_SG = gpc_SG.std(coeffs_SG)
    #     out_std_mc = np.std(data_mc)
    #
    #     print("Calculating sobol coefficients ...")
    #     out_sobol_reg, out_sobol_idx_reg = gpc_reg.sobol(coeffs_reg)
    #     out_sobol_tens, out_sobol_idx_tens = gpc_tens.sobol(coeffs_tens)
    #     out_sobol_SG, out_sobol_idx_SG = gpc_SG.sobol(coeffs_SG)
    #
    #     print("Calculating global sensitivity indices ...")
    #     out_globalsens_reg = gpc_reg.globalsens(coeffs_reg)
    #     out_globalsens_tens = gpc_tens.globalsens(coeffs_tens)
    #     out_globalsens_SG = gpc_SG.globalsens(coeffs_SG)
    #
    #     print("Calculating output PDFs ...")
    #     pdf_x_reg, pdf_y_reg = gpc_reg.pdf(coeffs_reg, N_mc)
    #     pdf_x_tens, pdf_y_tens = gpc_tens.pdf(coeffs_tens, N_mc)
    #     pdf_x_SG, pdf_y_SG = gpc_SG.pdf(coeffs_SG, N_mc)
    #
    #     kde_mc = scipy.stats.gaussian_kde(data_mc.transpose(), bw_method=0.2 / data_mc.std(ddof=1))
    #     pdf_x_mc = np.linspace(data_mc.min(), data_mc.max(), 100)
    #     pdf_y_mc = kde_mc(pdf_x_mc)
    #     pdf_x_mc = pdf_x_mc[np.newaxis].T
    #     pdf_y_mc = pdf_y_mc[np.newaxis].T
    #
    #
    #     print("Comparing gpc results to bruteforce Monte Carlo simulations ...")
    #
    #     # compare results to predefined error value (and interpolate if necessary)
    #     data_reg = gpc_reg.evaluate(coeffs_reg,grid_mc.coords_norm)
    #     data_tens = gpc_tens.evaluate(coeffs_tens, grid_mc.coords_norm)
    #     data_SG = gpc_SG.evaluate(coeffs_SG, grid_mc.coords_norm)
    #
    #     eps_reg = pygpc.NRMSD(data_reg, data_mc)
    #     eps_tens = pygpc.NRMSD(data_tens, data_mc)
    #     eps_SG = pygpc.NRMSD(data_SG, data_mc)
    #
    #     eps0 = 1 # error tolerance in %
    #
    #     self.expect_true(eps_reg < eps0, 'gPC regression test failed with error = {:1.2f}%'.format(eps_reg[0]))
    #     self.expect_true(eps_tens < eps0, 'gPC tensored grid test failed with error = {:1.2f}%'.format(eps_tens[0]))
    #     self.expect_true(eps_SG < eps0, 'gPC sparse grid test failed with error = {:1.2f}%'.format(eps_SG[0]))
    #
    #     print("done!\n")

    # def test_2_adaptive_gpc(self):
    #     from pygpc.testfuns.testfunctions import SphereModel
    #     from pygpc.Problem import Problem    #
    #
    #     print("1. Testing adaptive gPC")
    #
    #     save_res_fn = ''    # location to save intermediate results to
    #     eps = 1E-3          # relative error bound
    #     N_points = 201      # Number of grid-points in x- and z-direction
    #
    #     params = OrderedDict()  # we must use an ordered list right form the start, otherwise the order will be mixed
    #     params["R"] = [80, 90, 100]
    #     params["phi_electrode"] = 15
    #     params["N_points"] = 201
    #     params["sigma_1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 5], pdf_limits=[0.15, 0.45])
    #     params["sigma_2"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 3], pdf_limits=[0.01, 0.02])
    #     params["sigma_3"] = pygpc.RandomParameter.Beta(pdf_shape=[2, 2], pdf_limits=[0.4, 0.6])
    #
    #     # anodal and cathodal position at angle phi_electrode in x-z plane
    #     anode_pos = np.array(
    #         [np.cos(params["phi_electrode"] / 180.0 * np.pi) * (params["R"][2]), 0, np.sin(params["phi_electrode"] / 180.0 * np.pi) * (params["R"][2])])
    #     cathode_pos = np.array([-anode_pos[0], anode_pos[1], anode_pos[2]])
    #     anode_pos = anode_pos[:, np.newaxis]
    #     cathode_pos = cathode_pos[:, np.newaxis]
    #
    #     # define points where to evaluate electric potential inside spheres (x-z plane)
    #     points = np.array(
    #         np.meshgrid([np.linspace(-params["R"][2], params["R"][2], N_points)], [np.linspace(-params["R"][2], params["R"][2], N_points)])).T.reshape(-1, 2)
    #     points = points[np.sqrt(np.sum(points ** 2, 1)) <= params["R"][2], :]
    #     points = np.array([points[:, 0], np.zeros(points.shape[0]), points[:, 1]]).T
    #
    #     # enrich params dict by further properties
    #     params["anode_pos"]     = anode_pos
    #     params["cathode_pos"]   = cathode_pos
    #     params["points"]        = points
    #
    #     # create specfic problem as a combination of
    #     # a) the simulation mode, and
    #     # b) the model parameters for this very instance of the model
    #     sphericalTES = Problem(SphereModel, params)
    #
    #     reg, phi = pygpc.run_reg_adaptive2_parallel(problem=sphericalTES,
    #                                                 order_start=0,
    #                                                 order_end=10,
    #                                                 interaction_order_max=2,
    #                                                 eps=eps,
    #                                                 print_out=True,
    #                                                 seed=1,
    #                                                 save_res_fn=save_res_fn,
    #                                                 n_cpu=4)
    #
    #     # perform final gpc expansion including all simulations
    #     coeffs_phi = reg.expand(phi)
    #
    #     # postprocessing
    #     mean = reg.mean(coeffs_phi)
    #     std = reg.std(coeffs_phi)
    #     sobol, sobol_idx = reg.sobol(coeffs_phi)
    #     sobol_1st, sobol_idx_1st = pygpc.extract_sobol_order(sobol, sobol_idx, order=1)
    #     sobol_2nd, sobol_idx_2nd = pygpc.extract_sobol_order(sobol, sobol_idx, order=2)
    #     globalsens = reg.globalsens(coeffs_phi)
    #
    #     # plot mean and standard deviation, define regular grid and interpolate data on it (on same points)
    #     xi = np.linspace(-params["R"][2], params["R"][2], N_points)
    #     zi = xi
    #
    #     fig = plt.figure('size', figsize=[6, 10])
    #
    #     for i in range(2):
    #         fig.add_subplot(2, 1, i + 1)
    #
    #         if i == 0:
    #             pdata = mean
    #             title = 'Mean'
    #         elif i == 1:
    #             pdata = std
    #             title = 'Standard Deviation'
    #
    #         pdata_int = griddata((points[:, 0], points[:, 2]), pdata[0, :], (xi[None, :], zi[:, None]), method='linear')
    #         CS = plt.contour(xi, zi, pdata_int, 30, linewidths=0.5, colors='k')
    #         CS = plt.contourf(xi, zi, pdata_int, 30, cmap=plt.cm.jet)
    #         plt.colorbar()
    #         plt.title(title)
    #
    #         for j in range(3):
    #             plt.plot(np.cos(np.linspace(0, 2 * np.pi, 360)) * params["R"][j],
    #                      np.sin(np.linspace(0, 2 * np.pi, 360)) * params["R"][j],
    #                      'k')
    #
    #     print("done!\n")


if __name__ == '__main__':
    unittest.main()
