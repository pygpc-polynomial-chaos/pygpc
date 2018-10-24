# -*- coding: utf-8 -*-
"""
Unittest class of for the pygpc framework
"""

import unittest
import pygpc
import numpy as np
import scipy.stats
from scipy.interpolate import griddata
import sys
import matplotlib.pyplot as plt


# first test fixture (class)
class TestPygpcMethods(unittest.TestCase):

    # setup method called before every test case
    def setUp(self):
        pass

    def run(self, result=None):
        self._result = result
        self._num_expectations = 0
        super(TestPygpcMethods, self).run(result)

    def failed(self, failure):
        try:
            raise failure
        except failure.__class__:
            self._result.addFailure(self, sys.exc_info())

    def expect_equal(self, a, b, msg=''):
        if a != b:
            msg = '({}) Expected {} to equal {}. '.format(self._num_expectations, a, b) + msg
            self.failed(self.failureException(msg))
        self._num_expectations += 1

    def expect_true(self, a, msg):
        if not a:
            self.failed(self.failureException(msg))
        self._num_expectations += 1

    def test_1_regular_gpc(self):

        print("1. Testing regular gPC")

        # set simulation parameters
        random_vars = ['x1', 'x2']      # label of random variables
        DIM = 2                         # number of random variables
        testfun = "peaks"               # test function
        pdf_type = ["beta", "beta"]      # Type of input PDFs in each dimension
        p = [1, 3]                      # first shape parameter of beta distribution (also often: alpha)
        q = [1, 6]                      # second shape parameter of beta distribution (also often: beta)
        pdf_shape = [p, q]
        a = [0.5, -0.5]                 # lower bounds of random variables
        b = [1.5, 1]                    # upper bounds of random variables
        limits = [a, b]
        order = [10, 10]                # expansion order in each dimension
        order_max = 10                  # maximum order in all dimensions
        interaction_order = 2           # interaction order between variables

        # random grid parameters:
        N_rand = int(1.5 * pygpc.misc.get_num_coeffs(DIM, order[0]))  # number of grid points

        # Sparse grid parameters:
        gridtype_sparse = ["jacobi", "jacobi"]  # type of quadrature rule in each dimension
        level = [4, 4]                          # level of sparse grid in each dimension
        level_max = max(level) + DIM - 1        # maximum level in all dimensions
        order_sequence_type = 'exp'             # exponential: order = 2**level + 1

        # Tensored grid parameters:
        N_tens = [order[0], order[1]]           # number of grid points in each dimension
        gridtype_tens = ["jacobi", "jacobi"]    # type of quadrature rule in each dimension

        # Monte Carlo simulations (bruteforce for comparison)
        N_monte_carlo = int(1E5)                         # number of random samples

        # generate grids for computations
        grid_rand = pygpc.grid.RandomGrid(pdf_type, pdf_shape, limits, N_rand)
        
        grid_mc = pygpc.grid.RandomGrid(pdf_type, pdf_shape, limits, N_monte_carlo)

        grid_sg = pygpc.grid.SparseGrid(pdf_type, gridtype_sparse, pdf_shape, limits, level, level_max,
                                        interaction_order, order_sequence_type)
        grid_tens = pygpc.grid.TensorGrid(pdf_type, gridtype_tens, pdf_shape, limits, N_tens)

        # % generate gpc objects
        gpc_reg = pygpc.reg.Reg(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid_rand, random_vars)
        gpc_tens = pygpc.quad.Quad(pdf_type, pdf_shape, limits, order, order_max, interaction_order,
                              grid_tens, random_vars)
        gpc_sg = pygpc.quad.Quad(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid_sg, random_vars)

        # % evaluate model function on different grids
        data_rand = pygpc.testfun.peaks(grid_rand.coords)
        data_mc = pygpc.testfun.peaks(grid_mc.coords)
        data_tens = pygpc.testfun.peaks(grid_tens.coords)
        data_sg = pygpc.testfun.peaks(grid_sg.coords)
        data_mc = pygpc.testfun.peaks(grid_mc.coords)

        # % determine gpc coefficients
        coeffs_reg = gpc_reg.get_coeffs_expand(data_rand)
        coeffs_tens = gpc_tens.get_coeffs_expand(data_tens)
        coeffs_sg = gpc_sg.get_coeffs_expand(data_sg)

        # perform postprocessing
        print("Calculating mean...")
        out_mean_reg = gpc_reg.get_mean_value(coeffs_reg)
        out_mean_tens = gpc_tens.get_mean_value(coeffs_tens)
        out_mean_sg = gpc_sg.get_mean_value(coeffs_sg)
        out_mean_mc = np.mean(data_mc)

        print("Calculating standard deviation ...")
        out_std_reg = gpc_reg.get_standard_deviation(coeffs_reg)
        out_std_tens = gpc_tens.get_standard_deviation(coeffs_tens)
        out_std_SG = gpc_sg.get_standard_deviation(coeffs_sg)
        out_std_mc = np.std(data_mc)

        print("Calculating sobol coefficients ...")
        out_sobol_reg, out_sobol_idx_reg, out_sobol_idx_bool_reg = gpc_reg.get_sobol_indices(coeffs=coeffs_reg)
        out_sobol_tens, out_sobol_idx_tens, out_sobol_idx_bool_tens = gpc_tens.get_sobol_indices(coeffs=coeffs_tens)
        out_sobol_sg, out_sobol_idx_sg,out_sobol_idx_bool_sg = gpc_sg.get_sobol_indices(coeffs=coeffs_sg)
        #
        print("Calculating global sensitivity indices ...")
        out_globalsens_reg = gpc_reg.get_global_sens(coeffs_reg)
        out_globalsens_tens = gpc_tens.get_global_sens(coeffs_tens)
        out_globalsens_SG = gpc_sg.get_global_sens(coeffs_sg)

        print("Calculating output PDFs ...")
        pdf_x_reg, pdf_y_reg = gpc_reg.get_pdf(coeffs_reg, N_monte_carlo)
        pdf_x_tens, pdf_y_tens = gpc_tens.get_pdf(coeffs_tens, N_monte_carlo)
        pdf_x_SG, pdf_y_SG = gpc_sg.get_pdf(coeffs_sg, N_monte_carlo)

        # print("Calculating standard deviation ...")
        # out_std_reg = gpc_reg.std(coeffs_reg)
        # out_std_tens = gpc_tens.std(coeffs_tens)
        # out_std_SG = gpc_sg.std(coeffs_SG)
        # out_std_mc = np.std(data_mc)
        #
        # print("Calculating sobol coefficients ...")
        # out_sobol_reg, out_sobol_idx_reg = gpc_reg.sobol(coeffs_reg)
        # out_sobol_tens, out_sobol_idx_tens = gpc_tens.sobol(coeffs_tens)
        # out_sobol_SG, out_sobol_idx_SG = gpc_sg.sobol(coeffs_SG)
        #
        # print("Calculating global sensitivity indices ...")
        # out_globalsens_reg = gpc_reg.globalsens(coeffs_reg)
        # out_globalsens_tens = gpc_tens.globalsens(coeffs_tens)
        # out_globalsens_SG = gpc_sg.globalsens(coeffs_SG)
        #
        # print("Calculating output PDFs ...")
        # pdf_x_reg, pdf_y_reg = gpc_reg.pdf(coeffs_reg, N_monte_carlo)
        # pdf_x_tens, pdf_y_tens = gpc_tens.pdf(coeffs_tens, N_monte_carlo)
        # pdf_x_sg, pdf_y_sg = gpc_sg.pdf(coeffs_SG, N_monte_carlo)
        #
        # kde_mc = scipy.stats.gaussian_kde(data_mc.transpose(), bw_method=0.2 / data_mc.std(ddof=1))
        # pdf_x_mc = np.linspace(data_mc.min(), data_mc.max(), 100)
        # pdf_y_mc = kde_mc(pdf_x_mc)
        # pdf_x_mc = pdf_x_mc[np.newaxis].T
        # pdf_y_mc = pdf_y_mc[np.newaxis].T
        #
        #
        # print("Comparing gpc results to bruteforce Monte Carlo simulations ...")
        #
        # # compare results to predefined error value (and interpolate if necessary)
        # data_reg = gpc_reg.evaluate(coeffs_reg,grid_mc.coords_norm)
        # data_tens = gpc_tens.evaluate(coeffs_tens, grid_mc.coords_norm)
        # data_sg = gpc_sg.evaluate(coeffs_sg, grid_mc.coords_norm)
        #
        # eps_reg = pygpc.NRMSD(data_reg, data_mc)
        # eps_tens = pygpc.NRMSD(data_tens, data_mc)
        # eps_sg = pygpc.NRMSD(data_sg, data_mc)
        #
        #
        # # error tolerance in %
        # eps0 = 1
        #
        # self.expect_true(eps_reg < eps0, 'gPC regression test failed with error = {:1.2f}%'.format(eps_reg[0]))
        # self.expect_true(eps_tens < eps0, 'gPC tensored grid test failed with error = {:1.2f}%'.format(eps_tens[0]))
        # self.expect_true(eps_SG < eps0, 'gPC sparse grid test failed with error = {:1.2f}%'.format(eps_SG[0]))
        #
        # print("done!\n")

    # def test_2_adaptive_gpc(self):
    #     print("1. Testing adaptive gPC")
    #     # Model parameters
    #     save_res_fn = ''
    #     R = [80, 90, 100]  # Radii of spheres in mm
    #     phi_electrode = 15  # Polar angle of electrode location in deg
    #     N_points = 201  # Number of grid-points in x- and z-direction
    #
    #     # Statistical parameters
    #     random_vars = ['sigma_1', 'sigma_2', 'sigma_3']
    #     pdf_type = ["beta", "beta", "beta"]
    #     DIM = 3  # number of random variables
    #     a = [0.15, 0.01, 0.4]  # lower bounds of conductivities in S/m
    #     b = [0.45, 0.02, 0.6]  # upper bounds of conductivities in S/m
    #     p = [5, 1, 2]  # first shape parameter of pdf
    #     q = [5, 3, 2]  # second shape parameter of pdf
    #     max_order = 0  # maximum order at initialization
    #
    #     eps = 1E-3  # relative error bound
    #     pdf_shape = [p, q]
    #     limits = [a, b]
    #
    #     # anodal and cathodal position at angle phi_electrode in x-z plane
    #     anode_pos = np.array(
    #         [np.cos(phi_electrode / 180.0 * np.pi) * (R[2]), 0, np.sin(phi_electrode / 180.0 * np.pi) * (R[2])])
    #     cathode_pos = np.array([-anode_pos[0], anode_pos[1], anode_pos[2]])
    #     anode_pos = anode_pos[:, np.newaxis]
    #     cathode_pos = cathode_pos[:, np.newaxis]
    #
    #     # define points where to evaluate electric potential inside spheres (x-z plane)
    #     points = np.array(
    #         np.meshgrid([np.linspace(-R[2], R[2], N_points)], [np.linspace(-R[2], R[2], N_points)])).T.reshape(-1, 2)
    #     points = points[np.sqrt(np.sum(points ** 2, 1)) <= R[2], :]
    #     points = np.array([points[:, 0], np.zeros(points.shape[0]), points[:, 1]]).T
    #
    #     ########################################################################################
    #     # run adaptive gpc (regression) passing the goal function func(x, args())
    #     reg, phi = pygpc.run_reg_adaptive2(random_vars=random_vars,
    #                                        pdf_type=pdf_type,
    #                                        pdf_shape=pdf_shape,
    #                                        limits=limits,
    #                                        func=pygpc.tf.potential_3layers_surface_electrodes,
    #                                        args=(R, anode_pos, cathode_pos, points, 50),
    #                                        order_start=0,
    #                                        order_end=10,
    #                                        interaction_order_max=2,
    #                                        eps=eps,
    #                                        print_out=True,
    #                                        seed=None,
    #                                        save_res_fn=save_res_fn)
    #
    #     ########################################################################################
    #
    #     # perform final gpc expansion including all simulations
    #     coeffs_phi = reg.expand(phi)
    #
    #     # postprocessing
    #     mean = reg.mean(coeffs_phi)
    #     std = reg.std(coeffs_phi)
    #     sobol, sobol_idx = reg.sobol(coeffs_phi)
    #     globalsens = reg.globalsens(coeffs_phi)
    #
    #     # plot mean and standard deviation, define regular grid and interpolate data on it (on same points)
    #     xi = np.linspace(-R[2], R[2], N_points)
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
    #             plt.plot(np.cos(np.linspace(0, 2 * np.pi, 360)) * R[j],
    #                      np.sin(np.linspace(0, 2 * np.pi, 360)) * R[j],
    #                      'k')
    #
    #     print("done!\n")


if __name__ == '__main__':
    unittest.main()
