# -*- coding: utf-8 -*-
"""
Class that provides polynomial chaos regression methods
"""

import time
import random

from .gpc import *
from .misc import *


class Reg(gPC):
    def __init__(self, pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None):
        """
        Regression gPC subclass
        -----------------------
        Reg(self, pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars)
        
        Parameters:
        -----------------------
            random_vars: list of str [dim]
                string labels of the random variables
            pdf_type: list of str [dim]
                type of pdf 'beta' or 'norm'
            pdf_shape: list of list of float
                shape parameters of pdfs
                beta-dist:   [[alpha_1, ...], [beta_1, ...]    ]
                normal-dist: [[mean_1, ...],  [std_1, ...]]
            limits: list of list of float
                upper and lower bounds of random variables
                beta-dist:   [[min_1, ...], [max_1, ...]]
                normal-dist: [[0, ... ], [0, ... ]] (not used)
            order: list of int [dim]
                maximum individual expansion order
                generates individual polynomials also if maximum expansion order in order_max is exceeded
            order_max: int
                maximum expansion order (sum of all exponents)
                the maximum expansion order considers the sum of the orders of combined polynomials only
            interaction_order: int
                number of random variables, which can interact with each other
                all polynomials are ignored, which have an interaction order greater than the specified
            grid: object
                grid object generated in .grid.py including grid.coords and grid.coords_norm
        """
        gPC.__init__(self)
        self.random_vars = random_vars
        self.pdf_type = pdf_type
        self.pdf_shape = pdf_shape
        self.limits = limits
        self.order = order
        self.order_max = order_max
        self.interaction_order = interaction_order
        self.dim = len(pdf_type)
        self.grid = grid
        self.N_grid = grid.coords.shape[0]
        self.relative_error_loocv = []
        self.nan_elm = []  # which elements were dropped due to NAN

        # setup polynomial basis functions
        self.init_polynomial_basis()

        # construct gpc matrix [Ngrid x Npolybasis]
        self.init_gpc_matrix()

        # get mean values of input random variables
        self.mean_random_vars = self.get_mean_random_vars()

    def get_coeffs_expand(self, sim_results):
        """
        Determine the gPC coefficients by the regression method.

        coeffs = get_coeffs_expand(self, sim_results)

        Parameters:
        ----------------------------------
        sim_results: np.array of float [N_grid x N_out]
            results from simulations with N_out output quantities,

        Returns:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gPC coefficients
        """

        # coeffs    ... [N_coeffs x N_points]
        # gpc_matrix_inv      ... [N_coeffs x N_grid]
        # sim_results      ... [N_grid   x N_points]

        vprint('Determine gPC coefficients ...', verbose=self.verbose)
        self.N_out = sim_results.shape[1]

        if sim_results.shape[0] != self.gpc_matrix_inv.shape[1] and \
           sim_results.shape[1] != self.gpc_matrix_inv.shape[1]:
            print("Please check format of input sim_results: matrix [N_grid x N_out] !")
        else:
            sim_results = sim_results.T

        return np.dot(self.gpc_matrix_inv, sim_results)

    def get_loocv(self, sim_results):
        """
        Perform leave one out cross validation of gPC with maximal 100 points
        and add result to self.relative_error_loocv.

        Parameters:
        ----------------------------------
        sim_results: np.array() [N_grid x N_out]
            Results from N_grid simulations with N_out output quantities

        Returns:
        ----------------------------------
        relative_error_loocv: float
            relative mean error of leave one out cross validation
        """

        # define number of performed cross validations (max 100)
        N_loocv_points = np.min((sim_results.shape[0], 100))

        # make list of indices, which are randomly sampled
        loocv_point_idx = random.sample(list(range(sim_results.shape[0])), N_loocv_points)

        start = time.time()
        relative_error = np.zeros(N_loocv_points)
        for i in range(N_loocv_points):
            # get mask of eliminated row
            mask = np.arange(sim_results.shape[0]) != loocv_point_idx[i]

            # invert reduced gpc matrix
            gpc_matrix_inv_loo = np.linalg.pinv(self.gpc_matrix[mask, :])

            # determine gpc coefficients (this takes a lot of time for large problems)
            coeffs_loo = np.dot(gpc_matrix_inv_loo, sim_results[mask, :])
            sim_results_temp = sim_results[loocv_point_idx[i], :]
            relative_error[i] = scipy.linalg.norm(sim_results_temp -np.dot(self.A[loocv_point_idx[i], :], coeffs_loo))\
                                / scipy.linalg.norm(sim_results_temp)
            fancy_bar("LOOCV", int(i + 1), int(N_loocv_points))

        # store result in relative_error_loocv
        self.relative_error_loocv.append(np.mean(relative_error))
        vprint(" (" + str(time.time() - start) + ")", verbose=self.verbose)

        return self.relative_error_loocv[-1]
