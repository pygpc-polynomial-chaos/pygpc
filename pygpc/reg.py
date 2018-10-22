# -*- coding: utf-8 -*-
"""
Class that provides polynomial chaos regression methods
"""

import time
import random
import numpy as np
import scipy
import sys
from builtins import range

from .misc import *
from .gpc import *
from pygpc import iprint, wprint


class Reg(gPC):
    """
    Regression gPC subclass

    Reg(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None)

    Attributes
    ----------
    N_grid: int
        number of grid points
    dim: int
        number of uncertain parameters to process
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str
        string labels of the random variables
    relative_error_loocv: list of float
        relative error of the leave-one-out-cross-validation
    nan_elm: list of float
        which elements were dropped due to NaN

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str, optional, default=None
        string labels of the random variables
    """

    def __init__(self, pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None):
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
        self.nan_elm = []

        # setup polynomial basis functions
        self.init_polynomial_basis()

        # setup polynomial basis functions
        self.init_polynomial_index()

        # construct gpc matrix [Ngrid x Npolybasis]
        self.init_gpc_matrix()

        # get mean values of input random variables
        self.mean_random_vars = self.get_mean_random_vars()

    def get_coeffs_expand(self, sim_results):
        """
        Determine the gPC coefficients by the regression method.

        coeffs = get_coeffs_expand(sim_results)

        Parameters
        ----------
        sim_results: [N_grid x N_out] np.ndarray of float
            results from simulations with N_out output quantities,

        Returns
        -------
        coeffs: [N_coeffs x N_out] np.ndarray of float
            gPC coefficients
        """

        iprint('Determine gPC coefficients...')

        # handle (N,) arrays
        if len(sim_results.shape) == 1:
            self.N_out = 1
        else:
            self.N_out = sim_results.shape[1]

        try:
            return np.dot(self.gpc_matrix_inv, sim_results.T)
        except ValueError:
            wprint("Please check format of parameter sim_results: [N_grid x N_out] np.ndarray.")
            raise

    def get_loocv(self, sim_results):
        """
        Perform leave one out cross validation of gPC with maximal 100 points
        and add result to self.relative_error_loocv.

        relative_error_loocv = get_loocv(sim_results)

        Parameters
        ----------
        sim_results: [N_grid x N_out] np.ndarray
            Results from N_grid simulations with N_out output quantities

        Returns
        -------
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
            display_fancy_bar("LOOCV", int(i + 1), int(N_loocv_points))

        # store result in relative_error_loocv
        self.relative_error_loocv.append(np.mean(relative_error))
        iprint(" (" + str(time.time() - start) + ")")

        return self.relative_error_loocv[-1]
