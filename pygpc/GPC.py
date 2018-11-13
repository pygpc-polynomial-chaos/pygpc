# -*- coding: utf-8 -*-
"""
Class that provides general polynomial chaos methods
"""
import os
import ctypes
import numpy as np

from .Grid import *
from .misc import get_all_combinations
from builtins import range


class GPC(object):
    """
    General gPC base class

    Attributes
    ----------
    N_grid: int
        number of grid points
    N_poly: int
        number of polynomials psi
    N_samples: int
        number of samples xi

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

    sobol: [N_sobol x N_out] np.ndarray
        Sobol indices of N_out output quantities
    sobol_idx: [N_sobol] list of np.ndarray
        List of parameter label indices belonging to Sobol indices
    cpu: bool
        flag to execute the calculation on the cpu
    gpu: bool
        flag to execute the calculation on the gpu
    verbose: bool
        boolean value to determine if to print out the progress into the standard output
    gpc_matrix: [N_samples x N_poly] np.ndarray
        generalized polynomial chaos matrix
    gpc_matrix_inv: [N_poly x N_samples] np.ndarray
        pseudo inverse of the generalized polynomial chaos matrix
    gpc_coeffs: [N_poly x N_out] np.ndarray
        coefficient matrix of independent regions of interest for every coefficient
    poly: [dim x order_span] list of list of np.poly1d:
        polynomial objects containing the coefficients that are used to build the gpc matrix
    poly_gpu: np.ndarray
        polynomial coefficients stored in a np.ndarray that can be processed on a graphic card
    poly_idx: [N_poly x dim] np.ndarray
        multi indices to determine the degree of the used sub-polynomials
    poly_idx_gpu [N_poly x dim] np.ndarray
        multi indices to determine the degree of the used sub-polynomials stored in a np.ndarray that can be processed
        on a graphic card
    poly_der: [dim x order_span] list of list of np.poly1d:
        derivative of the polynomial objects containing the coefficients that are used to build the gpc matrix
    poly_norm: [order_span x dim] np.ndarray
        normalizing scaling factors of the used sub-polynomials
    poly_norm_basis: [N_poly] np.ndarray
        normalizing scaling factors of the polynomial basis functions
    sobol_idx_bool: list of np.ndarray of bool
        boolean mask that determines which multi indices are unique
    """

    def __init__(self, problem):

        # GPC
        self.problem = problem
        self.cpu = True
        self.gpu = None
        self.verbose = True
        self.gpc_matrix = None
        self.gpc_matrix_inv = None
        self.nan_elm = []


        # SGPC
        self.order = None
        self.order_max = None
        self.interaction_order = None

        # EGPC

        # external
        self.gpc_coeffs = None      # DELETE THIS

        # in basis
        self.poly = None
        self.poly_gpu = None
        self.poly_idx = None
        self.poly_idx_gpu = None
        self.poly_norm = None
        self.poly_norm_basis = None
        self.poly_der = None
        self.N_poly = None

        # in grid (DONE)
        self.n_grid = None
        self.grid = None                # object itself

        # misc (DONE)
        self.sobol = None               # not needed
        self.sobol_idx = None           # not needed
        self.sobol_idx_bool = None      # not needed
        self.N_samples = None           # not needed


        # in Problem (DONE)
        self.random_vars = None
        self.pdf_shape = None
        self.pdf_type = None
        self.dim = None
        self.N_out = None               # not needed
        self.limits = None
        self.mean_random_vars = None

    # TODO: @Konstantin: adapt this to new basis definition and do not invert the matrix here (GPC) with basis as input
    def init_gpc_matrix(self):
        """
        Construct the gPC matrix and the Moore-Penrose-pseudo-inverse.

        init_gpc_matrix()
        """

        iprint('Constructing gPC matrix...')
        gpc_matrix = np.ones([self.grid.coords.shape[0], self.N_poly])

        def cpu(self, gpc_matrix):
            for i_poly in range(self.N_poly):
                for i_dim in range(self.dim):
                    gpc_matrix[:, i_poly] *= \
                        self.poly[self.poly_idx[i_poly][i_dim]][i_dim](self.grid.coords_norm[:, i_dim])
            self.gpc_matrix = gpc_matrix

        # TODO: @Lucas: Bitte an neue Basisfunktionsklassen anpassen
        def gpu(self, gpc_matrix):
            # get parameters
            number_of_variables = len(self.poly[0])
            highest_degree = len(self.poly)

            # handle pointer
            polynomial_coeffs_pointer = self.poly_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            polynomial_index_pointer = self.poly_idx_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            xi_pointer = self.grid.coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            polynomial_matrix_pointer = gpc_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            number_of_psi_size_t = ctypes.c_size_t(self.N_poly)
            number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
            highest_degree_size_t = ctypes.c_size_t(highest_degree)
            number_of_xi_size_t = ctypes.c_size_t(self.grid.coords.shape[0])

            # handle shared object
            dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'gpc.so'), mode=ctypes.RTLD_GLOBAL)
            cuda_pce = dll.polynomial_chaos_matrix
            cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
                                [ctypes.POINTER(ctypes.c_double)] * 2 + [ctypes.c_size_t] * 4

            # evaluate CUDA implementation
            cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, polynomial_matrix_pointer,
                     number_of_psi_size_t, number_of_variables_size_t, highest_degree_size_t, number_of_xi_size_t)

        # choose between gpu or cpu execution
        if self.cpu:
            cpu(self, gpc_matrix)
        else:
            gpu(self, gpc_matrix)

        # assign gpc matrix to member variable
        self.gpc_matrix = gpc_matrix

        # invert gpc matrix gpc_matrix_inv [N_basis x self.grid.coords.shape[0]]
        self.gpc_matrix_inv = np.linalg.pinv(gpc_matrix)

    # TODO: @Konstantin: adapt this to new basis definition (GPC)
    def extend_gpc_matrix_samples(self, samples_poly_ratio, seed=None):
        """
        Add sample points according to input pdfs to grid and extend the gpc matrix such that the ratio of
        rows/columns equals samples_poly_ratio.

        extend_gpc_matrix_samples(samples_poly_ratio, seed=None):

        Parameters
        ----------
        samples_poly_ratio: float
            ratio between number of samples and number of polynomials the matrix will be extended until
        seed: float, optional, default=None
            random seeding point
        """

        # Number of new grid points
        N_grid_new = int(np.ceil(samples_poly_ratio * self.gpc_matrix.shape[1] - self.gpc_matrix.shape[0]))

        if N_grid_new > 0:
            # Generate new grid points
            new_grid_points = RandomGrid(self.pdf_type, self.pdf_shape, self.limits, N_grid_new, seed=seed)

            # append points to existing grid
            self.grid.coords = np.vstack([self.grid.coords, new_grid_points.coords])
            self.grid.coords_norm = np.vstack([self.grid.coords_norm, new_grid_points.coords_norm])

            # determine new row of gpc matrix
            gpc_matrix_new_rows = np.ones([N_grid_new, self.N_poly])
            for i_poly in range(self.N_poly):
                for i_dim in range(self.dim):
                    gpc_matrix_new_rows[:, i_poly] *= self.poly[self.poly_idx[i_poly][i_dim]][i_dim]\
                    (new_grid_points.coords_norm[:, i_dim])

            # append new row to gpc matrix    
            self.gpc_matrix = np.vstack([self.gpc_matrix, gpc_matrix_new_rows])

            # invert gpc matrix gpc_matrix_inv [N_basis x self.grid.coords.shape[0]]
            self.gpc_matrix_inv = np.linalg.pinv(self.gpc_matrix)

    # TODO: @Konstantin: adapt this to new basis definition (GPC)
    def replace_gpc_matrix_samples(self, idx, seed=None):
        """
        Replace distinct sample points from the gpc matrix.

        replace_gpc_matrix_samples(idx, seed=None)

        Parameters
        ----------
        idx: np.ndarray
            array of grid indices of obj.grid.coords[idx,:] which are going to be replaced
            (rows of gPC matrix will be replaced by new ones)
        seed: float, optional, default=None
            random seeding point
        """

        # Generate new grid points
        new_grid_points = RandomGrid(self.pdf_type, self.pdf_shape, self.limits, idx.size, seed=seed)

        # append points to existing grid
        self.grid.coords[idx, :] = new_grid_points.coords
        self.grid.coords_norm[idx, :] = new_grid_points.coords_norm

        # determine new row of gpc matrix
        gpc_matrix_new_rows = np.ones([idx.size, self.N_poly])
        for i_poly in range(self.N_poly):
            for i_dim in range(self.dim):
                gpc_matrix_new_rows[:, i_poly] *= self.poly[self.poly_idx[i_poly][i_dim]][i_dim] \
                    (new_grid_points.coords_norm[:, i_dim])

        # append new row to gpc matrix
        self.gpc_matrix[idx, :] = gpc_matrix_new_rows

        # invert gpc matrix gpc_matrix_inv [N_basis x self.grid.coords.shape[0]]
        self.gpc_matrix_inv = np.linalg.pinv(self.gpc_matrix)

    def get_multi_indices_max_order(self, dim, max_order):
        """
        Computes all multi-indices with a maximum overall order of max_order.

        multi_indices = get_multi_indices_max_order(length, max_order)

        Parameters
        ----------
        dim : int
            Number of random parameters (length of multi-index tuples)
        max_order : int
            Maximum order (over all parameters)

        Returns
        -------
        multi_indices: np.ndarray [n_basis x dim]
            Multi-indices for a classical maximum order gpc
        """

        multi_indices = []
        for i_max_order in range(max_order + 1):
            # Chose (length-1) the splitting points of the array [0:(length+max_order)]
            # 1:length+max_order-1
            s = get_all_combinations(np.arange(dim + i_max_order - 1) + 1, dim - 1)

            m = s.shape[0]

            s1 = np.zeros([m, 1])
            s2 = (dim + i_max_order) + s1

            v = np.diff(np.hstack([s1, s, s2]))
            v = v - 1

            if i_max_order == 0:
                multi_indices = v
            else:
                multi_indices = np.vstack([multi_indices, v])

        return multi_indices.astype(int)
