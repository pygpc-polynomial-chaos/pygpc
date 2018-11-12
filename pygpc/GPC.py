# -*- coding: utf-8 -*-
"""
Class that provides general polynomial chaos methods
"""

import ctypes
import scipy
import numpy as np
from .Grid import *
from builtins import range


class GPC:
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

        self.order = None
        self.order_max = None
        self.interaction_order = None
        self.cpu = True
        self.gpu = None
        self.verbose = True
        self.gpc_matrix = None
        self.gpc_matrix_inv = None

        # external
        self.gpc_coeffs = None

        # in basis
        self.poly = None
        self.poly_gpu = None
        self.poly_idx = None
        self.poly_idx_gpu = None
        self.poly_norm = None
        self.poly_norm_basis = None
        self.poly_der = None
        self.N_poly = None

        # in grid
        self.N_grid = None
        self.grid = None

        # misc (DONE)
        self.sobol = None               # not needed
        self.sobol_idx = None           # not needed
        self.sobol_idx_bool = None      # not needed
        self.N_samples = None           # not needed


        # in problem (DONE)
        self.random_vars = None
        self.pdf_shape = None
        self.pdf_type = None
        self.dim = None
        self.N_out = None               # not needed
        self.limits = None
        self.mean_random_vars = None

    def init_polynomial_coeffs(self, order_begin, order_end):
        """
        Calculate polynomial basis functions of a given order range and add it to the polynomial lookup tables.
        The size, including the polynomials that won't be used, is [max_individual_order x dim].

        .. math::
           \\begin{tabular}{l*{4}{c}}
            Polynomial          & Dimension 1 & Dimension 2 & ... & Dimension M \\\\
           \\hline
            Polynomial 1        & [Coefficients] & [Coefficients] & \\vdots & [Coefficients] \\\\
            Polynomial 2        & 0 & [Coefficients] & \\vdots & [Coefficients] \\\\
           \\vdots              & \\vdots & \\vdots & \\vdots & \\vdots \\\\
            Polynomial N        & [Coefficients] & [Coefficients] & 0 & [Coefficients] \\\\
           \\end{tabular}


        init_polynomial_coeffs(poly_idx_added)

        Parameters
        ----------
        order_begin: int
            order of polynomials to begin with
        order_end: int
            order of polynomials to end with
        """

        self.poly_norm = np.zeros([order_end-order_begin, self.dim])

        for i_dim in range(self.dim):

            for i_order in range(order_begin, order_end):

                if self.pdf_type[i_dim] == "beta":
                    p = self.pdf_shape[0][i_dim]  # beta-distr: alpha=p /// jacobi-poly: alpha=q-1  !!!
                    q = self.pdf_shape[1][i_dim]  # beta-distr: beta=q  /// jacobi-poly: beta=p-1   !!!

                    # determine polynomial normalization factor
                    beta_norm = (scipy.special.gamma(q) * scipy.special.gamma(p) / scipy.special.gamma(p + q) * (
                        2.0) ** (p + q - 1)) ** (-1)

                    jacobi_norm = 2 ** (p + q - 1) / (2.0 * i_order + p + q - 1) * scipy.special.gamma(i_order + p) * \
                                  scipy.special.gamma(i_order + q) / (scipy.special.gamma(i_order + p + q - 1) *
                                                                      scipy.special.factorial(i_order))
                    # initialize norm
                    self.poly_norm[i_order, i_dim] = (jacobi_norm * beta_norm)

                    # add entry to polynomial lookup table
                    self.poly[i_order][i_dim] = scipy.special.jacobi(i_order, q - 1, p - 1, monic=0) / np.sqrt(
                        self.poly_norm[i_order, i_dim])

                if self.pdf_type[i_dim] == "normal" or self.pdf_type[i_dim] == "norm":
                    # determine polynomial normalization factor
                    hermite_norm = scipy.special.factorial(i_order)
                    self.poly_norm[i_order, i_dim] = hermite_norm

                    # add entry to polynomial lookup table
                    self.poly[i_order][i_dim] = scipy.special.hermitenorm(i_order, monic=0) / np.sqrt(
                        self.poly_norm[i_order, i_dim])

    def init_polynomial_basis(self):
        """
        Initialize polynomial basis functions for a maximum order expansion.

        init_polynomial_basis()
        """

        # calculate maximum order of polynomials
        N_max = int(max(self.order))

        # 2D list of polynomials (lookup)
        self.poly = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]
        # 2D array of polynomial normalization factors (lookup) [N_max+1 x dim]
        self.poly_norm = np.zeros([N_max + 1, self.dim])

        # Setup list of polynomials and their coefficients up to the desired order
        self.init_polynomial_coeffs(0, N_max + 1)

    # TODO: @Lucas: Diese GPU Funktion müsste für allgemeine Basisfunktionen angepasst werden
    def init_polynomial_basis_gpu(self):
        """
        Initialized polynomial basis coefficients for graphic card. Converts list of lists of self.polynomial_bases
        into np.ndarray that can be processed on a graphic card.

        init_polynomial_basis_gpu()
        """

        # transform list of lists of polynom objects into np.ndarray
        number_of_variables = len(self.poly[0])
        highest_degree = len(self.poly)
        number_of_polynomial_coeffs = number_of_variables * (highest_degree + 1) * (highest_degree + 2) / 2
        self.poly_gpu = np.empty([number_of_polynomial_coeffs])
        for degree in range(highest_degree):
            degree_offset = number_of_variables * degree * (degree + 1) / 2
            single_degree_coeffs = np.empty([degree + 1, number_of_variables])
            for var in range(number_of_variables):
                single_degree_coeffs[:, var] = np.flipud(self.poly[degree][var].c)
            self.poly_gpu[degree_offset:degree_offset + single_degree_coeffs.size] = single_degree_coeffs.flatten(
                order='C')

    def init_polynomial_index(self):
        """
        Initialize polynomial multi indices. Determine 2D multi-index array (order) of basis functions and
        generate multi-index list up to maximum order. The size is [No. of basis functions x dim].

        .. math::
           \\begin{tabular}{l*{4}{c}}
            Polynomial Index    & Dimension 1 & Dimension 2 & ... & Dimension M \\\\
           \\hline
            Basis 1             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
            Basis 2             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\vdots              & [Order D1] & [Order D2] & \\vdots  & [Order M] \\\\
            Basis N           & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\end{tabular}

        init_polynomial_index()
        """

        if self.dim == 1:
            self.poly_idx = np.linspace(0, self.order_max, self.order_max + 1, dtype=int)[:, np.newaxis]
        else:
            self.poly_idx = get_multi_indices(self.dim, self.order_max)

        for i_dim in range(self.dim):
            # add multi-indexes to list when not yet included
            if self.order[i_dim] > self.order_max:
                poly_add_dim = np.linspace(self.order_max + 1, self.order[i_dim],
                                           self.order[i_dim] - (self.order_max + 1) + 1)
                poly_add_all = np.zeros([poly_add_dim.shape[0], self.dim])
                poly_add_all[:, i_dim] = poly_add_dim
                self.poly_idx = np.vstack([self.poly_idx, poly_add_all.astype(int)])
            # delete multi-indexes from list when they exceed individual max order of parameter     
            elif self.order[i_dim] < self.order_max:
                self.poly_idx = self.poly_idx[self.poly_idx[:, i_dim] <= self.order[i_dim], :]

        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.dim:
            self.poly_idx = self.poly_idx[np.sum(self.poly_idx > 0, axis=1) <= self.interaction_order, :]

        # Convert to np.int32 for GPU
        self.poly_idx_gpu = self.poly_idx.astype(np.int32)

        # get size
        self.N_poly = self.poly_idx.shape[0]

        # construct array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [N_poly_basis x 1]
        self.poly_norm_basis = np.ones([self.poly_idx.shape[0], 1])
        for i_poly in range(self.poly_idx.shape[0]):
            for i_dim in range(self.dim):
                self.poly_norm_basis[i_poly] *= self.poly_norm[self.poly_idx[i_poly, i_dim], i_dim]

    def extend_polynomial_basis(self, poly_idx_added):
        """
        Extend polynomial basis functions and add new columns to gpc matrix.

        extend_polynomial_basis(poly_idx_added)

        Parameters
        ----------
        poly_idx_added: [N_poly_added x dim] np.ndarray
            array of added polynomials (order)
        """

        # determine if polynomials in poly_idx_added are already present in self.poly_idx if so, delete them
        poly_idx_tmp = []
        for new_row in poly_idx_added:
            not_in_poly_idx = True
            for row in self.poly_idx:
                if np.allclose(row, new_row):
                    not_in_poly_idx = False
            if not_in_poly_idx:
                poly_idx_tmp.append(new_row)

        # if all polynomials are already present end routine
        if len(poly_idx_tmp) == 0:
            return
        else:
            poly_idx_added = np.vstack(poly_idx_tmp)

        # determine highest order added        
        order_max_added = np.max(np.max(poly_idx_added))

        # get current maximum order 
        order_max_current = len(self.poly) - 1

        # preallocate new rows to polynomial lists
        for _ in range(order_max_added - order_max_current):
            self.poly.append([0 for _ in range(self.dim)])
            self.poly_norm = np.vstack([self.poly_norm, np.zeros(self.dim)])

        # Extend list of polynomials and their coefficients up to the desired order
        self.init_polynomial_coeffs(order_max_current + 1, order_max_added + 1)

        # append new multi-indexes to old poly_idx array
        # self.poly_idx = unique_rows(self.poly_idx)
        self.poly_idx = np.vstack([self.poly_idx, poly_idx_added])
        self.N_poly = self.poly_idx.shape[0]

        # extend array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [N_poly_basis x 1]
        N_poly_new = poly_idx_added.shape[0]
        poly_norm_basis_new = np.ones([N_poly_new, 1])
        for i_poly in range(N_poly_new):
            for i_dim in range(self.dim):
                poly_norm_basis_new[i_poly] *= self.poly_norm[poly_idx_added[i_poly, i_dim], i_dim]

        self.poly_norm_basis = np.vstack([self.poly_norm_basis, poly_norm_basis_new])

        # append new columns to gpc matrix [self.grid.coords.shape[0] x N_poly_new]
        gpc_matrix_new_columns = np.zeros([self.grid.coords.shape[0], N_poly_new])
        for i_poly_new in range(N_poly_new):
            for i_dim in range(self.dim):
                gpc_matrix_new_columns[:, i_poly_new] *= self.poly[poly_idx_added[i_poly_new][i_dim]][i_dim]\
                (self.grid.coords_norm[:, i_dim])

        # append new column to gpc matrix
        self.gpc_matrix = np.hstack([self.gpc_matrix, gpc_matrix_new_columns])

        # invert gpc matrix gpc_matrix_inv [N_basis x self.grid.coords.shape[0]]
        self.gpc_matrix_inv = np.linalg.pinv(self.gpc_matrix)

    # TODO: @Konstantin: adapt this to new basis definition
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

    # TODO: @Konstantin: adapt this to new basis definition
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

    # TODO: @Konstantin: adapt this to new basis definition and do not invert the matrix here
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
