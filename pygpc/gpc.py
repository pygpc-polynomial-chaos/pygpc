# -*- coding: utf-8 -*-
"""
Class that provides general polynomial chaos methods
"""
# TODO: transform into meta class

import ctypes
import scipy
import os
from .misc import *
from .grid import *
from .postproc import *


class gPC:
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
    N_out: int
        number of output coefficients
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
    sobol: [N_sobol x N_out] np.ndarray
        Sobol indices of N_out output quantities
    sobol_idx: [N_sobol] list of np.ndarray
        List of parameter label indices belonging to Sobol indices
    cpu: bool
        flag to execute the calculation on the cpu
    gpu: bool
        flag to execute the calculation on the gpu
    verbose: bool
        flag to print out the progress in the standard output
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
    """
    def __init__(self):
        self.random_vars = None
        self.gpc_coeffs = None
        self.pdf_shape = None
        self.pdf_type = None
        self.poly = None
        self.poly_gpu = None
        self.poly_idx = None
        self.poly_idx_gpu = None
        self.poly_norm = None #
        self.poly_norm_basis = None #
        self.poly_der = None
        self.dim = None
        self.order = None
        self.order_max = None
        self.grid = None
        self.sobol = None
        self.sobol_idx = None
        self.sobol_idx_bool = None #
        self.interaction_order = None
        self.limits = None
        self.N_poly = None
        self.N_out = None
        self.N_grid = None
        self.N_samples = None
        self.mean_random_vars = None
        self.cpu = None
        self.gpu = None
        self.verbose = True
        self.gpc_matrix = None
        self.gpc_matrix_inv = None

    def init_polynomial_coeffs(self, order_begin, order_end):
        """
        Calculate polynomial basis functions of a given order range and add it to the polynomial lookup tables.

        init_polynomial_coeffs(poly_idx_added)

        Parameters:
        ----------------------------------
        oder_begin: int
            order of polynomials to begin with
        order_end: int
            order of polynomials to end with
        """

        #  poly    |     dim_1     dim_2    ...    dim_M
        # -----------------------------------------------
        # Poly_1   |  [coeffs]  [coeffs]   ...  [coeffs]
        # Poly_2   |  [coeffs]  [coeffs]   ...  [coeffs]
        #   ...    |  [coeffs]  [coeffs]   ...   [0]
        #   ...    |  [coeffs]  [coeffs]   ...   [0]
        #   ...    |  [coeffs]  [coeffs]   ...   ...
        # Poly_N   |   [0]      [coeffs]   ...   [0]
        #
        # size: [max_individual_order x dim] (includes polynomials also not used)

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
        """Initialize polynomial basis functions for a maximum order expansion."""

        # calculate maximum order of polynomials
        N_max = int(np.max(self.order))

        # 2D list of polynomials (lookup)
        self.poly = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]
        # 2D array of polynomial normalization factors (lookup) [N_max+1 x dim]
        self.poly_norm = np.zeros([N_max + 1, self.dim])

        # Setup list of polynomials and their coefficients up to the desired order
        self.init_polynomial_coeffs(0, N_max + 1)

    def init_polynomial_basis_gpu(self):
        """
        Initialized polynomial basis coefficients for graphic card. Converts list of lists of self.polynomial_bases
        into np.ndarray that can be processed on a graphic card."""

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
        """Initialize polynomial multi indices."""

        # Determine 2D multi-index array (order) of basis functions w.r.t. 2D array
        # of polynomials self.poly
        #
        # poly_idx |     dim_1       dim_2       ...    dim_M
        # -------------------------------------------------------
        # basis_1  |  [order_D1]  [order_D2]     ...  [order_DM]    
        # basis_2  |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        #  ...     |  [order_D1]  [order_D2]     ...  [order_DM]
        # basis_Nb |  [order_D1]  [order_D2]     ...  [order_DM]
        #
        # size: [No. of basis functions x dim]

        # generate multi-index list up to maximum order
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

        Parameters:
        ----------------------------------
        poly_idx_added: np.array of int [N_poly_added x dim]
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
        for i in range(order_max_added - order_max_current):
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

    def extend_gpc_matrix_samples(self, samples_poly_ratio, seed=None):
        """
        Add sample points according to input pdfs to grid and extend the gpc matrix such that the ratio of
        rows/columns is samples_poly_ratio.

        extend_gpc_matrix_samples(samples_poly_ratio, seed=None):

        Parameters:
        ----------------------------------
        samples_poly_ratio: float
            ratio between number of samples and number of polynomials the matrix will be extended until
        seed (optional): float
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

    def replace_gpc_matrix_samples(self, idx, seed=None):
        """
        Replace distinct sample points from the gpc matrix.

        replace_gpc_matrix_samples(idx, seed=None)

        Parameters:
        ----------------------------------
        idx: np.array of int
            array of grid indices of obj.grid.coords[idx,:] which are going to be replaced
            (rows of gPC matrix will be replaced by new ones)
        seed (optional): float
            Random seeding point
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

    def init_gpc_matrix(self):
        """
        Construct the gpc matrix self.gpc_matrix [self.grid.coords.shape[0] x N_poly] and invert it using the
        Moore Penrose pseudo inverse self.gpc_matrix_inv.
        """

        vprint('Constructing gPC matrix ...', verbose=self.verbose)
        gpc_matrix = np.ones([self.grid.coords.shape[0], self.N_poly])

        def cpu(self, gpc_matrix):
            for i_poly in range(self.N_poly):
                for i_dim in range(self.dim):
                    gpc_matrix[:, i_poly] *= self.poly[self.poly_idx[i_poly][i_dim]][i_dim](self.grid.coords_norm[:, i_dim])
            self.gpc_matrix = gpc_matrix

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

    def get_mean(self, coeffs):
        """
        Calculate the expected value.

        mean = mean(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        mean: np.array of float [1 x N_out]
            mean
        """

        mean = coeffs[0, :]
        mean = mean[np.newaxis, :]
        return mean

    def get_std(self, coeffs):
        """
        Calculate the standard deviation.

        std = std(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        std: np.array of float [1 x N_out]
            standard deviation
        """

        # return np.sqrt(np.sum(np.multiply(np.square(self.coeffs[1:,:]),self.poly_norm_basis[1:,:]),axis=0))
        std = np.sqrt(np.sum(np.square(coeffs[1:, :]), axis=0))
        std = std[np.newaxis, :]
        return std

    def get_pdf_mc(self, N_samples, coeffs=None, output_idx=None):
        """
        Randomly sample the gpc expansion to determine output pdfs in specific points.

        xi, y = MC_sampling(coeffs, N_samples, output_idx=None)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        N_samples: int
            number of random samples drawn from the respective input pdfs
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
        xi: np.array of float [N_samples x dim]
            generated samples in normalized coordinates
        y: np.array of float [N_samples x N_out]
            gpc solutions
        """

        # handle input parameters
        if not coeffs:
            coeffs=self.gpc_coeffs

        self.N_out = coeffs.shape[1]

        # if output index list is not provided, sample all gpc outputs
        if not output_idx:
            output_idx = np.linspace(0, self.N_out - 1, self.N_out)
            output_idx = output_idx[np.newaxis, :]

        # seed the random numbers generator
        np.random.seed()

        # generate random samples for each random input variable [N_samples x dim]
        xi = np.zeros([N_samples, self.dim])
        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == "beta":
                xi[:, i_dim] = (np.random.beta(self.pdf_shape[0][i_dim],
                                               self.pdf_shape[1][i_dim], [N_samples, 1]) * 2.0 - 1)[:, 0]
            if self.pdf_type[i_dim] == "norm" or self.pdf_type[i_dim] == "normal":
                xi[:, i_dim] = (np.random.normal(0, 1, [N_samples, 1]))[:, 0]

        # TODO: pce necessary?
        # pce = self.evaluate(coeffs, xi, output_idx)
        # return xi, pce

        return xi

    def get_pce(self, coeffs=None, xi=None, output_idx=None):
        """
        Calculates the gpc approximation in points with output_idx and normalized parameters xi
        (interval: [-1, 1])

        y = evaluate(self, coeffs, xi, output_idx)

        example: y = evaluate( [[xi_1_p1 ... xi_dim_p1] ,
                                [xi_1_p2 ... xi_dim_p2]],
                                np.array([[0,5,13]])    )

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x dim]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)
        cpu (optional): bool
            Choice if the matrices should be processed on the CPU or GPU (Default: CPU)

        Returns:
        ----------------------------------
        y: np.array of float [N_xi x N_out]
            gpc approximation at normalized coordinates xi
        """

        def cpu():
            y = np.zeros([xi.shape[0], coeffs.shape[1]])
            for i_poly in range(self.N_poly):
                gpc_matrix_new_row = np.ones(xi.shape[0])
                for i_dim in range(self.dim):
                    gpc_matrix_new_row *= self.poly[self.poly_idx[i_poly][i_dim]][i_dim](xi[:, i_dim])
                y += np.outer(gpc_matrix_new_row, coeffs[i_poly, :])
            return y

        def gpu():
            # initialize matrices and parameters
            y = np.zeros([xi.shape[0], coeffs.shape[1]])
            number_of_variables = len(self.poly[0])
            highest_degree = len(self.poly)

            # handle pointer
            polynomial_coeffs_pointer = self.poly_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            polynomial_index_pointer = self.poly_idx_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            xi_pointer = xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sim_result_pointer = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            sim_coeffs_pointer = coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            number_of_xi_size_t = ctypes.c_size_t(xi.shape[0])
            number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
            number_of_psi_size_t = ctypes.c_size_t(coeffs.shape[0])
            highest_degree_size_t = ctypes.c_size_t(highest_degree)
            number_of_result_vectors_size_t = ctypes.c_size_t(coeffs.shape[1])

            # handle shared object
            dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'pce.so'), mode=ctypes.RTLD_GLOBAL)
            cuda_pce = dll.polynomial_chaos_matrix
            cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
                                [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_size_t] * 5

            # evaluate CUDA implementation
            cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, sim_result_pointer,
                     sim_coeffs_pointer, number_of_psi_size_t, number_of_result_vectors_size_t,
                     number_of_variables_size_t,
                     highest_degree_size_t, number_of_xi_size_t)
            return y

        # handle input parameters
        self.N_out = coeffs.shape[1]
        self.N_poly = self.poly_idx.shape[0]
        if not coeffs:
            coeffs = self.gpc_coeffs
        if not xi:
            xi = self.grid.coords_norm
        if len(xi.shape) == 1:
            xi = xi[:, np.newaxis]
        if np.any(output_idx):
            output_idx = np.array(output_idx)
            if len(output_idx.shape):
                output_idx = output_idx[np.newaxis, :]
            coeffs = coeffs[:, output_idx]

        if self.cpu:
            return cpu()
        else:
            return gpu()

    def get_sobol_indices(self, coeffs=None):
        """
        Determine the available sobol indices.

        sobol, sobol_idx = get_sobol_indices(self, coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        sobol: np.array of float [N_sobol x N_out]
            Not normalized sobol_indices
        sobol_idx: list of np.array of int [N_sobol x dim]
            List containing the parameter combinations in rows of sobol
        sobol_idx_bool: list of np.array of bool

        """

        vprint("Determining Sobol indices", verbose=self.verbose)

        # handle input parameters
        if not coeffs:
            coeffs = self.gpc_coeffs
        N_sobol_theoretical = 2 ** self.dim - 1
        N_coeffs = coeffs.shape[0]
        if N_coeffs == 1:
            raise Exception('Number of coefficients is 1 ... no sobol indices to calculate ...')

        # Generate boolean matrix of all basis functions where order > 0 = True
        # size: [N_coeffs x dim] 
        sobol_mask = self.poly_idx != 0

        # look for unique combinations (i.e. available sobol combinations)
        # size: [N_sobol x dim]
        sobol_idx_bool = unique_rows(sobol_mask)

        # delete the first row where all polys are order 0 (no sensitivity)
        sobol_idx_bool = np.delete(sobol_idx_bool, [0], axis=0)
        N_sobol_available = sobol_idx_bool.shape[0]

        # check which basis functions contribute to which sobol coefficient set 
        # True for specific coeffs if it contributes to sobol coefficient
        # size: [N_coeffs x N_sobol]
        sobol_poly_idx = np.zeros([N_coeffs, N_sobol_available])
        for i_sobol in range(N_sobol_available):
            sobol_poly_idx[:, i_sobol] = np.all(sobol_mask == sobol_idx_bool[i_sobol], axis=1)

        # calculate sobol coefficients matrix by summing up the individual
        # contributions to the respective sobol coefficients
        # size [N_sobol x N_points]    
        sobol = np.zeros([N_sobol_available, coeffs.shape[1]])
        for i_sobol in range(N_sobol_available):
            sobol[i_sobol, :] = np.sum(np.square(coeffs[sobol_poly_idx[:, i_sobol] == 1, :]), axis=0)
            # not normalized polynomials:             
            # sobol[i_sobol, :] = np.sum(np.multiply(np.square(coeffs[sobol_poly_idx[:, i_sobol] == 1, :]),
                                                   # self.poly_norm_basis[sobol_poly_idx[:, i_sobol] == 1, :]), axis=0)

        # sort sobol coefficients in descending order (w.r.t. first output only ...)
        idx_sort_descend_1st = np.argsort(sobol[:, 0], axis=0)[::-1]
        sobol = sobol[idx_sort_descend_1st, :]
        sobol_idx_bool = sobol_idx_bool[idx_sort_descend_1st]

        # get list of sobol indices
        sobol_idx = [0 for x in range(sobol_idx_bool.shape[0])]
        for i_sobol in range(sobol_idx_bool.shape[0]):
            sobol_idx[i_sobol] = np.array([i for i, x in enumerate(sobol_idx_bool[i_sobol, :]) if x])

        return sobol, sobol_idx, sobol_idx_bool

    def write_log_sobol(self, path, sobol_rel_order_mean, sobol_rel_1st_order_mean, sobol_extracted_idx_1st):
        # start log
        log = open(os.path.splitext(path)[0] + '.txt', 'w')
        log.write("Sobol indices:\n")
        log.write("==============\n")
        log.write("\n")

        # print order ratios
        log.write("Ratio: order / total variance over all output quantities:\n")
        log.write("---------------------------------------------------------\n")
        for i in range(len(sobol_rel_order_mean)):
            log.write("Order {}: {:.4f}\n".format(i + 1, sobol_rel_order_mean[i]))

        log.write("\n")

        # print 1st order ratios of parameters
        log.write("Ratio: 1st order Sobol indices of parameters / total variance over all output quantities\n")
        log.write("----------------------------------------------------------------------------------------\n")

        random_vars = []
        max_len = max([len(self.random_vars[i]) for i in range(len(self.random_vars))])
        for i in range(len(sobol_rel_1st_order_mean)):
            log.write("{}{:s}: {:.4f}\n".format(
                (max_len - len(self.random_vars[sobol_extracted_idx_1st[i]])) * ' ',
                self.random_vars[sobol_extracted_idx_1st[i]],
                sobol_rel_1st_order_mean[i]))
            random_vars.append(self.random_vars[sobol_extracted_idx_1st[i]])

        log.close()

    def get_sobol_order(self, coeffs=None, sobol=None, sobol_idx=None, sobol_idx_bool=None):
        """
        Evaluate order of determined sobol indices.

        sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std, sobol_rel_1st_order_mean, sobol_rel_1st_order_std
        = get_sobol_order(self, coeffs=None, sobol=None, sobol_idx=None, sobol_idx_bool=None)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        sobol: np.array of float [N_sobol x N_out]
            Not normalized sobol_indices
        sobol_idx: list of np.array of int [N_sobol x dim]
            List containing the parameter combinations in rows of sobol
        sobol_rel_order_mean: nparray of float
            Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,)
            over all output quantities
        sobol_rel_order_std: nparray of float
            Standard deviation of the proportion of the Sobol indices of the different order to the total variance
            (1st, 2nd, etc..,) over all output quantities
        sobol_rel_1st_order_mean: nparray of float
            Average proportion of the random variables of the 1st order Sobol indices to the total variance over all
            output quantities
        sobol_rel_1st_order_std: nparray of float
            Standard deviation of the proportion of the random variables of the 1st order Sobol indices to the total
            variance over all output quantities
        """

        # handle input parameters
        if not coeffs:
            coeffs = self.gpc_coeffs
        if not(sobol and sobol_idx and sobol_idx_bool):
            if sobol or sobol_idx or sobol_idx_bool:
                print('Please put in sobol, sobol_idx and sobol_idx_bool or none of them')
                return
            else:
                sobol, sobol_idx, sobol_idx_bool = self.get_sobol_indices(coeffs)

        # get max order
        order_max = np.max(np.sum(sobol_idx_bool, axis=1))

        # total variance
        var = np.sum(sobol, axis=0).flatten()

        # get NaN values
        not_nan_mask = np.logical_not(np.isnan(var))

        sobol_rel_order_mean = []
        sobol_rel_order_std = []
        sobol_rel_1st_order_mean = []
        sobol_rel_1st_order_std = []
        str_out = []

        # get maximum length of random_vars label
        max_len = max([len(self.random_vars[i]) for i in range(len(self.random_vars))])

        for i in range(order_max):
            # extract sobol coefficients of order i
            sobol_extracted, sobol_extracted_idx = get_extracted_sobol_order(sobol, sobol_idx, i + 1)

            sobol_rel_order_mean.append(np.sum(np.sum(sobol_extracted[:, not_nan_mask], axis=0).flatten())
                                        / np.sum(var[not_nan_mask]))
            sobol_rel_order_std.append(0)

            vprint("\tRatio: Sobol indices order {} / total variance: {:.4f}".format(i + 1, sobol_rel_order_mean[i]),
                   self.verbose)

            # for first order indices, determine ratios of all random variables
            if i == 0:
                # deep copy
                sobol_extracted_idx_1st = sobol_extracted_idx[:]
                for j in range(sobol_extracted.shape[0]):
                    sobol_rel_1st_order_mean.append(np.sum(sobol_extracted[j, not_nan_mask].flatten())
                                                    / np.sum(var[not_nan_mask]))
                    sobol_rel_1st_order_std.append(0)

                    str_out.append("\t{}{}: {:.4f}".format((max_len -
                                                            len(self.random_vars[
                                                                    sobol_extracted_idx_1st[j]])) * ' ',
                                                           self.random_vars[sobol_extracted_idx_1st[j]],
                                                           sobol_rel_1st_order_mean[j]))

        sobol_rel_order_mean = np.array(sobol_rel_order_mean)
        sobol_rel_1st_order_mean = np.array(sobol_rel_1st_order_mean)

        # print output of 1st order Sobol indice ratios of parameters
        if self.verbose:
            for j in range(len(str_out)):
                print(str_out[j])

        return sobol, sobol_idx, sobol_rel_order_mean, sobol_rel_order_std,\
               sobol_rel_1st_order_mean, sobol_rel_1st_order_std

    def get_global_sens(self, coeffs):
        """
        Determine the global derivative based sensitivity coefficients.

        Reference:
        D. Xiu, Fast Numerical Methods for Stochastic Computations: A Review,
        Commun. Comput. Phys., 5 (2009), pp. 242-272 eq. (3.14) page 255

        get_global_sens = calc_globalsens(coeffs)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients

        Returns:
        ----------------------------------
        get_global_sens: np.array of float [dim x N_out]
            global derivative based sensitivity coefficients
        """

        N_max = int(len(self.poly))

        self.poly_der = [[0 for _ in range(self.dim)] for _ in range(N_max)]
        poly_der_int = [[0 for _ in range(self.dim)] for _ in range(N_max)]
        poly_int = [[0 for _ in range(self.dim)] for _ in range(N_max)]
        knots_list_1d = [0 for _ in range(self.dim)]
        weights_list_1d = [0 for _ in range(self.dim)]

        # generate quadrature points for numerical integration for each random
        # variable separately (2*N_max points for high accuracy)

        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == 'beta':  # Jacobi polynomials
                knots_list_1d[i_dim], weights_list_1d[i_dim] = quadrature_jacobi_1d(2 * N_max,
                                                                                    self.pdf_shape[0][i_dim] - 1,
                                                                                    self.pdf_shape[1][i_dim] - 1)
            if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == "normal":  # Hermite polynomials
                knots_list_1d[i_dim], weights_list_1d[i_dim] = quadrature_hermite_1d(2 * N_max)

        # pre-process polynomials
        for i_dim in range(self.dim):
            for i_order in range(N_max):
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_dim] = np.polyder(self.poly[i_order][i_dim])

                # evaluate poly and poly_der at quadrature points and integrate w.r.t.
                # pdf (multiply with weights and sum up)
                # saved like self.poly [N_order x dim]
                poly_int[i_order][i_dim] = np.sum(
                    np.dot(self.poly[i_order][i_dim](knots_list_1d[i_dim]), weights_list_1d[i_dim]))
                poly_der_int[i_order][i_dim] = np.sum(
                    np.dot(self.poly_der[i_order][i_dim](knots_list_1d[i_dim]), weights_list_1d[i_dim]))

        N_poly = self.poly_idx.shape[0]
        poly_der_int_multi = np.zeros([self.dim, N_poly])

        for i_sens in range(self.dim):
            for i_poly in range(N_poly):
                poly_der_int_single = 1

                # evaluate complete integral (joint basis function)                
                for i_dim in range(self.dim):
                    if i_dim == i_sens:
                        poly_der_int_single *= poly_der_int[self.poly_idx[i_poly][i_dim]][i_dim]
                    else:
                        poly_der_int_single *= poly_int[self.poly_idx[i_poly][i_dim]][i_dim]

                poly_der_int_multi[i_sens, i_poly] = poly_der_int_single

        # sum up over all coefficients        
        # [dim x N_points]  = [dim x N_poly] * [N_poly x N_points]
        return np.dot(poly_der_int_multi, coeffs) / (2 ** self.dim)

    def get_local_sens(self, coeffs, xi):
        """
        Determine the local derivative based sensitivity coefficients in the point of operation xi
        in normalized coordinates.

        get_local_sens = calc_localsens(coeffs, xi)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        xi: np.array of float [1 x dim]
            point in variable space to evaluate local sensitivity in (normalized coordinates!)

        Returns:
        ----------------------------------
        get_local_sens: np.array of float [dim x N_out]
            local sensitivity
        """

        N_max = len(self.poly)

        self.poly_der = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]
        poly_der_xi = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]
        poly_opvals = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]

        # pre-process polynomials
        for i_dim in range(self.dim):
            for i_order in range(N_max + 1):
                # evaluate the derivatives of the polynomials
                self.poly_der[i_order][i_dim] = np.polyder(self.poly[i_order][i_dim])

                # evaluate poly and poly_der at point of operation
                poly_opvals[i_order][i_dim] = self.poly[i_order][i_dim](xi[1, i_dim])
                poly_der_xi[i_order][i_dim] = self.poly_der[i_order][i_dim](xi[1, i_dim])

        N_vals = 1
        poly_sens = np.zeros([self.dim, self.N_poly])

        for i_sens in range(self.dim):
            for i_poly in range(self.N_poly):
                poly_sens_single = np.ones(N_vals)

                # construct polynomial basis according to partial derivatives                
                for i_dim in range(self.dim):
                    if i_dim == i_sens:
                        poly_sens_single *= poly_der_xi[self.poly_idx[i_poly][i_dim]][i_dim]
                    else:
                        poly_sens_single *= poly_opvals[self.poly_idx[i_poly][i_dim]][i_dim]
                poly_sens[i_sens, i_poly] = poly_sens_single

        # sum up over all coefficients        
        # [dim x N_points]  = [dim x N_poly]  *   [N_poly x N_points]
        return np.dot(poly_sens, coeffs)

    def get_pdf(self, coeffs, N_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        get_pdf = pdf(coeffs, N_samples, output_idx)

        Parameters:
        ----------------------------------
        coeffs: np.array of float [N_coeffs x N_out]
            gpc coefficients
        N_samples: int
            Number of samples used to estimate output pdf
        output_idx (optional): np.array of int [1 x N_out]
            idx of output quantities to consider (Default: all outputs)

        Returns:
        ----------------------------------
            pdf_x: nparray [100 x N_out]
                x-coordinates of output pdf (output quantity),
            pdf_y: nparray [100 x N_out]
                y-coordinates of output pdf (probability density of output quantity)
        """

        self.N_out = coeffs.shape[1]

        # if output index array is not provided, determine pdfs of all outputs 
        if not output_idx:
            output_idx = np.linspace(0, self.N_out - 1, self.N_out)
            output_idx = output_idx[np.newaxis, :]

        # sample gPC expansion
        samples_in, samples_out = self.get_pdf_mc(coeffs, N_samples, output_idx)

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, self.N_out])
        pdf_y = np.zeros([100, self.N_out])

        for i_out in range(coeffs.shape[1]):
            kde = scipy.stats.gaussian_kde(samples_out.transpose(), bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)
            pdf_y[:, i_out] = kde(pdf_x[:, i_out])

        return pdf_x, pdf_y

    def get_mean_random_vars(self):
        """
        Determine the average values of the input random variables from their pdfs.

        Returns:
        --------
        mean_random_vars: nparray of float [N_random_vars]
            Average values of the input random variables
        """
        mean_random_vars = np.zeros(self.dim)

        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == 'normal':
                mean_random_vars[i_dim] = self.pdf_shape[0][i_dim]

            if self.pdf_type[i_dim] == 'beta':
                mean_random_vars[i_dim] = (float(self.pdf_shape[0][i_dim]) /
                                           (self.pdf_shape[0][i_dim] + self.pdf_shape[1][i_dim])) * \
                                          (self.limits[1][i_dim] - self.limits[0][i_dim]) + \
                                          (self.limits[0][i_dim])

        return mean_random_vars
