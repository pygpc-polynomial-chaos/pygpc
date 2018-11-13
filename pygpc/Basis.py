import numpy as np
import scipy.special


class Basis:
    """
    Basis class of gPC

    Attributes
    ----------
    b: list of BasisFunction object instances [n_basis x n_dim]
        Parameter wise basis function objects used in gPC.
        Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.
    """
    def __init__(self):
        """
        Constructor; initializes the Basis class
        """
        self.b = None
        self.b_gpu = None
        self.dim = None
        self.n_basis = None

    # TODO: @Konstantin initialize with parameters: order_(start), interaction_order, order = [x, x, x]
    def init_basis_sgpc(self, order, order_max, interaction_order):
        """
        Initializes basis functions for standard gPC.

        Parameters
        ----------
        order: [dim] list of int
            Maximum individual expansion order
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum expansion order (sum of all exponents)
            The maximum expansion order considers the sum of the orders of combined polynomials only
        interaction_order: int
            Number of random variables, which can interact with each other
            All polynomials are ignored, which have an interaction order greater than specified

        .. math::
           \\begin{tabular}{l*{4}{c}}
            Polynomial Index    & Dimension 1 & Dimension 2 & ... & Dimension M \\\\
           \\hline
            Basis 1             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
            Basis 2             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\vdots              & [Order D1] & [Order D2] & \\vdots  & [Order M] \\\\
            Basis N           & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\end{tabular}

        Notes
        -----
        Adds Attributes:

        b: list of BasisFunction object instances [n_basis x n_dim]
            Parameter wise basis function objects used in gPC.
            Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.

        """
        # OLD: #1 init_polynomial_index
        ###############################
        # # calculate maximum order of polynomials
        # N_max = int(max(self.order))
        #
        # # 2D list of polynomials (lookup)
        # self.poly = [[0 for _ in range(self.dim)] for _ in range(N_max + 1)]
        # # 2D array of polynomial normalization factors (lookup) [N_max+1 x dim]
        # self.poly_norm = np.zeros([N_max + 1, self.dim])
        #
        # # Setup list of polynomials and their coefficients up to the desired order
        # self.init_polynomial_coeffs(0, N_max + 1)

        # OLD: #2 init_polynomial_index
        ###############################

        self.dim = len(order)

        if self.dim == 1:
            self.b = np.linspace(0, order_max, order_max + 1, dtype=int)[:, np.newaxis]
        else:
            self.b = self.get_multi_indices_max_order(self.dim, order_max)

        for i_dim in range(self.dim):
            # add multi-indexes to list when not yet included
            if self.order[i_dim] > self.order_max:
                b_add_dim = np.linspace(self.order_max + 1, self.order[i_dim],
                                        self.order[i_dim] - (self.order_max + 1) + 1)
                b_add_all = np.zeros([b_add_dim.shape[0], self.dim])
                b_add_all[:, i_dim] = b_add_dim
                self.b = np.vstack([self.b, b_add_all.astype(int)])
            # delete multi-indexes from list when they exceed individual max order of parameter
            elif self.order[i_dim] < self.order_max:
                self.b = self.b[self.b[:, i_dim] <= self.order[i_dim], :]

        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.dim:
            self.b = self.b[np.sum(self.b > 0, axis=1) <= self.interaction_order, :]

        # Convert to np.int32 for GPU
        self.b_gpu = self.b.astype(np.int32)

        # get size
        self.n_basis = self.b.shape[0]

        # construct array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [N_poly_basis x 1]
        self.b_norm_basis = np.ones([self.b.shape[0], 1])
        for i_poly in range(self.b.shape[0]):
            for i_dim in range(self.dim):
                self.b_norm_basis[i_poly] *= self.poly_norm[self.b[i_poly, i_dim], i_dim]


    # TODO: implement this into "init_basis" -> MAYBE NOT NEEDED HERE -> SHOULD BE IN BASIS
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

    # TODO: @Konstantin (> gpc > extend_polynomial_basis)
    def extend_basis(self, basis_added):
        """
        Extend set of basis functions and update gpc matrix (append columns).

        Parameters
        ----------
        basis_added: list of list of BasisFunction instances [N_basis_added x D]

        OLD:
        ==================
        Extend polynomial basis functions and add new columns to gpc matrix.

        extend_polynomial_basis(poly_idx_added)

        Parameters
        ----------
        poly_idx_added: [N_poly_added x dim] np.ndarray
            array of added polynomials (order)
        """
        # determine if polynomials in poly_idx_added are already present in self.b if so, delete them
        poly_idx_tmp = []
        for new_row in poly_idx_added:
            not_in_poly_idx = True
            for row in self.b:
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
        # self.b = unique_rows(self.b)
        self.b = np.vstack([self.b, poly_idx_added])
        self.n_basis = self.b.shape[0]

        # extend array of scaling factors to normalize basis functions <psi^2> = int(psi^2*p)dx
        # [N_poly_basis x 1]
        N_poly_new = poly_idx_added.shape[0]
        b_norm_basis_new = np.ones([N_poly_new, 1])
        for i_poly in range(N_poly_new):
            for i_dim in range(self.dim):
                b_norm_basis_new[i_poly] *= self.poly_norm[poly_idx_added[i_poly, i_dim], i_dim]

        self.b_norm_basis = np.vstack([self.b_norm_basis, b_norm_basis_new])

        # append new columns to gpc matrix [self.grid.coords.shape[0] x N_poly_new]
        gpc_matrix_new_columns = np.zeros([self.grid.coords.shape[0], N_poly_new])
        for i_poly_new in range(N_poly_new):
            for i_dim in range(self.dim):
                gpc_matrix_new_columns[:, i_poly_new] *= self.poly[poly_idx_added[i_poly_new][i_dim]][i_dim] \
                    (self.grid.coords_norm[:, i_dim])

        # append new column to gpc matrix
        self.gpc_matrix = np.hstack([self.gpc_matrix, gpc_matrix_new_columns])

        # invert gpc matrix gpc_matrix_inv [N_basis x self.grid.coords.shape[0]]
        self.gpc_matrix_inv = np.linalg.pinv(self.gpc_matrix)

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



