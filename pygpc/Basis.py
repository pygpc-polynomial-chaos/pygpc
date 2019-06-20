from .BasisFunction import *
from .misc import get_multi_indices_max_order
import uuid
import numpy as np


class Basis:
    """
    Basis class of gPC

    Attributes
    ----------
    b: list of BasisFunction object instances [n_basis x n_dim]
        Parameter wise basis function objects used in gPC.
        Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.
    b_gpu: ???
        ???
    b_id: list of UUID objects (version 4) [n_basis]
        Unique IDs of global basis functions
    b_norm: ndarray [n_basis x dim]
        Normalization factor of individual basis functions
    b_norm_basis: ndarray [n_basis x 1]
        Normalization factor of global basis functions
    dim:
        Number of variables
    n_basis: int
        Total number of (global) basis function
    """
    def __init__(self):
        """
        Constructor; initializes the Basis class
        """
        self.b = None
        self.b_gpu = np.array(())
        self.b_gpu_grad = []
        self.b_id = None
        self.b_norm = None
        self.b_norm_basis = None
        self.dim = None
        self.n_basis = 0

    def init_basis_sgpc(self, problem, order, order_max, order_max_norm, interaction_order):
        """
        Initializes basis functions for standard gPC.

        Parameters
        ----------
        problem : Problem object
            GPC Problem to analyze
        order : list of int [dim]
            Maximum individual expansion order
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max : int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        order_max_norm : float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more. This truncation scheme
            is also referred to "hyperbolic polynomial chaos expansion" such that sum(a_i^q)^1/q <= p,
            where p is order_max and q is order_max_norm (for more details see eq. (27) in [1]).
        interaction_order : int
            Number of random variables, which can interact with each other
            All polynomials are ignored, which have an interaction order greater than specified

        Notes
        -----
        .. [1] Blatman, G., & Sudret, B. (2011). Adaptive sparse polynomial chaos expansion based on least angle 
           regression. Journal of Computational Physics, 230(6), 2345-2367.

        .. math::
           \\begin{tabular}{l*{4}{c}}
            Polynomial Index    & Dimension 1 & Dimension 2 & ... & Dimension M \\\\
           \\hline
            Basis 1             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
            Basis 2             & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\vdots              & [Order D1] & [Order D2] & \\vdots  & [Order M] \\\\
            Basis N           & [Order D1] & [Order D2] & \\vdots & [Order M] \\\\
           \\end{tabular}

        Adds Attributes:

        b: list of BasisFunction object instances [n_basis x n_dim]
            Parameter wise basis function objects used in gPC.
            Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.

        """

        self.dim = problem.dim

        if self.dim == 1:
            multi_indices = np.linspace(0, order_max, order_max + 1, dtype=int)[:, np.newaxis]
        else:
            multi_indices = get_multi_indices_max_order(self.dim, order_max, order_max_norm)

        for i_dim in range(self.dim):
            # add multi-indexes to list when not yet included
            if order[i_dim] > order_max:
                multi_indices_add_dim = np.linspace(order_max + 1,
                                                    order[i_dim],
                                                    order[i_dim] - (order_max + 1) + 1)
                multi_indices_add_all = np.zeros([multi_indices_add_dim.shape[0], self.dim])
                multi_indices_add_all[:, i_dim] = multi_indices_add_dim
                multi_indices = np.vstack([multi_indices, multi_indices_add_all.astype(int)])

            # delete multi-indexes from list when they exceed individual max order of parameter
            elif order[i_dim] < order_max:
                multi_indices = multi_indices[multi_indices[:, i_dim] <= order[i_dim], :]

        # Consider interaction order (filter out multi-indices exceeding it)
        if interaction_order < self.dim:
            multi_indices = multi_indices[np.sum(multi_indices > 0, axis=1) <= interaction_order, :]

        # get total number of basis functions
        self.n_basis = multi_indices.shape[0]

        # construct 2D list with BasisFunction objects
        self.b = [[0 for _ in range(self.dim)] for _ in range(self.n_basis)]

        for i_basis in range(self.n_basis):
            for i_dim, p in enumerate(problem.parameters_random):   # OrderedDict of RandomParameter objects
                self.b[i_basis][i_dim] = problem.parameters_random[p].init_basis_function(
                    order=multi_indices[i_basis, i_dim])

        # Generate unique IDs of basis functions
        self.b_id = [uuid.uuid4() for _ in range(self.n_basis)]

        # initialize normalization factor (self.b_norm and self.b_norm_basis)
        self.init_b_norm()

        # initialize gpu coefficient array
        self.init_polynomial_basis_gpu(problem)

    def init_b_norm(self):
        """
        Construct array of scaling factors self.b_norm [n_basis x dim] and self.b_norm_basis [n_basis x 1]
        to normalize basis functions <psi^2> = int(psi^2*p)dx
        """
        # read individual normalization factors from function objects
        self.b_norm = np.array([list(map(lambda x:x.fun_norm, _b)) for _b in self.b])

        # determine global normalization factor of basis function
        self.b_norm_basis = np.prod(self.b_norm, axis=1)

    def set_basis_poly(self, order, order_max, order_max_norm, interaction_order, problem):
        """
        Sets up polynomial basis self.b for given order, order_max_norm and interaction order.

        Parameters
        ----------
        order : list of int
            Maximum individual expansion order
            Generates individual polynomials also if maximum expansion order in order_max is exceeded
        order_max: int
            Maximum global expansion order.
            The maximum expansion order considers the sum of the orders of combined polynomials together with the
            chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
            monomial orders.
        order_max_norm: float
            Norm for which the maximum global expansion order is defined [0, 1]. Values < 1 decrease the total number
            of polynomials in the expansion such that interaction terms are penalized more.
            sum(a_i^q)^1/q <= p, where p is order_max and q is order_max_norm (for more details see eq (11) in [1]).
        interaction_order :
            Number of random variables, which can interact with each other
            All polynomials are ignored, which have an interaction order greater than specified
        problem :
            GPC Problem to analyze
        """
        dim = len(order)

        # determine new possible set of basis functions for next main iteration
        multi_indices_all_new = get_multi_indices_max_order(dim=dim,
                                                            order_max=order_max,
                                                            order_max_norm=order_max_norm)

        # delete multi-indices, which are already present
        if self.b is not None:
            multi_indices_all_current = np.array([list(map(lambda x: x.p["i"], _b)) for _b in b])

            idx_old = np.hstack([np.where((multi_indices_all_current[i, :] == multi_indices_all_new).all(axis=1))
                                 for i in range(multi_indices_all_current.shape[0])])

            multi_indices_all_new = np.delete(multi_indices_all_new, idx_old, axis=0)

        # filter out polynomials of interaction_order
        interaction_order_list = np.sum(multi_indices_all_new > 0, axis=1)
        multi_indices_added = multi_indices_all_new[interaction_order_list <= interaction_order, :]

        if multi_indices_added.any():

            # construct 2D list with new BasisFunction objects
            b_added = [[0 for _ in range(dim)] for _ in range(multi_indices_added.shape[0])]

            for i_basis in range(multi_indices_added.shape[0]):
                for i_p, p in enumerate(problem.parameters_random):
                    b_added[i_basis][i_p] = problem.parameters_random[p].init_basis_function(
                        order=multi_indices_added[i_basis, i_p])

            # extend basis
            self.extend_basis(b_added)

    def extend_basis(self, b_added):
        """
        Extend set of basis functions. Skips basis functions, which are already present in self.b.

        Parameters
        ----------
        b_added: 2D list of BasisFunction instances [n_b_added x dim]
            Individual BasisFunctions to add
        """

        # add b_added to b (check for duplicates) and generate IDs
        for i_row, _b in enumerate(b_added):
            if _b not in self.b:
                self.b.append(_b)
                self.b_id.append(uuid.uuid4())

        # update size
        self.n_basis = len(self.b)

        # update normalization factors
        self.init_b_norm()

    # TODO: @Lucas (GPU) adapt this to function objects
    def init_polynomial_basis_gpu(self, problem):
        """
        Initialize polynomial basis coefficients for GPU. Converts list of lists of self.b
        into np.ndarray that can be processed on a GPU.
        """
        for i_basis in range(len(self.b)):
            for i_dim in range(problem.dim):
                polynomial_order = self.b[i_basis][i_dim].fun.order
                self.b_gpu = np.append(self.b_gpu, polynomial_order)
                self.b_gpu = np.append(self.b_gpu, np.flip(self.b[i_basis][i_dim].fun.c))

        for i_dim_gradient in range(problem.dim):
            _b_gpu_grad = np.array(())
            for i_basis in range(len(self.b)):
                for i_dim in range(problem.dim):
                    if i_dim == i_dim_gradient:
                        polynomial = self.b[i_basis][i_dim].fun.deriv()
                        polynomial_order = polynomial.order
                        _b_gpu_grad = np.append(_b_gpu_grad, polynomial_order)
                        _b_gpu_grad = np.append(_b_gpu_grad, np.flip(polynomial.c))
                    else:
                        polynomial_order = self.b[i_basis][i_dim].fun.order
                        _b_gpu_grad = np.append(_b_gpu_grad, polynomial_order)
                        _b_gpu_grad = np.append(_b_gpu_grad, np.flip(self.b[i_basis][i_dim].fun.c))

            self.b_gpu_grad.append(_b_gpu_grad)

# # TODO: implement this into "init_basis" -> MAYBE NOT NEEDED HERE -> SHOULD BE IN BASIS
   #  def init_polynomial_coeffs(self, order_begin, order_end):
   #      """
   #      Calculate polynomial basis functions of a given order range and add it to the polynomial lookup tables.
   #      The size, including the polynomials that won't be used, is [max_individual_order x dim].
   #
   #      .. math::
   #         \\begin{tabular}{l*{4}{c}}
   #          Polynomial          & Dimension 1 & Dimension 2 & ... & Dimension M \\\\
   #         \\hline
   #          Polynomial 1        & [Coefficients] & [Coefficients] & \\vdots & [Coefficients] \\\\
   #          Polynomial 2        & 0 & [Coefficients] & \\vdots & [Coefficients] \\\\
   #         \\vdots              & \\vdots & \\vdots & \\vdots & \\vdots \\\\
   #          Polynomial N        & [Coefficients] & [Coefficients] & 0 & [Coefficients] \\\\
   #         \\end{tabular}
   #
   #
   #      init_polynomial_coeffs(poly_idx_added)
   #
   #      Parameters
   #      ----------
   #      order_begin: int
   #          order of polynomials to begin with
   #      order_end: int
   #          order of polynomials to end with
   #      """
   #
   #      self.poly_norm = np.zeros([order_end-order_begin, self.dim])
   #
   #      for i_dim in range(self.dim):
   #
   #          for i_order in range(order_begin, order_end):
   #
   #              if self.pdf_type[i_dim] == "beta":
   #                  p = self.pdf_shape[0][i_dim]  # beta-distr: alpha=p /// jacobi-poly: alpha=q-1  !!!
   #                  q = self.pdf_shape[1][i_dim]  # beta-distr: beta=q  /// jacobi-poly: beta=p-1   !!!
   #
   #                  # determine polynomial normalization factor
   #                  beta_norm = (scipy.special.gamma(q) * scipy.special.gamma(p) / scipy.special.gamma(p + q) * (
   #                      2.0) ** (p + q - 1)) ** (-1)
   #
   #                  jacobi_norm = 2 ** (p + q - 1) / (2.0 * i_order + p + q - 1) * scipy.special.gamma(i_order + p) * \
   #                                scipy.special.gamma(i_order + q) / (scipy.special.gamma(i_order + p + q - 1) *
   #                                                                    scipy.special.factorial(i_order))
   #                  # initialize norm
   #                  self.poly_norm[i_order, i_dim] = (jacobi_norm * beta_norm)
   #
   #                  # add entry to polynomial lookup table
   #                  self.poly[i_order][i_dim] = scipy.special.jacobi(i_order, q - 1, p - 1, monic=0) / np.sqrt(
   #                      self.poly_norm[i_order, i_dim])
   #
   #              if self.pdf_type[i_dim] == "normal" or self.pdf_type[i_dim] == "norm":
   #                  # determine polynomial normalization factor
   #                  hermite_norm = scipy.special.factorial(i_order)
   #                  self.poly_norm[i_order, i_dim] = hermite_norm
   #
   #                  # add entry to polynomial lookup table
   #                  self.poly[i_order][i_dim] = scipy.special.hermitenorm(i_order, monic=0) / np.sqrt(
   #                      self.poly_norm[i_order, i_dim])

