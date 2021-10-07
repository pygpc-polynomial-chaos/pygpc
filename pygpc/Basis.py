import os
import uuid
import time
import numpy as np
import multiprocessing
import multiprocessing.pool
import matplotlib.pyplot as plt
from _functools import partial
from .misc import get_multi_indices
# from mpl_toolkits.mplot3d import Axes3D
from .BasisFunction import *


class Basis:
    """
    Basis class of gPC

    Attributes
    ----------
    b : list of list of BasisFunction object instances [n_basis][n_dim]
        Parameter wise basis function objects used in gPC.
        Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.
    b_array : ndarray of float [n_poly_coeffs]
        Polynomial coefficients of basis functions
    b_id : list of UUID objects (version 4) [n_basis]
        Unique IDs of global basis functions
    b_norm : ndarray of float [n_basis x dim]
        Normalization factor of individual basis functions
    b_norm_basis : ndarray of float [n_basis x 1]
        Normalization factor of global basis functions
    dim : int
        Number of variables
    n_basis : int
        Total number of (global) basis function
    multi_indices: ndarray [n_basis x dim]
        Multi-indices of polynomial basis functions
    """
    def __init__(self):
        """
        Constructor; initializes the Basis class
        """
        self.b = None
        self.b_array = None
        self.b_array_grad = None
        self.b_id = None
        self.b_norm = None
        self.b_norm_basis = None
        self.dim = None
        self.n_basis = 0
        self.multi_indices = None

    def set_basis(self, i_basis, problem):
        """
        Worker function to initialize a global basis function (called by multiprocessing.pool).
        It also initializes polynomial basis coefficients for fast processing. Converts list of lists of basis
        into np.ndarray that can be processed on multi core systems.

        Parameters
        ----------
        i_basis : int
            Index of global basis function
        problem : Problem class instance
            gPC problem

        Returns
        -------
        b_ : list [n_dim]
            List containing the individual basis functions of the parameters
        b_a_ : ndarray of int
            Concatenated list of polynomial basis coefficients
        b_a_grad_ : ndarray of int
            Concatenated list of polynomial basis coefficients for gradient evaluation
        """

        b_ = [0 for _ in range(problem.dim)]
        b_a_ = []
        b_a_grad_ = []

        for i_dim, p in enumerate(problem.parameters_random):   # OrderedDict of RandomParameter objects
            b_[i_dim] = problem.parameters_random[p].init_basis_function(order=self.multi_indices[i_basis, i_dim])

        for i_dim in range(problem.dim):
            for i_dim_inner in range(problem.dim):
                if i_dim == 0:
                    b_a_ = b_a_ + [np.array([b_[i_dim_inner].fun.order]),
                                   b_[i_dim_inner].fun.c]
                if i_dim == i_dim_inner:
                    b_a_grad_ = b_a_grad_ + [np.array([b_[i_dim_inner].fun.deriv().order]),
                                             b_[i_dim_inner].fun.deriv().c]
                else:
                    b_a_grad_ = b_a_grad_ + [np.array([b_[i_dim_inner].fun.order]),
                                             b_[i_dim_inner].fun.c]

        b_a_ = np.concatenate(b_a_)
        b_a_grad_ = np.concatenate(b_a_grad_)

        return b_, b_a_, b_a_grad_

    def init_basis_sgpc(self, problem, order, order_max, order_max_norm, interaction_order,
                        interaction_order_current=None):
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
        interaction_order_current : int, optional, default: interaction_order
            Number of random variables currently interacting with respect to the highest order.
            (interaction_order_current <= interaction_order)
            The parameters for lower orders are all interacting with "interaction order".

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
        assert self.dim == len(order), "gPC order does not fit to number of random variables"

        if self.dim == 1:
            self.multi_indices = np.linspace(0, order_max, order_max + 1, dtype=int)[:, np.newaxis]
        else:
            self.multi_indices = get_multi_indices(order=order,
                                                   order_max=order_max,
                                                   order_max_norm=order_max_norm,
                                                   interaction_order=interaction_order,
                                                   interaction_order_current=interaction_order_current)

        # get total number of basis functions
        self.n_basis = self.multi_indices.shape[0]

        # construct 2D list with BasisFunction objects and array with coefficients and
        # initialize array of basis coefficients
        workhorse_partial = partial(self.set_basis, problem=problem)

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            out = pool.map(workhorse_partial, range(self.n_basis))
            self.b = [o[0] for o in out]
            self.b_array = np.concatenate([o[1] for o in out])
            self.b_array_grad = np.concatenate([o[2] for o in out])

        # This is the single core implementation:
        # self.b = [[0 for _ in range(self.dim)] for _ in range(self.n_basis)]
        #
        # for i_basis in range(self.n_basis):
        #     for i_dim, p in enumerate(problem.parameters_random):   # OrderedDict of RandomParameter objects
        #         self.b[i_basis][i_dim] = problem.parameters_random[p].init_basis_function(
        #             order=self.multi_indices[i_basis, i_dim])

        # Generate unique IDs of basis functions
        self.b_id = [uuid.uuid4() for _ in range(self.n_basis)]

        # initialize normalization factor (self.b_norm and self.b_norm_basis)
        self.init_basis_norm()

    def init_basis_norm(self):
        """
        Construct array of scaling factors self.b_norm [n_basis x dim] and self.b_norm_basis [n_basis x 1]
        to normalize basis functions <psi^2> = int(psi^2*p)dx
        """
        # read individual normalization factors from function objects
        self.b_norm = np.array([list(map(lambda x:x.fun_norm, _b)) for _b in self.b])

        # determine global normalization factor of basis function
        self.b_norm_basis = np.prod(self.b_norm, axis=1)

    def set_basis_poly(self, order, order_max, order_max_norm, interaction_order, interaction_order_current, problem):
        """
        Sets up polynomial basis self.b for given order, order_max_norm and interaction order. Adds only the basis
        functions, which are not yet included.

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
        interaction_order_current : int, optional, default: interaction_order
            Number of random variables currently interacting with respect to the highest order.
            (interaction_order_current <= interaction_order)
            The parameters for lower orders are all interacting with interaction_order.
        problem :
            GPC Problem to analyze
        """
        b_added = None

        dim = len(order)

        # determine new possible set of basis functions for next main iteration
        multi_indices_all_new = get_multi_indices(order=order,
                                                  order_max=order_max,
                                                  order_max_norm=order_max_norm,
                                                  interaction_order=interaction_order,
                                                  interaction_order_current=interaction_order_current)

        # delete multi-indices, which are already present
        if self.b is not None:
            multi_indices_all_current = np.array([list(map(lambda x: x.p["i"], _b)) for _b in self.b])

            idx_old = np.hstack([np.where((multi_indices_all_current[i, :] == multi_indices_all_new).all(axis=1))
                                 for i in range(multi_indices_all_current.shape[0])])

            multi_indices_all_new = np.delete(multi_indices_all_new, idx_old, axis=0)

        if multi_indices_all_new.any():

            # construct 2D list with new BasisFunction objects
            b_added = [[0 for _ in range(dim)] for _ in range(multi_indices_all_new.shape[0])]

            for i_basis in range(multi_indices_all_new.shape[0]):
                for i_p, p in enumerate(problem.parameters_random):
                    b_added[i_basis][i_p] = problem.parameters_random[p].init_basis_function(
                        order=multi_indices_all_new[i_basis, i_p])

            # extend basis
            self.extend_basis(b_added)

        return b_added

    def extend_basis(self, b_added):
        """
        Extend set of basis functions. Skips basis functions, which are already present in self.b.

        Parameters
        ----------
        b_added: list of list of BasisFunction instances [n_b_added][dim]
            Individual BasisFunctions to add
        """
        if self.b is None:
            self.b = []

        if self.b_id is None:
            self.b_id = []

        # add b_added to b (check for duplicates) and generate IDs
        for i_row, _b in enumerate(b_added):
            if _b not in self.b:
                self.b.append(_b)
                self.b_id.append(uuid.uuid4())

        # update size
        self.n_basis = len(self.b)

        # update normalization factors
        self.init_basis_norm()

        # extend array of basis coefficients
        self.extend_basis_array(b_added)

    def init_basis_array(self):
        """
        Initialize polynomial basis coefficients for fast processing. Converts list of lists of self.b
        into np.ndarray that can be processed on multi core systems.
        """

        _b_array = []
        _b_array_grad = []
        for i_basis in range(self.n_basis):
            for i_dim_outer in range(self.dim):
                for i_dim_inner in range(self.dim):
                    if i_dim_outer == 0:
                        _b_array = _b_array + [np.array([self.b[i_basis][i_dim_inner].fun.order]),
                                               self.b[i_basis][i_dim_inner].fun.c]
                    if i_dim_outer == i_dim_inner:
                        _b_array_grad = _b_array_grad + [np.array([self.b[i_basis][i_dim_inner].fun.deriv().order]),
                                                         self.b[i_basis][i_dim_inner].fun.deriv().c]
                    else:
                        _b_array_grad = _b_array_grad + [np.array([self.b[i_basis][i_dim_inner].fun.order]),
                                                         self.b[i_basis][i_dim_inner].fun.c]

        self.b_array = np.concatenate(_b_array)
        self.b_array_grad = np.concatenate(_b_array_grad)

    def extend_basis_array(self, b_added):
        """
        Extends polynomial basis coefficients for fast processing. Converts list of lists of b_added
        into np.ndarray that can be processed on multi core systems.

        Parameters
        ----------
        b_added: list of list of BasisFunction instances [n_b_added][dim]
            Individual BasisFunctions to add
        """

        _b_array = []
        _b_array_grad = []
        for i_basis in range(len(b_added)):
            for i_dim_outer in range(self.dim):
                for i_dim_inner in range(self.dim):
                    if i_dim_outer == 0:
                        _b_array = _b_array + [np.array([b_added[i_basis][i_dim_inner].fun.order]),
                                               b_added[i_basis][i_dim_inner].fun.c]
                    if i_dim_outer == i_dim_inner:
                        _b_array_grad = _b_array_grad + [np.array([b_added[i_basis][i_dim_inner].fun.deriv().order]),
                                                         b_added[i_basis][i_dim_inner].fun.deriv().c]
                    else:
                        _b_array_grad = _b_array_grad + [np.array([b_added[i_basis][i_dim_inner].fun.order]),
                                                         b_added[i_basis][i_dim_inner].fun.c]

        if self.b_array is not None:
            self.b_array = np.hstack((self.b_array, np.concatenate(_b_array)))
        else:
            self.b_array = np.concatenate(_b_array)

        if self.b_array_grad is not None:
            self.b_array_grad = np.hstack((self.b_array_grad, np.concatenate(_b_array_grad)))
        else:
            self.b_array_grad = np.concatenate(_b_array_grad)

    def plot_basis(self, dims, fn_plot=None, dynamic_plot_update=False):
        """
        Generate 2D or 3D cube-plot of basis functions.

        Parameters
        ----------
        dims : list of int of length [2] or [3]
            Indices of parameters in gPC expansion to plot
        fn_plot : str, optional, default: None
            Filename of plot to save (with .png or .pdf extension)

        Returns
        -------
        <File> : *.png and *.pdf file
            Plot of basis functions
        """

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=14)

        multi_indices = np.array([list(map(lambda x: x.p["i"], _b)) for _b in self.b])

        fig = plt.figure(figsize=[6, 6])

        if len(dims) == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        for i_poly in range(multi_indices.shape[0]):

            if len(dims) == 2:
                ax.scatter(multi_indices[i_poly, dims[0]],  # lower corner coordinates
                           multi_indices[i_poly, dims[1]],
                           c=np.array([[51, 153, 255]]) / 255.0,  # bar colour
                           marker="s",
                           s=450)  # transparency of the bars

                ax.set_xlabel("$x_1$", fontsize=18)
                ax.set_ylabel("$x_2$", fontsize=18)

                ax.set_xlim([-1, np.max(multi_indices) + 1])
                ax.set_ylim([-1, np.max(multi_indices) + 1])

                ax.set_xticklabels(range(np.max(multi_indices) + 1))
                ax.set_xticks(range(np.max(multi_indices) + 1))
                ax.set_yticklabels(range(np.max(multi_indices) + 1))
                ax.set_yticks(range(np.max(multi_indices) + 1))

                ax.set_aspect('equal', 'box')

            else:
                ax.bar3d(multi_indices[i_poly, dims[0]] - 0.4,  # lower corner coordinates
                         multi_indices[i_poly, dims[1]] - 0.4,
                         multi_indices[i_poly, dims[2]] - 0.4,
                         0.8, 0.8, 0.8,  # width, depth and height
                         color=np.array([51, 153, 255]) / 255.0,  # bar colour
                         alpha=1)  # transparency of the bars
                ax.view_init(elev=30, azim=45)

                ax.set_xlabel("$x_1$", fontsize=18)
                ax.set_ylabel("$x_2$", fontsize=18)
                ax.set_zlabel("$x_3$", fontsize=18)

                ax.set_xlim([0, np.max(multi_indices) + 1])
                ax.set_ylim([0, np.max(multi_indices) + 1])
                ax.set_zlim([0, np.max(multi_indices) + 1])

                ax.set_xticklabels(range(np.max(multi_indices) + 1))
                ax.set_xticks(range(np.max(multi_indices) + 1))
                ax.set_yticklabels(range(np.max(multi_indices) + 1))
                ax.set_yticks(range(np.max(multi_indices) + 1))
                ax.set_zticklabels(range(np.max(multi_indices) + 1))
                ax.set_zticks(range(np.max(multi_indices) + 1))

        if fn_plot is not None:
            if os.path.splitext(fn_plot) not in [".pdf", ".png"]:
                fn_plot = fn_plot + ".png"
            plt.savefig(fn_plot, dpi=600)
