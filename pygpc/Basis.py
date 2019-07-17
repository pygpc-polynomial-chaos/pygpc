# -*- coding: utf-8 -*-
from .BasisFunction import *
from .misc import get_multi_indices
import uuid
import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ModuleNotFoundError:
    warnings.warn("If you want to use plot functionality from pygpc, "
                  "please install matplotlib (pip install matplotlib).")
    pass


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

        if self.dim == 1:
            multi_indices = np.linspace(0, order_max, order_max + 1, dtype=int)[:, np.newaxis]
        else:
            multi_indices = get_multi_indices(order=order,
                                              order_max=order_max,
                                              order_max_norm=order_max_norm,
                                              interaction_order=interaction_order,
                                              interaction_order_current=interaction_order_current)

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
        self.init_polynomial_basis_gpu()

    def init_b_norm(self):
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

        # initialize gpu coefficient array
        self.init_polynomial_basis_gpu()

    # TODO: @Lucas (GPU) adapt this to function objects
    def init_polynomial_basis_gpu(self):
        """
        Initialize polynomial basis coefficients for GPU. Converts list of lists of self.b
        into np.ndarray that can be processed on a GPU.
        """
        for i_basis in range(len(self.b)):
            for i_dim in range(self.dim):
                polynomial_order = self.b[i_basis][i_dim].fun.order
                self.b_gpu = np.append(self.b_gpu, polynomial_order)
                self.b_gpu = np.append(self.b_gpu, np.flip(self.b[i_basis][i_dim].fun.c))

        for i_dim_gradient in range(self.dim):
            _b_gpu_grad = np.array(())
            for i_basis in range(len(self.b)):
                for i_dim in range(self.dim):
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

    def plot_basis(self, dims, fn_plot=None, dynamic_plot_update=False):
        """
        Generate 2D or 3D cube-plot of basis functions.

        Parameters
        ----------
        dims : list of int of length [2] or [3]
            Indices of parameters in gPC expansion to plot
        fn_plot : str, optional, default: None
            Filename of plot to save (without extension, it will be saved in .png and .pdf format)

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
                ax.bar3d(multi_indices[i_poly, dims[0]] - 0.4, # lower corner coordinates
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
