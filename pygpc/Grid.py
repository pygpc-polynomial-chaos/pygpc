# -*- coding: utf-8 -*-
"""
Functions and classes that provide data and methods for the generation and processing of numerical grids
"""

import numpy as np
from scipy.fftpack import ifft
from sklearn.utils.extmath import cartesian

from .misc import *
from .io import iprint


class Grid:
    """
    Grid class
    """
    def __init__(self, problem):
        """
        Constructor; Initialize Grid class

        Parameters
        ----------
        problem : Problem object
            Object instance of gPC Problem to investigate
        """
        self.N_grid = None
        self.problem = problem

    @staticmethod
    def get_quadrature_jacobi_1d(n, b, a):
        """
        Get knots and weights of Jacobi polynomials (beta distribution).

        knots, weights = Grid.get_quadrature_jacobi_1d(n, b, a)

        Parameters
        ----------
        n: int
            Number of knots
        a: float
            Lower limit of quadrature coefficients
        b: float
            Upper limit of quadrature coefficients

        Returns
        -------
        knots: np.ndarray
            Knots of the grid
        weights: np.ndarray
            Weights of the grid
        """
        # make array to count N: 0, 1, ..., N-1
        n_arr = np.arange(1, n)

        # compose diagonals for companion matrix
        t01 = 1.0 * (b - a) / (2 + a + b)
        t02 = 1.0 * ((b - a) * (a + b)) / ((2 * n_arr + a + b) * (2 * n_arr + 2 + a + b))
        t1 = np.append(t01, t02)
        t2 = np.sqrt((4.0 * n_arr * (n_arr + a) * (n_arr + b) * (n_arr + a + b)) / (
                (2 * n_arr - 1 + a + b) * (2 * n_arr + a + b) ** 2 * (2 * n_arr + 1 + a + b)))

        # compose companion matrix
        t = np.diag(t1) + np.diag(t2, 1) + np.diag(t2, -1)

        # evaluate roots of polynomials (the abscissas are the roots of the
        # characteristic polynomial, i.d. the eigenvalues of the companion matrix)
        # the weights can be derived from the corresponding eigenvectors.
        eigvals, eigvecs = np.linalg.eig(t)
        idx_sorted = np.argsort(eigvals)
        eigvals_sorted = eigvals[idx_sorted]

        weights = 2.0 * eigvecs[0, idx_sorted] ** 2
        knots = eigvals_sorted

        return knots, weights

    @staticmethod
    def get_quadrature_hermite_1d(n):
        """
        Get knots and weights of Hermite polynomials (normal distribution).

        knots, weights = Grid.get_quadrature_hermite_1d(n)

        Parameters
        ----------
        n: int
            number of knots

        Returns
        -------
        knots: np.ndarray
            knots of the grid
        weights: np.ndarray
            weights of the grid
        """
        n = np.int(n)
        knots, weights = np.polynomial.hermite_e.hermegauss(n)
        weights = np.array(list(2.0 * weights / np.sum(weights)))

        return knots, weights

    @staticmethod
    def get_quadrature_clenshaw_curtis_1d(n):
        """
        Get the Clenshaw Curtis nodes and weights.

        knots, weights = Grid.get_quadrature_clenshaw_curtis_1d(n)

        Parameters
        ----------
        n: int
            Number of knots

        Returns
        -------
        knots: np.ndarray
            Knots of the grid
        weights: np.ndarray
            Weights of the grid
        """
        n = np.int(n)

        if n == 1:
            knots = 0
            weights = 2
        else:
            n = n - 1
            c = np.zeros((n, 2))
            k = 2 * (1 + np.arange(np.floor(n / 2)))
            c[::2, 0] = 2 / np.hstack((1, 1 - k * k))
            c[1, 1] = -n
            v = np.vstack((c, np.flipud(c[1:n, :])))
            f = np.real(ifft(v, n=None, axis=0))
            knots = f[0:n, 1]
            weights = np.hstack((f[0, 0], 2 * f[1:n, 0], f[n, 0]))

        return knots, weights

    @staticmethod
    def get_quadrature_fejer1_1d(n):
        """
        Computes the Fejer type 1 nodes and weights.

        This method uses a direct approach after Davis and Rabinowitz (2007) [1] and Gautschi (1967) [2].
        The paper by Waldvogel (2006) [3] exhibits a more efficient approach using Fourier transforms.

        knots, weights = Grid.get_quadrature_fejer1_1d(n)

        Parameters
        ----------
        n: int
            Number of knots

        Returns
        -------
        knots: np.ndarray
            Knots of the grid
        weights: np.ndarray
            Weights of the grid

        Notes
        -----
        .. [1] Davis, P. J., & Rabinowitz, P. (2007). Methods of numerical integration.
           Courier Corporation, second edition, ISBN: 0486453391.
           [2] Gautschi, W. (1967). Numerical quadrature in the presence of a singularity.
           SIAM Journal on Numerical Analysis, 4(3), 357-362.
           [3] Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw–Curtis quadrature rules.
           BIT Numerical Mathematics, 46(1), 195-202.
        """
        n = np.int(n)

        theta = np.zeros(n)

        for i in range(0, n):
            theta[i] = float(2 * n - 1 - 2 * i) * np.pi / float(2 * n)

        knots = np.zeros(n)

        for i in range(0, n):
            knots[i] = np.cos(theta[i])

        weights = np.zeros(n)

        for i in range(0, n):
            weights[i] = 1.0
            jhi = (n // 2)
            for j in range(0, jhi):
                angle = 2.0 * float(j + 1) * theta[i]
                weights[i] = weights[i] - 2.0 * np.cos(angle) / float(4 * (j + 1) ** 2 - 1)

        for i in range(0, n):
            weights[i] = 2.0 * weights[i] / float(n)

        return knots, weights

    @staticmethod
    def get_quadrature_fejer2_1d(n):
        """
        Computes the Fejer type 2 nodes and weights (Clenshaw Curtis without boundary nodes).

        This method uses a direct approach after Davis and Rabinowitz (2007) [1] and Gautschi (1967) [2].
        The paper by Waldvogel (2006) [3] exhibits a more efficient approach using Fourier transforms.

        knots, weights = Grid.get_quadrature_fejer2_1d(n)

        Parameters
        ----------
        n: int
            Number of knots

        Returns
        -------
        knots: np.ndarray
            Knots of the grid
        weights: np.ndarray
            Weights of the grid

        Notes
        -----
        .. [1] Davis, P. J., & Rabinowitz, P. (2007). Methods of numerical integration.
           Courier Corporation, second edition, ISBN: 0486453391.
           [2] Gautschi, W. (1967). Numerical quadrature in the presence of a singularity.
           SIAM Journal on Numerical Analysis, 4(3), 357-362.
           [3] Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw–Curtis quadrature rules.
           BIT Numerical Mathematics, 46(1), 195-202.
        """
        n = np.int(n)

        if n == 1:
            knots = np.array([0.0])
            weights = np.array([2.0])

        elif n == 2:
            knots = np.array([-0.5, +0.5])
            weights = np.array([1.0, 1.0])

        else:
            theta = np.zeros(n)
            p = 1

            for i in range(0, n):
                theta[i] = float(n - i) * np.pi / float(n + 1)

            knots = np.zeros(n)

            for i in range(0, n):
                knots[i] = np.cos(theta[i])

            weights = np.zeros(n)

            for i in range(0, n):
                weights[i] = 1.0
                jhi = ((n - 1) // 2)

                for j in range(0, jhi):
                    angle = 2.0 * float(j + 1) * theta[i]
                    weights[i] = weights[i] - 2.0 * np.cos(angle) / float(4 * (j + 1) ** 2 - 1)
                    p = 2 * ((n + 1) // 2) - 1

                weights[i] = weights[i] - np.cos(float(p + 1) * theta[i]) / float(p)

            for i in range(0, n):
                weights[i] = 2.0 * weights[i] / float(n + 1)

        return knots, weights

    @staticmethod
    def get_quadrature_patterson_1d(n):
        """
        Computes the nested Gauss-Patterson nodes and weights for n = 1,3,7,15,31 nodes.

        knots, weights = Grid.get_quadrature_patterson_1d(n)

        Parameters
        ----------
        n: int
            Number of knots (possible values: 1, 3, 7, 15, 31)

        Returns
        -------
        knots: np.ndarray
            Knots of the grid
        weights: np.ndarray
            Weights of the grid
        """
        x = np.zeros(n)
        w = np.zeros(n)

        if n == 1:

            x = 0.0

            w = 2.0

        elif n == 3:

            x[0] = -0.77459666924148337704
            x[1] = 0.0
            x[2] = 0.77459666924148337704

            w[0] = 0.555555555555555555556
            w[1] = 0.888888888888888888889
            w[2] = 0.555555555555555555556

        elif n == 7:

            x[0] = -0.96049126870802028342
            x[1] = -0.77459666924148337704
            x[2] = -0.43424374934680255800
            x[3] = 0.0
            x[4] = 0.43424374934680255800
            x[5] = 0.77459666924148337704
            x[6] = 0.96049126870802028342

            w[0] = 0.104656226026467265194
            w[1] = 0.268488089868333440729
            w[2] = 0.401397414775962222905
            w[3] = 0.450916538658474142345
            w[4] = 0.401397414775962222905
            w[5] = 0.268488089868333440729
            w[6] = 0.104656226026467265194

        elif n == 15:

            x[0] = -0.99383196321275502221
            x[1] = -0.96049126870802028342
            x[2] = -0.88845923287225699889
            x[3] = -0.77459666924148337704
            x[4] = -0.62110294673722640294
            x[5] = -0.43424374934680255800
            x[6] = -0.22338668642896688163
            x[7] = 0.0
            x[8] = 0.22338668642896688163
            x[9] = 0.43424374934680255800
            x[10] = 0.62110294673722640294
            x[11] = 0.77459666924148337704
            x[12] = 0.88845923287225699889
            x[13] = 0.96049126870802028342
            x[14] = 0.99383196321275502221

            w[0] = 0.0170017196299402603390
            w[1] = 0.0516032829970797396969
            w[2] = 0.0929271953151245376859
            w[3] = 0.134415255243784220360
            w[4] = 0.171511909136391380787
            w[5] = 0.200628529376989021034
            w[6] = 0.219156858401587496404
            w[7] = 0.225510499798206687386
            w[8] = 0.219156858401587496404
            w[9] = 0.200628529376989021034
            w[10] = 0.171511909136391380787
            w[11] = 0.134415255243784220360
            w[12] = 0.0929271953151245376859
            w[13] = 0.0516032829970797396969
            w[14] = 0.0170017196299402603390

        elif n == 31:

            x[0] = -0.99909812496766759766
            x[1] = -0.99383196321275502221
            x[2] = -0.98153114955374010687
            x[3] = -0.96049126870802028342
            x[4] = -0.92965485742974005667
            x[5] = -0.88845923287225699889
            x[6] = -0.83672593816886873550
            x[7] = -0.77459666924148337704
            x[8] = -0.70249620649152707861
            x[9] = -0.62110294673722640294
            x[10] = -0.53131974364437562397
            x[11] = -0.43424374934680255800
            x[12] = -0.33113539325797683309
            x[13] = -0.22338668642896688163
            x[14] = -0.11248894313318662575
            x[15] = 0.0
            x[16] = 0.11248894313318662575
            x[17] = 0.22338668642896688163
            x[18] = 0.33113539325797683309
            x[19] = 0.43424374934680255800
            x[20] = 0.53131974364437562397
            x[21] = 0.62110294673722640294
            x[22] = 0.70249620649152707861
            x[23] = 0.77459666924148337704
            x[24] = 0.83672593816886873550
            x[25] = 0.88845923287225699889
            x[26] = 0.92965485742974005667
            x[27] = 0.96049126870802028342
            x[28] = 0.98153114955374010687
            x[29] = 0.99383196321275502221
            x[30] = 0.99909812496766759766

            w[0] = 0.00254478079156187441540
            w[1] = 0.00843456573932110624631
            w[2] = 0.0164460498543878109338
            w[3] = 0.0258075980961766535646
            w[4] = 0.0359571033071293220968
            w[5] = 0.0464628932617579865414
            w[6] = 0.0569795094941233574122
            w[7] = 0.0672077542959907035404
            w[8] = 0.0768796204990035310427
            w[9] = 0.0857559200499903511542
            w[10] = 0.0936271099812644736167
            w[11] = 0.100314278611795578771
            w[12] = 0.105669893580234809744
            w[13] = 0.109578421055924638237
            w[14] = 0.111956873020953456880
            w[15] = 0.112755256720768691607
            w[16] = 0.111956873020953456880
            w[17] = 0.109578421055924638237
            w[18] = 0.105669893580234809744
            w[19] = 0.100314278611795578771
            w[20] = 0.0936271099812644736167
            w[21] = 0.0857559200499903511542
            w[22] = 0.0768796204990035310427
            w[23] = 0.0672077542959907035404
            w[24] = 0.0569795094941233574122
            w[25] = 0.0464628932617579865414
            w[26] = 0.0359571033071293220968
            w[27] = 0.0258075980961766535646
            w[28] = 0.0164460498543878109338
            w[29] = 0.00843456573932110624631
            w[30] = 0.00254478079156187441540
        else:
            print("Number of points does not match Gauss-Patterson quadrature rule.")
            raise NotImplementedError

        knots = x
        weights = w

        return knots, weights

    def get_denormalized_coordinates(self, coords_norm):
        """
        Denormalize grid from standardized ([-1, 1] except hermite) to original parameter space for simulations.

        coords = Grid.get_denormalized_coordinates(coords_norm)

        Parameters
        ----------
        coords_norm: [N_samples x dim] np.ndarray
            normalized [-1, 1] coordinates xi

        Returns
        -------
        coords: [N_samples x dim] np.ndarray
            Denormalized coordinates xi
        """
        coords = np.zeros(coords_norm.shape)

        for i_dim in range(coords_norm.shape[1]):

            if self.problem.pdf_type[i_dim] == "beta":
                coords[:, i_dim] = (coords_norm[:, i_dim] + 1) / \
                                   2 * (self.problem.pdf_limits[i_dim][1] - self.problem.pdf_limits[i_dim][0]) \
                                   + self.problem.pdf_limits[i_dim][0]

            if self.problem.pdf_type[i_dim] == "norm" or self.problem.pdf_type[i_dim] == "normal":
                coords[:, i_dim] = coords_norm[:, i_dim] * self.problem.pdf_shape[i_dim][1] + \
                                   self.problem.pdf_shape[i_dim][0]

        return coords

    def get_normalized_coordinates(self, coords):
        """
        Normalize grid from original parameter (except hermite) to standardized ([-1, 1] space for simulations.

        coords_norm = Grid.get_normalized_coordinates(coords)

        Parameters
        ----------
        coords: [N_samples x dim] np.ndarray
            Denormalized coordinates xi in original parameter space

        Returns
        -------
        coords_norm: [N_samples x dim] np.ndarray
            Normalized [-1, 1] coordinates xi
        """
        coords_norm = np.zeros(coords.shape)

        for i_dim in range(coords.shape[1]):

            if self.problem.pdf_type[i_dim] == "beta":
                coords_norm[:, i_dim] = (coords[:, i_dim] - self.problem.pdf_limits[i_dim][0])
                coords_norm[:, i_dim] = coords_norm[:, i_dim] / \
                                        (self.problem.pdf_limits[i_dim][1] - self.problem.pdf_limits[i_dim][0]) * \
                                        2.0 - 1

            if self.problem.pdf_type[i_dim] == "norm" or self.problem.pdf_type[i_dim] == "normal":
                coords_norm[:, i_dim] = (coords[:, i_dim] - self.problem.pdf_shape[i_dim][0]) / \
                                        self.problem.pdf_shape[i_dim][1]

        return coords_norm


class TensorGrid(Grid):
    """
    Generate TensorGrid object instance.

    TensorGrid(problem, parameters):

    Attributes
    ----------
    self.grid_type: [N_vars] list of str
        Type of quadrature used to construct tensor grid ('jacobi', 'hermite', 'cc', 'fejer2')
    self.knots_dim_list: [dim] list of np.ndarray
        Knots of grid in each dimension
    self.weights_dim_list: [dim] list of np.ndarray
        Weights of grid in each dimension
    self.weights: np.ndarray [n_grid x dim]
        Weights of the grid (all)
    self.coords: [n_grid x dim] np.ndarray
        Denormalized coordinates xi
    self.coords_norm: [n_grid x dim] np.ndarray
        Normalized [-1, 1] coordinates xi
    self.n_grid: int
        Total number of nodes in grid.
    """

    def __init__(self, problem, parameters):
        """
        Constructor; Initializes TensorGrid object instance.

        Parameters
        ----------
        problem: Problem object
            Object instance of gPC Problem to investigate
        parameters: dict
            Grid parameters
            - parameters["grid_type"] ... list of str [dim]: type of grid ('jacobi', 'hermite', 'cc', 'fejer2')
            - parameters["n_dim"] ... list of int [dim]: Number of nodes in each dimension

        Examples
        --------
        >>> import pygpc
        >>> pygpc.Grid.TensorGrid(problem, {"grid_type": ["hermite", "jacobi"], "n_dim": [5, 6]})
        """
        super(TensorGrid, self).__init__(problem)
        self.parameters = parameters
        self.grid_type = parameters["grid_type"]

        # get knots and weights of polynomials in each dimension
        self.knots_dim_list = []
        self.weights_dim_list = []
        for i_dim in range(self.problem.dim):

            # Jacobi polynomials
            if self.grid_type[i_dim] == 'jacobi':
                knots, weights = self.get_quadrature_jacobi_1d(self.parameters["n_dim"][i_dim],
                                                               self.problem.pdf_shape[i_dim][0] - 1,
                                                               self.problem.pdf_shape[i_dim][1] - 1)

            # Hermite polynomials
            elif self.grid_type[i_dim] == 'hermite':
                knots, weights = self.get_quadrature_hermite_1d(self.parameters["n_dim"][i_dim])

            # Clenshaw Curtis
            elif self.grid_type[i_dim] == 'clenshaw_curtis':
                knots, weights = self.get_quadrature_clenshaw_curtis_1d(self.parameters["n_dim"][i_dim])

            # Fejer type 2 (Clenshaw Curtis without boundary nodes)
            elif self.grid_type[i_dim] == 'fejer2':
                knots, weights = self.get_quadrature_fejer2_1d(self.parameters["n_dim"][i_dim])

            # Gauss-Patterson (Nested Legendre rule)
            elif self.grid_type[i_dim] == 'patterson':
                knots, weights = self.get_quadrature_patterson_1d(self.parameters["n_dim"][i_dim])

            else:
                knots = []
                weights = []
                AttributeError("Specified grid_type {} not implemented!".format(self.parameters["grid_type"][i_dim]))

            self.knots_dim_list.append(knots)
            self.weights_dim_list.append(weights)

        # combine coordinates to full tensor grid (all combinations)
        self.coords_norm = cartesian(self.knots_dim_list)

        # rescale normalized coordinates in case of normal distributions and "fejer2" or "cc" grids
        # +- 0.675 * sigma -> 50%
        # +- 1.645 * sigma -> 90%
        # +- 1.960 * sigma -> 95%        
        # +- 2.576 * sigma -> 99%
        # +- 3.000 * sigma -> 99.73%
        for i_dim in range(self.dim):
            if (self.problem.pdf_type[i_dim] == "norm" or self.problem.pdf_type[i_dim] == "normal") and (
                    not (self.grid_type[i_dim] == "hermite")):
                self.coords_norm[:, i_dim] = self.coords_norm[:, i_dim] * 1.960

        # determine combined weights of Gauss quadrature
        self.weights = np.prod(cartesian(self.weights_dim_list), axis=1) / (2.0 ** self.problem.dim)

        # denormalize grid to original parameter space
        self.coords = self.get_denormalized_coordinates(self.coords_norm)

        # Total number of nodes in grid
        self.n_grid = self.coords.shape[0]


class SparseGrid:
    """
    SparseGrid object instance.

    SparseGrid(problem, parameters)

    pdf_type, grid_type, grid_shape, limits, level, level_max, interaction_order, order_sequence_type, make_grid=True, verbose=True)

    Attributes
    ----------
    pdf_type: [N_vars] list of str
        variable specific type of PDF ("beta", "normal")
    grid_type: [N_vars] list of str
        specify type of quadrature used to construct sparse grid ('jacobi', 'hermite', 'cc', 'fejer2')
    grid_shape: [2 x N_vars] list of list of float
        shape parameters of PDF
        beta (jacobi):  [alpha, beta]
        norm (hermite): [mean, std]
    limits: [2 x N_vars] list of list of float
        upper and lower bounds of PDF
        beta (jacobi):  [min, max]
        norm (hermite): [0, 0] (unused)
    level: [N_vars] list of int
        number of levels in each dimension
    level_max: int
        global combined level maximum
    interaction_order: int
        interaction order of parameters and grid, i.e. the grid points are lying between this number of dimensions
    order_sequence_type: str
        type of order sequence ('lin', 'exp') common: 'exp'
    make_grid: boolean
        boolean value to determine if to generate grid during initialization
    verbose: bool
        boolean value to determine if to print out the progress into the standard output
    coords_norm: [N_samples x dim] np.ndarray
        normalized [-1, 1] coordinates xi
    weights: np.ndarray
        weights of the grid
    coords: [N_samples x dim] np.ndarray
        denormalized coordinates xi
    level_sequence: list of int
        list containing the levels
    order_sequence: list of int
        list containing the polynomial order of the levels
    dim: int
        number of uncertain parameters to process

    """

    def __init__(self, problem, parameters):
        """
        Constructor; Initializes SparseGrid class


        Parameters
        ----------
        problem: Problem object

        parameters: dict

        grid_type: [N_vars] list of str
            specify type of quadrature used to construct sparse grid ('jacobi', 'hermite', 'cc', 'fejer2')
        level: [N_vars] list of int
            number of levels in each dimension
        level_max: int
            global combined level maximum
        interaction_order: int
            interaction order of parameters and grid, i.e. the grid points are lying between this number of dimensions
        order_sequence_type: str
            type of order sequence ('lin', 'exp') common: 'exp'
        make_grid: boolean, optional, default=True
            boolean value to determine if to generate grid during initialization
        verbose: bool, optional, default=True
            boolean value to determine if to print out the progress into the standard output

        Notes
        -----
        Adds Attributes:


        """

        self.problem = problem
        self.grid_type = parameters["grid_type"]
        self.level = parameters["level"]  # number of levels in each dimension [dim x 1]
        self.level_max = parameters["level_max"]  # global combined level maximum
        self.interaction_order = parameters["interaction_order"]  # interaction order of parameters and grid
        self.order_sequence_type = parameters["order_sequence_type"]  # 'lin', 'exp' type of order sequence (common: 'exp')

        self.coords = None  # coordinates of gpc model calculation in the system space
        self.coords_norm = None  # coordinates of gpc model calculation in the gpc space [-1,1]
        self.weights = None  # weights for numerical integration for every point in the coordinate space
        self.level_sequence = []  # integer sequence of levels
        self.order_sequence = []  # integer sequence of polynom order of levels

        # output while grid generation on/off
        if "verbose" not in parameters.keys():
            self.verbose = False

        # Generate grid if not specified
        if "verbose" not in parameters.keys():
            self.make_grid = True

        # grid is generated during initialization or coords, coords_norm and weights are added manually
        if parameters["make_grid"]:
            self.calc_multi_indices()
            self.calc_coords_weights()
        else:
            iprint('Sparse grid initialized but not generated. Please add coords / coords_norm and weights manually.')

    def calc_multi_indices(self):
        """
        Calculate the multi index list needed for the calculation of the SparseGrid.
        """
        for i_dim in range(self.problem.dim):
            
            if self.grid_type[i_dim] == 'fejer2':
                self.level_sequence.append(
                    [element for element in range(1, self.level[i_dim] + 1)])
            else:
                self.level_sequence.append(
                    [element for element in range(self.level[i_dim] + 1)])

            if self.order_sequence_type == 'exp':  # order = 2**level + 1

                if self.grid_type[i_dim] == 'fejer2':  # start with order = 1 @ level = 1
                    self.order_sequence.append((np.power(2, np.arange(1, self.level[i_dim])) - 1).tolist())
                    self.order_sequence[i_dim][0] = 1

                elif self.grid_type[i_dim] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    self.order_sequence.append((np.power(2, np.arange(0, self.level[i_dim])) + 1).tolist())

                else:  # start with order = 1 @ level = 0
                    self.order_sequence.append(
                        (2 ** np.linspace(0, self.level[i_dim], self.level[i_dim] + 1) + 1).tolist())
                    self.order_sequence[i_dim][0] = 1

            elif self.order_sequence_type == 'lin':  # order = level
                if self.grid_type[i_dim] == 'fejer2':  # start with level = 1 @ order = 1
                    self.order_sequence.append(np.linspace(1, self.level[i_dim] + 1, self.level[i_dim] + 1).tolist())

                elif self.grid_type[i_dim] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    iprint("Not possible in case of Gauss-Patterson grid.")

                else:  # start with
                    self.order_sequence.append(np.linspace(1, self.level[i_dim] + 1, self.level[i_dim] + 1).tolist())

    def calc_l_level(self):
        """
        Calculate the l-level needed for the Fejer grid type 2.

        l_level = calc_l_level()

        Returns
        -------
        l_level: np.ndarray
            multi indices filtered by level capacity and interaction order
        """
        if "fejer2" in self.grid_type:
            if self.problem.dim == 1:
                l_level = np.array([np.linspace(1, self.level_max, self.level_max)]).transpose()
            else:
                l_level = get_multi_indices(self.problem.dim, self.level_max - self.problem.dim)
                l_level = l_level + 1
        else:
            if self.problem.dim == 1:
                l_level = np.array([np.linspace(0, self.level_max, self.level_max + 1)]).transpose()
            else:
                l_level = get_multi_indices(self.problem.dim, self.level_max)

        # filter out rows exceeding the individual level cap
        for i_dim in range(self.problem.dim):
            l_level = l_level[l_level[:, i_dim] <= self.level[i_dim]]

        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.problem.dim:
            if any("fejer2" in s for s in self.grid_type):
                l_level = l_level[np.sum(l_level > 1, axis=1) <= self.interaction_order, :]
            else:
                l_level = l_level[np.sum(l_level > 0, axis=1) <= self.interaction_order, :]

        return l_level

    def calc_grid(self):
        """
        Calculate a cubature lookup table for knots and weights.

        dl_k, dl_w = calc_grid()

        Returns
        -------
        dl_k: list of list of float
            cubature lookup table for knots
        dl_w: list of list of float
            cubature lookup table for weights
        """
        # make cubature lookup table for knots (dl_k) and weights (dl_w) [max(l) x dim]
        iprint("Generating difference grids...", tab=1)
        dl_k = [[0 for _ in range(self.dim)] for _ in range(int(np.amax(self.level) + 1))]
        dl_w = [[0 for _ in range(self.dim)] for _ in range(int(np.amax(self.level) + 1))]
        knots_l, weights_l, knots_l_1, weights_l_1 = 0, 0, 0, 0

        for i_dim in range(self.dim):

            for i_level in self.level_sequence[i_dim]:

                if self.grid_type[i_dim] == 'jacobi':  # Jacobi polynomials
                    knots_l, weights_l = get_quadrature_jacobi_1d(self.order_sequence[i_dim][i_level],
                                                              self.grid_shape[0][i_dim] - 1,
                                                              self.grid_shape[1][i_dim] - 1)
                    knots_l_1, weights_l_1 = get_quadrature_jacobi_1d(self.order_sequence[i_dim][i_level - 1],
                                                                  self.grid_shape[0][i_dim] - 1,
                                                                  self.grid_shape[1][i_dim] - 1)

                if self.grid_type[i_dim] == 'hermite':  # Hermite polynomials
                    knots_l, weights_l = get_quadrature_hermite_1d(self.order_sequence[i_dim][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_hermite_1d(self.order_sequence[i_dim][i_level - 1])

                if self.grid_type[i_dim] == 'patterson':  # Gauss-Patterson
                    knots_l, weights_l = get_quadrature_patterson_1d(self.order_sequence[i_dim][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_patterson_1d(self.order_sequence[i_dim][i_level - 1])

                if self.grid_type[i_dim] == 'cc':  # Clenshaw Curtis
                    knots_l, weights_l = get_quadrature_clenshaw_curtis_1d(self.order_sequence[i_dim][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_clenshaw_curtis_1d(self.order_sequence[i_dim][i_level - 1])

                if self.grid_type[i_dim] == 'fejer2':  # Fejer type 2
                    knots_l, weights_l = get_quadrature_fejer2_1d(self.order_sequence[i_dim][i_level - 1])
                    knots_l_1, weights_l_1 = get_quadrature_fejer2_1d(self.order_sequence[i_dim][i_level - 2])

                if (i_level == 0 and not self.grid_type[i_dim] == 'fejer2') or \
                   (i_level == 1 and (self.grid_type[i_dim] == 'fejer2')):
                    dl_k[i_level][i_dim] = knots_l
                    dl_w[i_level][i_dim] = weights_l
                else:
                    dl_k[i_level][i_dim] = np.hstack((knots_l, knots_l_1))
                    dl_w[i_level][i_dim] = np.hstack((weights_l, -weights_l_1))

        return dl_k, dl_w

    def calc_tensor_products(self):
        """
        Calculate the tensor products of the knots and the weights.

        dL_k, dL_w = calc_tensor_products()

        Returns
        -------
        dL_k: np.ndarray
            tensor product of knots
        dL_w: np.ndarray
            tensor product of weights
        """
        # make list of all tensor products according to multiindex list "l"
        iprint("Generating subgrids...", tab=1)
        dl_k, dl_w = self.calc_grid()
        l_level = self.calc_l_level()
        dL_k = []
        dL_w = []

        for i_l_level in range(l_level.shape[0]):

            knots = []
            weights = []

            for i_dim in range(self.dim):
                knots.append(np.asarray(dl_k[np.int(l_level[i_l_level, i_dim])][i_dim], dtype=float))
                weights.append(np.asarray(dl_w[np.int(l_level[i_l_level, i_dim])][i_dim], dtype=float))

            # tensor product of knots
            dL_k.append(cartesian(knots))

            # tensor product of weights
            dL_w.append(np.prod(cartesian(weights), axis=1))

        dL_w = np.hstack(dL_w)
        dL_k = np.vstack(dL_k)

        return dL_k, dL_w

    def calc_coords_weights(self):
        """
        Find similar points in grid and formulate calculate a list containing these points.
        """
        # find similar points in grid and formulate Point list
        iprint("Merging subgrids...", tab=1)
        dL_k, dL_w = self.calc_tensor_products()
        point_number_list = np.zeros(dL_w.shape[0]) - 1
        point_no = 0
        epsilon_k = 1E-6
        coords_norm = []

        while any(point_number_list < 0):
            not_found = point_number_list < 0
            dL_k_nf = dL_k[not_found]
            point_temp = np.zeros(dL_k_nf.shape[0]) - 1
            point_temp[np.sum(np.abs(dL_k_nf - dL_k_nf[0]), axis=1) < epsilon_k] = point_no
            point_number_list[not_found] = point_temp
            point_no = point_no + 1
            coords_norm.append(dL_k_nf[0, :])

        coords_norm = np.array(coords_norm)
        point_number_list = np.asarray(point_number_list, dtype=int)

        weights = np.zeros(np.amax(point_number_list) + 1) - 999

        for i_point in range(np.amax(point_number_list) + 1):
            weights[i_point] = np.sum(dL_w[point_number_list == i_point])

        # filter for very small weights
        iprint("Filter grid for very small weights...", tab=1)
        epsilon_w = 1E-8 / self.dim
        keep_point = np.abs(weights) > epsilon_w
        self.weights = weights[keep_point] / 2 ** self.dim
        coords_norm = coords_norm[keep_point]

        # rescale normalized coordinates in case of normal distributions and "fejer2" or "cc" grids
        # +- 0.675 * sigma -> 50%
        # +- 1.645 * sigma -> 90%
        # +- 1.960 * sigma -> 95%
        # +- 2.576 * sigma -> 99%
        # +- 3.000 * sigma -> 99.73%
        for i_dim in range(self.dim):
            if (self.pdf_type[i_dim] == "norm" or self.pdf_type[i_dim] == "normal") and (
                    not (self.grid_type[i_dim] == "hermite")):
                coords_norm[:, i_dim] = coords_norm[:, i_dim] * 1.960

        # denormalize grid to original parameter space
        iprint("Denormalizing grid for computations...", tab=1)
        self.coords_norm = coords_norm
        self.coords = get_denormalized_coordinates(coords_norm, self.pdf_type, self.grid_shape, self.limits)


class RandomGrid:
    """
    Generate RandomGrid object instance.

    RandomGrid(pdf_type, grid_shape, limits, N, seed=None)

    Attributes
    ----------
    pdf_type: [N_vars] list of str
        variable specific type of pdf ("beta", "normal")
    grid_shape: [2 x N_vars] list of list of float
        shape parameters of PDF
        beta (jacobi):  [alpha, beta]
        norm (hermite): [mean, std]
    limits: [2 x N_vars] list of list of float
        Upper and lower bounds of PDF
        beta (jacobi):  [min, max]
        norm (hermite): [0, 0] (unused)
    N: int
        number of random samples to generate
    seed: float
        seeding point to replicate random grids
    dim: int
        number of uncertain parameters to process
    coords: [N_samples x dim] np.ndarray
        denormalized coordinates xi
    coords_norm: [N_samples x dim] np.ndarray
        normalized [-1, 1] coordinates xi

    Parameters
    ----------
    pdf_type: [N_vars] list of str
        variable specific type of pdf ("beta", "normal")
    grid_shape: [2 x N_vars] list of list of float
        shape parameters of PDF
        beta (jacobi):  [alpha, beta]
        norm (hermite): [mean, std]
    limits: [2 x N_vars] list of list of float
        Upper and lower bounds of PDF
        beta (jacobi):  [min, max]
        norm (hermite): [0, 0] (unused)
    N: int
        number of random samples to generate
    seed: float, optional, default=None
        seeding point to replicate random grids
    """

    def __init__(self, pdf_type, grid_shape, limits, N, seed=None):
        self.pdf_type = pdf_type  # pdf_type: "beta", "normal" [1 x dim]
        self.grid_shape = grid_shape  # pdf_shape: jacobi:->[alpha and beta] hermite:->[mean, variance] list [2 x dim]
        self.limits = limits  # limits: [min, max]  list [2 x dim]
        self.N = int(N)  # Number of random samples
        self.dim = len(self.pdf_type)  # number of dimension
        self.seed = seed  # seed of random grid (if necessary to reproduce random grid)

        # generate random samples for each random input variable [N x dim]
        self.coords_norm = np.zeros([self.N, self.dim])
        if self.seed:
            np.random.seed(self.seed)
        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == "beta":
                self.coords_norm[:, i_dim] = (np.random.beta(self.grid_shape[0][i_dim], self.grid_shape[1][i_dim],
                                                             [self.N, 1]) * 2.0 - 1)[:, 0]
            if self.pdf_type[i_dim] == "norm" or self.pdf_type[i_dim] == "normal":
                self.coords_norm[:, i_dim] = (np.random.normal(0, 1, [self.N, 1]))[:, 0]

        # denormalize grid to original parameter space
        self.coords = get_denormalized_coordinates(self.coords_norm, self.pdf_type, self.grid_shape, self.limits)
