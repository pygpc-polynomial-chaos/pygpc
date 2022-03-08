import uuid
import copy
import numpy as np
import scipy.stats
from .io import iprint
from .Quadrature import *
from .RandomParameter import Beta
from .RandomParameter import Norm
from .misc import compute_chunks
from .misc import mutual_coherence
from .misc import get_multi_indices
from .misc import get_cartesian_product
from .misc import t_averaged_mutual_coherence
from .misc import average_cross_correlation_gram
from .misc import get_different_rows_from_matrices
from scipy.special import gamma

import multiprocessing.pool
from _functools import partial


class Grid(object):
    """
    Grid class

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    weights: ndarray of float [n_grid x dim]
        Weights of the grid (all)
    coords: ndarray of float [n_grid x dim]
        Denormalized coordinates xi
    coords_norm: ndarray of float [n_grid x dim]
        Normalized coordinates xi
    coords_gradient: ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm: ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id: list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    _weights: ndarray of float [n_grid x dim]
        Weights of the grid (all)
    _coords: ndarray of float [n_grid x dim]
        Denormalized coordinates xi
    _coords_norm: ndarray of float [n_grid x dim]
        Normalized coordinates xi
    _domains: ndarray of float [n_grid]
        Domain IDs of grid points for multi-element gPC
    _coords_gradient: ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    _coords_gradient_norm: ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id: list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    n_grid: int
        Total number of nodes in grid.
    """
    def __init__(self, parameters_random, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initialize Grid class
        """
        self._coords = coords                         # Coordinates of gpc model calculation in the system space
        self._coords_norm = coords_norm               # Coordinates of gpc model calculation in the gpc space
        self.coords_id = coords_id                    # Unique IDs of grid points
        self.coords_gradient_id = coords_gradient_id  # Unique IDs of grid gradient points
        self._weights = None                          # Weights for numerical integration
        self.parameters_random = parameters_random    # OrderedDict of RandomParameter instances
        self.dim = len(self.parameters_random)        # Number of random variables
        self._coords_gradient = coords_gradient       # Shifted coordinates for gradient calculation in the system space
        self._coords_gradient_norm = coords_gradient_norm  # Normalized coordinates for gradient calculation
        self.grid_pre = grid_pre                      # Previous grid the new grid is based on

        if coords is not None:
            self.n_grid = self.coords.shape[0]                    # Total number of grid points

        if coords_gradient is not None:
            self.n_grid_gradient = self.coords_gradient.shape[0]  # Total number of grid points for gradient calculation

        if coords_id is None and coords is not None:
            self.coords_id = self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]
            self.n_grid = self._coords.shape[0]

        if coords_gradient_id is None and coords_gradient is not None:
            self.coords_gradient_id = [uuid.uuid4() for _ in range(self.n_grid)]
            self.n_grid_gradient = self._coords_gradient.shape[0]

        if coords is not None and coords_norm is None:
            self.coords_norm = self.get_normalized_coordinates(self.coords)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = value

        if value is not None:
            self.n_grid = self._coords.shape[0]

            # Generate unique IDs of grid points
            if self.coords_id is None:
                self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    @property
    def coords_norm(self):
        return self._coords_norm

    @coords_norm.setter
    def coords_norm(self, value):
        self._coords_norm = value

        if value is not None:
            self.n_grid = self._coords_norm.shape[0]

            # Generate unique IDs of grid points
            if self.coords_id is None:
                self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    @property
    def coords_gradient(self):
        return self._coords_gradient

    @coords_gradient.setter
    def coords_gradient(self, value):
        self._coords_gradient = value

        if value is not None:
            self.n_grid_gradient = self._coords_gradient.shape[0]

            # Generate unique IDs of grid gradient points
            if self.coords_gradient_id is None:
                self.coords_gradient_id = [uuid.uuid4() for _ in range(self.n_grid_gradient)]

    @property
    def coords_gradient_norm(self):
        return self._coords_gradient_norm

    @coords_gradient_norm.setter
    def coords_gradient_norm(self, value):
        self._coords_gradient_norm = value

        if value is not None:
            self.n_grid_gradient = self._coords_gradient_norm.shape[0]

            # Generate unique IDs of grid gradient points
            if self.coords_gradient_id is None:
                self.coords_gradient_id = [uuid.uuid4() for _ in range(self.n_grid_gradient)]

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def get_denormalized_coordinates(self, coords_norm):
        """
        Denormalize grid from normalized to original parameter space for simulations.

        coords = Grid.get_denormalized_coordinates(coords_norm)

        Parameters
        ----------
        coords_norm: [N_samples x dim] np.ndarray
            Normalized coordinates xi

        Returns
        -------
        coords: [N_samples x dim] np.ndarray
            Denormalized coordinates xi
        """
        coords = np.zeros(coords_norm.shape)

        for i_p, p in enumerate(self.parameters_random):

            if self.parameters_random[p].pdf_type == "beta":
                coords[:, i_p, ] = (coords_norm[:, i_p, ] + 1) / \
                                   2 * (self.parameters_random[p].pdf_limits[1] -
                                        self.parameters_random[p].pdf_limits[0]) \
                                   + self.parameters_random[p].pdf_limits[0]

            if self.parameters_random[p].pdf_type in ["norm", "normal"]:
                coords[:, i_p, ] = coords_norm[:, i_p, ] * self.parameters_random[p].pdf_shape[1] + \
                                   self.parameters_random[p].pdf_shape[0]

            if self.parameters_random[p].pdf_type in ["gamma"]:
                coords[:, i_p, ] = coords_norm[:, i_p, ] / self.parameters_random[p].pdf_shape[1] + \
                                   self.parameters_random[p].pdf_shape[2]

        return coords

    def get_normalized_coordinates(self, coords):
        """
        Normalize grid from original parameter to normalized space for simulations.

        coords_norm = Grid.get_normalized_coordinates(coords)

        Parameters
        ----------
        coords : [N_samples x dim] np.ndarray
            Denormalized coordinates xi in original parameter space

        Returns
        -------
        coords_norm : [N_samples x dim] np.ndarray
            Normalized coordinates xi
        """
        coords_norm = np.zeros(coords.shape)

        for i_p, p in enumerate(self.parameters_random):

            if self.parameters_random[p].pdf_type == "beta":
                coords_norm[:, i_p] = (coords[:, i_p] - self.parameters_random[p].pdf_limits[0])
                coords_norm[:, i_p] = coords_norm[:, i_p] / \
                                      (self.parameters_random[p].pdf_limits[1] -
                                       self.parameters_random[p].pdf_limits[0]) * \
                                      2.0 - 1

            if self.parameters_random[p].pdf_type in ["norm", "normal"]:
                coords_norm[:, i_p] = (coords[:, i_p] - self.parameters_random[p].pdf_shape[0]) / \
                                      self.parameters_random[p].pdf_shape[1]

            if self.parameters_random[p].pdf_type in ["gamma"]:
                coords_norm[:, i_p] = (coords[:, i_p] - self.parameters_random[p].pdf_shape[2]) * \
                                      self.parameters_random[p].pdf_shape[1]

        return coords_norm

    def create_gradient_grid(self, delta=1e-3):
        """
        Creates new grid points to determine gradient of model function.
        Adds or updates self.coords_gradient, self.coords_gradient_norm and self.coords_gradient_id.

        Parameters
        ----------
        delta : float, optional, default: 1e-3
            Shift of grid-points along axes in normalized parameter space
        """
        # shift of gradient grid in normalized space
        delta = delta * np.eye(self.dim)

        # Create or update the gradient grid [n_grid x dim x dim]
        if self.coords_gradient is not None:
            n_grid_gradient = self.coords_gradient_norm.shape[0]
            self.coords_gradient = np.vstack((self.coords_gradient,
                                              np.zeros((self.n_grid-n_grid_gradient, self.dim, self.dim))))
            self.coords_gradient_norm = np.vstack((self.coords_gradient_norm,
                                                   np.zeros((self.n_grid-n_grid_gradient, self.dim, self.dim))))
        else:
            n_grid_gradient = 0
            self.coords_gradient = np.zeros((self.n_grid, self.dim, self.dim))
            self.coords_gradient_norm = np.zeros((self.n_grid, self.dim, self.dim))

        if n_grid_gradient < self.coords_gradient.shape[0]:
            # shift the grid
            for i_dim in range(self.dim):
                self.coords_gradient_norm[n_grid_gradient:, :, i_dim] = self.coords_norm[n_grid_gradient:, ] - \
                                                                        delta[i_dim, :]

            # determine coordinates in original parameter space
            self.coords_gradient[n_grid_gradient:, :, :] = \
                self.get_denormalized_coordinates(self.coords_gradient_norm[n_grid_gradient:, :, :])

            # total number of grid points
            self.n_grid_gradient = self.coords_gradient.shape[0]*self.coords_gradient.shape[2]

            # Generate unique IDs of grid points [n_grid]
            self.coords_gradient_id = copy.deepcopy(self.coords_id)


class TensorGrid(Grid):
    """
    Generate TensorGrid object instance.

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    options: dict
        Grid options
        - parameters["grid_type"] ... list of str [dim]: type of grid ('jacobi', 'hermite', 'cc', 'fejer2')
        - parameters["n_dim"] ... list of int [dim]: Number of nodes in each dimension
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    knots_dim_list : list of float [dim][n_knots]
        Knots of polynomials in each dimension
    weights_dim_list : list of float [dim][n_knots]
         Weights of polynomials in each dimension
    weights : ndarray of float [n_grid]
        Quadrature weights for each grid point

    Examples
    --------
    >>> import pygpc
    >>> pygpc.TensorGrid(parameters_random, options={"grid_type": ["hermite", "jacobi"], "n_dim": [5, 6]})

    Attributes
    ----------
    grid_type : [N_vars] list of str
        Type of quadrature used to construct tensor grid ('jacobi', 'hermite', 'clenshaw_curtis', 'fejer2')
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    options: dict
        Grid options
        - parameters["grid_type"] ... list of str [dim]: type of grid ('jacobi', 'hermite', 'cc', 'fejer2')
        - parameters["n_dim"] ... list of int [dim]: Number of nodes in each dimension
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    knots_dim_list : list of float [dim][n_knots]
        Knots of polynomials in each dimension
    weights_dim_list : list of float [dim][n_knots]
         Weights of polynomials in each dimension
    weights : ndarray of float [n_grid]
        Quadrature weights for each grid point
    """

    def __init__(self, parameters_random, options, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 knots_dim_list=None, weights_dim_list=None, weights=None):
        """
        Constructor; Initializes TensorGrid object instance; Generates grid
        """
        super(TensorGrid, self).__init__(parameters_random,
                                         coords=coords,
                                         coords_norm=coords_norm,
                                         coords_gradient=coords_gradient,
                                         coords_gradient_norm=coords_gradient_norm,
                                         coords_id=coords_id,
                                         coords_gradient_id=coords_gradient_id)
        self.options = options
        self.grid_type = options["grid_type"]
        self.n_dim = options["n_dim"]

        if coords is not None and coords_norm is not None:
            grid_present = True

            self.knots_dim_list = knots_dim_list
            self.weights_dim_list = weights_dim_list
            self.weights = weights

        else:
            grid_present = False

        if not grid_present:

            # get knots and weights of polynomials in each dimension
            self.knots_dim_list = []
            self.weights_dim_list = []

            for i_p, p in enumerate(self.parameters_random):

                # Jacobi polynomials
                if self.grid_type[i_p] == 'jacobi':
                    knots, weights = get_quadrature_jacobi_1d(self.n_dim[i_p],
                                                              self.parameters_random[p].pdf_shape[0] - 1,
                                                              self.parameters_random[p].pdf_shape[1] - 1,)

                # Hermite polynomials
                elif self.grid_type[i_p] == 'hermite':
                    knots, weights = get_quadrature_hermite_1d(self.n_dim[i_p])

                # Clenshaw Curtis
                elif self.grid_type[i_p] in ['clenshaw_curtis' or 'cc']:
                    knots, weights = get_quadrature_clenshaw_curtis_1d(self.n_dim[i_p])

                # Fejer type 2 (Clenshaw Curtis without boundary nodes)
                elif self.grid_type[i_p] == 'fejer2':
                    knots, weights = get_quadrature_fejer2_1d(self.n_dim[i_p])

                # Gauss-Patterson (Nested Legendre rule)
                elif self.grid_type[i_p] == 'patterson':
                    knots, weights = get_quadrature_patterson_1d(self.n_dim[i_p])

                else:
                    knots = []
                    weights = []
                    AttributeError("Specified grid_type {} not implemented!".format(self.grid_type[i_p]))

                self.knots_dim_list.append(knots)
                self.weights_dim_list.append(weights)

            # combine coordinates to full tensor grid (all combinations)
            # self.coords_norm = cartesian(self.knots_dim_list)
            self.coords_norm = get_cartesian_product(self.knots_dim_list)

            # rescale normalized coordinates in case of normal distributions and "fejer2" or "cc" grids
            # +- 0.675 * sigma -> 50%
            # +- 1.645 * sigma -> 90%
            # +- 1.960 * sigma -> 95%
            # +- 2.576 * sigma -> 99%
            # +- 3.000 * sigma -> 99.73%
            for i_p, p in enumerate(self.parameters_random):

                if self.parameters_random[p].pdf_type in ["norm", "normal"] and (not(self.grid_type[i_p] == "hermite")):
                    self.coords_norm[:, i_p] = self.coords_norm[:, i_p] * 1.960

            # determine combined weights of Gauss quadrature
            self.weights = np.prod(get_cartesian_product(self.weights_dim_list), axis=1) / (2.0 ** self.dim)

            # denormalize grid to original parameter space
            self.coords = self.get_denormalized_coordinates(self.coords_norm)

            # Total number of nodes in grid
            self.n_grid = self.coords.shape[0]

            # Generate and append unique IDs of grid points
            self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]


class SparseGrid(Grid):
    """
    SparseGrid object instance.

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    options: dict
        Grid parameters
        - grid_type ([N_vars] list of str) ... Type of quadrature rule used to construct sparse grid
          ('jacobi', 'hermite', 'clenshaw_curtis', 'fejer2', 'patterson')
        - level ([N_vars] list of int) ... Number of levels in each dimension
        - level_max (int) ... Global combined level maximum
        - interaction_order (int) ...Interaction order of parameters and grid, i.e. the grid points are lying
          between this number of dimensions
        - order_sequence_type (str) ... Type of order sequence ('lin', 'exp') common: 'exp'
        - make_grid (boolean, optional, default=True) ... Boolean value to determine if to generate grid
          during initialization
        - verbose (bool, optional, default=True) ... Print output messages into stdout
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    level_sequence: list of int
        Integer sequence of levels
    order_sequence: list of int
        Integer sequence of polynomial order of levels
    weights : ndarray of float [n_grid]
        Quadrature weights for each grid point

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.SparseGrid(parameters_random=parameters_random,
    >>>                         options={"grid_type": ["jacobi", "jacobi"],
    >>>                                  "level": [3, 3],
    >>>                                  "level_max": 3,
    >>>                                  "interaction_order": 2,
    >>>                                  "order_sequence_type": "exp"})

    Attributes
    ----------
    grid_type: [N_vars] list of str
        specify type of quadrature used to construct sparse grid ('jacobi', 'hermite', 'cc', 'fejer2')
    level: [N_vars] list of int
        number of levels in each dimension
    level_max: int
        global combined level maximum
    level_sequence: list of int
        list containing the levels
    interaction_order: int
        interaction order of parameters and grid, i.e. the grid points are lying between this number of dimensions
    order_sequence_type: str
        type of order sequence ('lin', 'exp') common: 'exp'
    order_sequence: list of int
        list containing the polynomial order of the levels
    coords : ndarray of float [n_grid_add x dim]
            Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    weights : ndarray of float [n_grid]
            Quadrature weights for each grid point
    verbose: bool
        boolean value to determine if to print out the progress into the standard output
    """

    def __init__(self, parameters_random, options, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 level_sequence=None, order_sequence=None, weights=None):
        """
        Constructor; Initializes SparseGrid class; Generates grid
        """

        super(SparseGrid, self).__init__(parameters_random,
                                         coords=coords,
                                         coords_norm=coords_norm,
                                         coords_gradient=coords_gradient,
                                         coords_gradient_norm=coords_gradient_norm,
                                         coords_id=coords_id,
                                         coords_gradient_id=coords_gradient_id)

        self.grid_type = options["grid_type"]    # Quadrature rule ('jacobi', 'hermite', 'clenshaw_curtis', 'fejer2')
        self.level = options["level"]            # Number of levels in each dimension [dim x 1]
        self.level_max = options["level_max"]    # Global combined level maximum
        self.interaction_order = options["interaction_order"]        # Interaction order of parameters and grid
        self.order_sequence_type = options["order_sequence_type"]    # Order sequence ('lin', 'exp' (common))
        self.level_sequence = []  # Integer sequence of levels
        self.order_sequence = []  # Integer sequence of polynomial order of levels

        # output while grid generation on/off
        if "verbose" not in options.keys():
            self.verbose = False

        # Generate grid if not specified
        if coords is not None and coords_norm is not None:
            grid_present = True

            self.level_sequence = level_sequence
            self.order_sequence = order_sequence
            self.weights = weights

        else:
            grid_present = False

        # Grid is generated during initialization or coords, coords_norm and weights are added manually
        if not grid_present:
            self.calc_multi_indices()
            self.calc_coords_weights()

            # Determine total number of grid points
            self.n_grid = self.coords.shape[0]

            # Generate unique IDs of grid points
            self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    def calc_multi_indices(self):
        """
        Calculate the multi index list needed for the calculation of the SparseGrid.
        """
        for i_p, p in enumerate(self.parameters_random):

            if self.grid_type[i_p] == 'fejer2':
                self.level_sequence.append([element for element in range(1, self.level[i_p] + 1)])

            else:
                self.level_sequence.append([element for element in range(self.level[i_p] + 1)])

            if self.order_sequence_type == 'exp':         # order = 2**level + 1

                if self.grid_type[i_p] == 'fejer2':       # start with order = 1 @ level = 1
                    self.order_sequence.append((np.power(2, np.arange(1, self.level[i_p])) - 1).tolist())
                    self.order_sequence[i_p][0] = 1

                elif self.grid_type[i_p] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    self.order_sequence.append((np.power(2, np.arange(0, self.level[i_p])) + 1).tolist())

                else:                                     # start with order = 1 @ level = 0
                    self.order_sequence.append(
                        (2 ** np.linspace(0, self.level[i_p], self.level[i_p] + 1) + 1).tolist())
                    self.order_sequence[i_p][0] = 1

            elif self.order_sequence_type == 'lin':       # order = level
                if self.grid_type[i_p] == 'fejer2':       # start with level = 1 @ order = 1
                    self.order_sequence.append(np.linspace(1, self.level[i_p] + 1, self.level[i_p] + 1).tolist())

                elif self.grid_type[i_p] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    iprint("Not possible in case of Gauss-Patterson grid.", tab=0, verbose=self.verbose)

                else:                                       # start with
                    self.order_sequence.append(np.linspace(1, self.level[i_p] + 1, self.level[i_p] + 1).tolist())

    def calc_l_level(self):
        """
        Calculate the l-level needed for the Fejer grid type 2.

        l_level = calc_l_level()

        Returns
        -------
        l_level: np.ndarray
            Multi indices filtered by level capacity and interaction order
        """
        if "fejer2" in self.grid_type:
            if self.dim == 1:
                l_level = np.array([np.linspace(1, self.level_max, self.level_max)]).transpose()
            else:
                l_level = get_multi_indices(self.dim, self.level_max - self.dim)
                l_level = l_level + 1
        else:
            if self.dim == 1:
                l_level = np.array([np.linspace(0, self.level_max, self.level_max + 1)]).transpose()
            else:
                l_level = get_multi_indices(order=[self.level_max] * self.dim,
                                            order_max=self.level_max,
                                            interaction_order=self.dim,
                                            order_max_norm=1.,
                                            interaction_order_current=None)

        # filter out rows exceeding the individual level cap
        for i_p in range(self.dim):
            l_level = l_level[l_level[:, i_p] <= self.level[i_p]]

        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.dim:
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
            Cubature lookup table for knots
        dl_w: list of list of float
            Cubature lookup table for weights
        """
        # make cubature lookup table for knots (dl_k) and weights (dl_w) [max(l) x dim]
        iprint("Generating difference grids...", tab=0, verbose=self.verbose)
        dl_k = [[0 for _ in range(self.dim)] for _ in range(int(np.amax(self.level) + 1))]
        dl_w = [[0 for _ in range(self.dim)] for _ in range(int(np.amax(self.level) + 1))]
        knots_l, weights_l, knots_l_1, weights_l_1 = 0, 0, 0, 0

        for i_p, p in enumerate(self.parameters_random):

            for i_level in self.level_sequence[i_p]:

                # Jacobi polynomials
                if self.grid_type[i_p] == 'jacobi':
                    knots_l, weights_l = get_quadrature_jacobi_1d(self.order_sequence[i_p][i_level],
                                                                  self.parameters_random[p].pdf_shape[0] - 1,
                                                                  self.parameters_random[p].pdf_shape[1] - 1)
                    knots_l_1, weights_l_1 = get_quadrature_jacobi_1d(self.order_sequence[i_p][i_level - 1],
                                                                      self.parameters_random[p].pdf_shape[0] - 1,
                                                                      self.parameters_random[p].pdf_shape[1] - 1)

                # Hermite polynomials
                if self.grid_type[i_p] == 'hermite':
                    knots_l, weights_l = get_quadrature_hermite_1d(
                        self.order_sequence[i_p][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_hermite_1d(
                        self.order_sequence[i_p][i_level - 1])

                # Gauss-Patterson
                if self.grid_type[i_p] == 'patterson':
                    knots_l, weights_l = get_quadrature_patterson_1d(
                        self.order_sequence[i_p][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_patterson_1d(
                        self.order_sequence[i_p][i_level - 1])

                # Clenshaw Curtis
                if self.grid_type[i_p] == 'clenshaw_curtis':
                    knots_l, weights_l = get_quadrature_clenshaw_curtis_1d(
                        self.order_sequence[i_p][i_level])
                    knots_l_1, weights_l_1 = get_quadrature_clenshaw_curtis_1d(
                        self.order_sequence[i_p][i_level - 1])

                # Fejer type 2
                if self.grid_type[i_p] == 'fejer2':
                    knots_l, weights_l = get_quadrature_fejer2_1d(
                        self.order_sequence[i_p][i_level - 1])
                    knots_l_1, weights_l_1 = get_quadrature_fejer2_1d(
                        self.order_sequence[i_p][i_level - 2])

                if (i_level == 0 and not self.grid_type[i_p] == 'fejer2') or \
                        (i_level == 1 and (self.grid_type[i_p] == 'fejer2')):
                    dl_k[i_level][i_p] = knots_l
                    dl_w[i_level][i_p] = weights_l
                else:
                    # noinspection PyTypeChecker
                    dl_k[i_level][i_p] = np.hstack((knots_l, knots_l_1))
                    # noinspection PyTypeChecker
                    dl_w[i_level][i_p] = np.hstack((weights_l, -weights_l_1))

        return dl_k, dl_w

    def calc_tensor_products(self):
        """
        Calculate the tensor products of the knots and the weights.

        dll_k, dll_w = calc_tensor_products()

        Returns
        -------
        dll_k: np.ndarray
            Tensor product of knots
        dll_w: np.ndarray
            Tensor product of weights
        """
        # make list of all tensor products according to multi-index list "l"
        iprint("Generating sub-grids...", tab=0, verbose=self.verbose)
        dl_k, dl_w = self.calc_grid()
        l_level = self.calc_l_level()
        dll_k = []
        dll_w = []

        for i_l_level in range(l_level.shape[0]):

            knots = []
            weights = []

            for i_p in range(self.dim):
                knots.append(np.asarray(dl_k[int(l_level[i_l_level, i_p])][i_p], dtype=float))
                weights.append(np.asarray(dl_w[int(l_level[i_l_level, i_p])][i_p], dtype=float))

            # tensor product of knots
            dll_k.append(get_cartesian_product(knots))

            # tensor product of weights
            dll_w.append(np.prod(get_cartesian_product(weights), axis=1))

        dll_w = np.hstack(dll_w)
        dll_k = np.vstack(dll_k)

        return dll_k, dll_w

    def calc_coords_weights(self):
        """
        Determine coords and weights of sparse grid by generating, merging and subtracting sub-grids.
        """
        # find similar points in grid and formulate Point list
        dll_k, dll_w = self.calc_tensor_products()
        point_number_list = np.zeros(dll_w.shape[0]) - 1
        point_no = 0
        epsilon_k = 1E-6
        coords_norm = []

        iprint("Merging sub-grids...", tab=0, verbose=self.verbose)

        while any(point_number_list < 0):
            not_found = point_number_list < 0
            dll_k_nf = dll_k[not_found]
            point_temp = np.zeros(dll_k_nf.shape[0]) - 1
            point_temp[np.sum(np.abs(dll_k_nf - dll_k_nf[0]), axis=1) < epsilon_k] = point_no
            point_number_list[not_found] = point_temp
            point_no = point_no + 1
            coords_norm.append(dll_k_nf[0, :])

        coords_norm = np.array(coords_norm)
        point_number_list = np.asarray(point_number_list, dtype=int)

        weights = np.zeros(np.amax(point_number_list) + 1) - 999

        for i_point in range(np.amax(point_number_list) + 1):
            weights[i_point] = np.sum(dll_w[point_number_list == i_point])

        # filter for very small weights
        iprint("Filter grid for very small weights...", tab=0, verbose=self.verbose)
        epsilon_w = 1E-8 / self.dim
        keep_point = np.abs(weights) > epsilon_w
        self.weights = weights[keep_point] / 2 ** self.dim
        coords_norm = coords_norm[keep_point]

        # rescale normalized coordinates in case of normal distributions and "fejer2" or "clenshaw_curtis" grids
        # +- 0.675 * sigma -> 50%
        # +- 1.645 * sigma -> 90%
        # +- 1.960 * sigma -> 95%
        # +- 2.576 * sigma -> 99%
        # +- 3.000 * sigma -> 99.73%
        for i_p, p in enumerate(self.parameters_random):
            if self.parameters_random[p].pdf_type in ["norm", "normal"] and (not(self.grid_type[i_p] == "hermite")):
                coords_norm[:, i_p] = coords_norm[:, i_p] * 1.960

        # denormalize grid to original parameter space
        iprint("Denormalizing grid for computations...", tab=0, verbose=self.verbose)
        self.coords_norm = coords_norm
        self.coords = self.get_denormalized_coordinates(coords_norm)


class RandomGrid(Grid):
    """
    RandomGrid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options : dict, optional, default=None
        RandomGrid options depending on the grid type
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.RandomGrid(parameters_random=parameters_random, n_grid=100, options=None)

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options : dict, optional, default=None
        RandomGrid options depending on the grid type
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initializes RandomGrid instance; Generates grid
        """
        super(RandomGrid, self).__init__(parameters_random,
                                         coords=coords,
                                         coords_norm=coords_norm,
                                         coords_gradient=coords_gradient,
                                         coords_gradient_norm=coords_gradient_norm,
                                         coords_id=coords_id,
                                         coords_gradient_id=coords_gradient_id,
                                         grid_pre=grid_pre)

        if n_grid is not None:
            self.n_grid = int(n_grid)

        self.options = options
        self.seed = None

        if self.options is None:
            self.options = dict()

        if "seed" in self.options.keys():
            self.seed = self.options["seed"]

            # Seed of random grid (if necessary to reproduce random grid)
            np.random.seed(self.seed)

    def extend_random_grid(self, n_grid_new=None, coords=None, coords_norm=None, classifier=None, domain=None,
                           gradient=False):
        """
        Add sample points according to input pdfs to grid (old points are kept). Define either the new total number of
        grid points with "n_grid_new" or add grid-points manually by providing "coords" and "coords_norm".

        extend_random_grid(n_grid_new, coords=None, coords_norm=None, seed=None, classifier=None, domain=None):

        Parameters
        ----------
        n_grid_new : float
            Total number of grid points in extended random grid (old points are kept)
            (n_grid_add = n_grid_new - n_grid_old)
        coords : ndarray of float [n_grid_add x dim]
            Grid points to add (model space)
        coords_norm : ndarray of float [n_grid_add x dim]
            Grid points to add (normalized space)
        classifier : Classifier object, optional, default: None
            Classifier
        domain : int, optional, default: None
            Adds grid points only in specified domain (needs Classifier object including a predict() method)
        gradient : bool, optional, default: False
            Add corresponding gradient grid points
        """

        # increase seed if needed to avoid creation of same grid points
        if "seed" in self.options.keys() and self.options["seed"] is not None:
            self.options["seed"] += 1
            self.seed = self.options["seed"]

        if n_grid_new is not None:
            # Number of new grid points
            n_grid_add = int(n_grid_new - self.n_grid)

            if n_grid_add > 0:
                # Generate new grid points
                if classifier is None:
                    if isinstance(self, Random):
                        new_grid = Random(parameters_random=self.parameters_random,
                                          n_grid=n_grid_add,
                                          options=self.options)

                        # append points to existing grid
                        self.coords = np.vstack([self.coords, new_grid.coords])
                        self.coords_norm = np.vstack([self.coords_norm, new_grid.coords_norm])

                    elif isinstance(self, LHS):
                        grid_pre = copy.deepcopy(self)
                        new_grid = LHS(parameters_random=self.parameters_random,
                                       n_grid=n_grid_add,
                                       grid_pre=grid_pre,
                                       options=self.options)

                        # append points to existing grid
                        self.coords = np.vstack([self.coords, new_grid.coords])
                        self.coords_norm = np.vstack([self.coords_norm, new_grid.coords_norm])

                    elif isinstance(self, L1) or isinstance(self, L1_LHS) or isinstance(self, LHS_L1) \
                            or isinstance(self, CO) or isinstance(self, FIM):
                        grid_pre = copy.deepcopy(self)
                        new_grid = self.__class__(parameters_random=self.parameters_random,
                                                  n_grid=n_grid_new,
                                                  grid_pre=grid_pre,
                                                  gpc=self.gpc,
                                                  options=self.options)

                        self.coords = new_grid.coords
                        self.coords_norm = new_grid.coords_norm

                else:
                    coords = np.zeros((n_grid_add, len(self.parameters_random)))
                    coords_norm = np.zeros((n_grid_add, len(self.parameters_random)))

                    # add grid points one by one because we are looking for samples in the right domain
                    for i in range(n_grid_add):
                        resample = True
                        while resample:
                            if isinstance(self, Random):
                                new_grid = Random(parameters_random=self.parameters_random,
                                                  n_grid=1,
                                                  options=self.options)

                                # test if grid point lies in right domain
                                if classifier.predict(new_grid.coords_norm)[0] == domain:
                                    coords[i, :] = new_grid.coords
                                    coords_norm[i, :] = new_grid.coords_norm
                                    resample = False

                            elif isinstance(self, LHS):
                                # check if gridpoint exceeds sampling reservoir
                                # if self.n_grid > max(10000, self.n_grid_lhs * 10)
                                # coords.np.append(pygpc.LSH( seed=seed+1))
                                coords_norm_test = self.lhs_extend(self.coords_norm_reservoir, 1)
                                # test if next grid point lies in right domain
                                if classifier.predict(coords_norm_test[len(coords_norm_test) - 1]) == domain:
                                    self.coords_norm_reservoir = coords_norm_test
                                    coords[i, :] = self.coords_reservoir[self.n_grid]
                                    coords_norm[i, :] = self.coords_norm_reservoir[self.n_grid]
                                    self.n_grid += 1
                                    resample = False

                                #else:
                                # delete tested point
                                # self.coords_norm_reservoir = np.delete(self.coords_norm_reservoir, self.n_grid, 0)
                                # self.coords_reservoir = np.delete(self.coords_reservoir, self.n_grid, 0)
                                # self.lhs_reservoir = np.delete(self.lhs_reservoir, self.n_grid, 0)

                    # append points to existing grid
                    self.coords = np.vstack([self.coords, coords])
                    self.coords_norm = np.vstack([self.coords_norm, coords_norm])

        elif coords is not None and coords_norm is not None:
            # Number of new grid points
            n_grid_add = coords.shape[0]

            # check if specified points are lying in right domain
            if classifier is not None:
                if not (classifier.predict(coords_norm) == domain).all():
                    raise AssertionError("Specified coordinates are not lying in right domain!")

            # append points to existing grid
            self.coords = np.vstack([self.coords, coords])
            self.coords_norm = np.vstack([self.coords_norm, coords_norm])

        else:
            raise ValueError("Specify either n_grid_new or coords and coords_norm")

        # Generate and append unique IDs of new grid points
        self.coords_id = self.coords_id + [uuid.uuid4() for _ in range(n_grid_add)]

        self.n_grid = self.coords.shape[0]

        if gradient:
            self.create_gradient_grid()

    def lhs_extend(self, array, n_extend):
        """
        Add sample points to already existing LHS samples

        Parameters
        ----------
        array: ndarray of float [m x n]
            Existing LHS samples with m samples points per n dimensions
        n_extend: int
            Number of new rows of samples needed

        Returns
        -------
        coords_norm : ndarray of float [m + n_extend x n]
            Existing LHS samples with added new samples
        """

        dim = np.shape(array)[1]
        n_old = np.shape(array)[0]
        n_new = n_old + n_extend
        i = 1
        n_new_loop = n_new
        np.random.seed(seed=self.seed)

        while i > 0:
            a_new = np.zeros([n_new_loop, dim]) - 1
            u = np.random.rand(n_new_loop, dim)

            for d in range(dim):
                k = 0
                for j in range(n_new_loop - 1):
                    if not (float(j / n_new_loop) < float(np.sort(array[:, d])[min(j, len(array) - 1)]) < float((j + 1) / n_new_loop)) \
                            and (float((j + 1) / n_new_loop) <= float(np.sort(array[:, d])[min((j + 2), len(array) - 1)])):

                        k = k + 1
                        if k is np.shape(a_new)[0] + 1:
                            k = 1
                        a_new[k - 1, d] = float((j + u[j, d]) / n_new_loop)

            a_new = a_new[(a_new != -1).all(axis=1), :]

            if n_extend > np.min((a_new.shape[0], n_new)):
                i = 1
                n_new_loop = n_new_loop * 2

            else:
                i = 0
                for d in range(dim):
                    np.random.shuffle(a_new[:, d])

                a_extend = a_new[np.random.choice(a_new.shape[0], size=n_extend, replace=False), :]

        coords_ = np.insert(array, n_old, a_extend, axis=0)

        return coords_


class Random(RandomGrid):
    """
    Random grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options : dict, optional, default=None
        RandomGrid options depending on the grid type
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.Random(parameters_random=parameters_random, n_grid=100)

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options : dict, optional, default=None
        RandomGrid options depending on the grid type
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initializes RandomGrid instance; Generates grid or copies provided content
        """
        if n_grid is not None:
            n_grid = int(n_grid)

        super(Random, self).__init__(parameters_random,
                                     n_grid=n_grid,
                                     options=options,
                                     coords=coords,
                                     coords_norm=coords_norm,
                                     coords_gradient=coords_gradient,
                                     coords_gradient_norm=coords_gradient_norm,
                                     coords_id=coords_id,
                                     coords_gradient_id=coords_gradient_id,
                                     grid_pre=grid_pre)

        if coords is not None or coords_norm is not None:
            grid_present = True
        else:
            grid_present = False

        if not grid_present:

            # Generate random samples for each random input variable [n_grid x dim]
            self.coords_norm = np.zeros([self.n_grid, self.dim])

            if self.grid_pre is not None:
                self.coords_norm[:self.grid_pre.n_grid, :] = self.grid_pre.coords_norm
                n_grid_add = n_grid - self.grid_pre.n_grid
                n_grid_start = self.grid_pre.n_grid
            else:
                n_grid_add = n_grid
                n_grid_start = 0

            # in case of seeding, the random grid is constructed element wise (same grid-points when n_grid differs)
            if self.seed:
                for i_grid in range(n_grid_start, n_grid):
                    for i_p, p in enumerate(self.parameters_random):

                        if self.parameters_random[p].pdf_type == "beta":
                            self.coords_norm[i_grid, i_p] = np.random.beta(self.parameters_random[p].pdf_shape[0],
                                                                           self.parameters_random[p].pdf_shape[1],
                                                                           1) * 2.0 - 1

                        if self.parameters_random[p].pdf_type in ["norm", "normal"]:

                            resample = True

                            while resample:
                                self.coords_norm[i_grid, i_p] = np.random.normal(loc=0,
                                                                                 scale=1,
                                                                                 size=1)
                                resample = self.coords_norm[i_grid, i_p] < self.parameters_random[p].x_perc_norm[0] or \
                                           self.coords_norm[i_grid, i_p] > self.parameters_random[p].x_perc_norm[1]

                        if self.parameters_random[p].pdf_type in ["gamma"]:

                            resample = True

                            while resample:
                                self.coords_norm[i_grid, i_p] = np.random.gamma(
                                    shape=self.parameters_random[p].pdf_shape[0],
                                    scale=1.0,
                                    size=1)
                                resample = self.coords_norm[i_grid, i_p] > self.parameters_random[p].x_perc_norm

            else:
                for i_p, p in enumerate(self.parameters_random):

                    if self.parameters_random[p].pdf_type == "beta":
                        self.coords_norm[n_grid_start:, i_p] = (np.random.beta(self.parameters_random[p].pdf_shape[0],
                                                                               self.parameters_random[p].pdf_shape[1],
                                                                               [n_grid_add, 1]) * 2.0 - 1)[:, 0]

                    if self.parameters_random[p].pdf_type in ["norm", "normal"]:
                        resample = True
                        if self.grid_pre is not None:
                            outlier_mask = np.hstack((np.zeros(self.grid_pre.n_grid, dtype=bool),
                                                     np.ones(n_grid_add, dtype=bool)))
                        else:
                            outlier_mask = np.ones(self.n_grid, dtype=bool)

                        while resample:
                            self.coords_norm[outlier_mask, i_p] = (np.random.normal(loc=0,
                                                                                    scale=1,
                                                                                    size=[np.sum(outlier_mask), 1]))[:, 0]

                            outlier_mask = np.logical_or(
                                self.coords_norm[:, i_p] < self.parameters_random[p].x_perc_norm[0],
                                self.coords_norm[:, i_p] > self.parameters_random[p].x_perc_norm[1])

                            resample = outlier_mask.any()

                    if self.parameters_random[p].pdf_type in ["gamma"]:
                        resample = True
                        outlier_mask = np.ones(self.n_grid, dtype=bool)
                        j = 0
                        while resample:
                            # print("Iteration: {}".format(j + 1))
                            self.coords_norm[outlier_mask, i_p] = (np.random.gamma(
                                shape=self.parameters_random[p].pdf_shape[0],
                                scale=1.0,
                                size=[np.sum(outlier_mask), 1]))[:, 0]

                            outlier_mask = np.array(self.coords_norm[:, i_p] > self.parameters_random[p].x_perc_norm)

                            resample = outlier_mask.any()

                            j += 1

            # Denormalize grid to original parameter space
            self.coords = self.get_denormalized_coordinates(self.coords_norm)

            # Generate unique IDs of grid points
            self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

        else:
            self.coords = coords
            self.coords_norm = coords_norm

            self.coords_gradient = coords_gradient
            self.coords_gradient_norm = coords_gradient_norm

            self.coords_id = coords_id
            self.coords_gradient_id = coords_gradient_id

            if self.coords is None:
                # Denormalize grid to original parameter space
                self.coords = self.get_denormalized_coordinates(self.coords_norm)

            if self.coords_norm is None:
                # Normalize grid to original parameter space
                self.coords = self.get_normalized_coordinates(self.coords)

            if self.coords_gradient is None and self.coords_gradient_norm is not None:
                # Denormalize grid to original parameter space
                self.coords_gradient = self.get_denormalized_coordinates(self.coords_gradient_norm)

            if self.coords_gradient_norm is None and self.coords_gradient is not None:
                # Normalize grid to original parameter space
                self.coords_gradient_norm = self.get_normalized_coordinates(self.coords_gradient)


class LHS(RandomGrid):
    """
    LHS grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    options: dict, optional, default=None
        Grid options:
        - criterion :
            - 'corr'            : optimizes design points in their spearman correlation coefficients
            - 'maximin' or 'm'  : optimizes design points in their maximum minimal distance using the Phi-P criterion
            - 'ese'             : uses an enhanced evolutionary algorithm to optimize the Phi-P criterion
        - seed : Seeding point to replicate random grids
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.LHS(parameters_random=parameters_random, n_grid=100, options={"seed": None, "criterion": "ese"})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
        - 'corr'            : optimizes design points in their spearman correlation coefficients
        - 'maximin' or 'm'  : optimizes design points in their maximum minimal distance using the Phi-P criterion
        - 'ese'             : uses an enhanced evolutionary algorithm to optimize the Phi-P criterion
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initializes RandomGrid instance; Generates grid or copies provided content
        """

        self.coords_norm_lhs = None
        self.perc_mask = None
        self.coords_reservoir = None
        self.coords_norm_reservoir = None
        self.grid_pre = grid_pre
        self.options = options
        self.criterion = None
        self.coords_norm_reservoir_perced = None

        if type(self.options) is dict:
            if "criterion" in self.options.keys():
                self.criterion = options["criterion"]
            else:
                self.criterion = ["ese"]

        if type(self.criterion) is not list:
            self.criterion = [self.criterion]

        super(LHS, self).__init__(parameters_random,
                                  n_grid=n_grid,
                                  options=options,
                                  coords=coords,
                                  coords_norm=coords_norm,
                                  coords_gradient=coords_gradient,
                                  coords_gradient_norm=coords_gradient_norm,
                                  coords_id=coords_id,
                                  coords_gradient_id=coords_gradient_id)

        self.shift_outer = False

        if self.criterion == ["ese"]:
            for p in parameters_random:
                if parameters_random[p].p_perc is None:
                    self.shift_outer = True

        if coords is not None and coords_norm is not None:
            grid_present = True
        else:
            grid_present = False

        if not grid_present:
            if self.n_grid > 0:
                self.sample_init(self.n_grid)

    def sample_init(self, n_grid):
        """
        Initialises all parameters for Latin Hypercube Sampling and creates a new design
        if there is at least one sampling point needed

        Parameters
        ----------
        n_grid : ndarray of float [n]
            The number of needed sampling points

        Returns
        -------
        coords : ndarray of float [n_grid_add x dim]
            Grid points to add (model space)
        coords_norm : ndarray of float [n_grid_add x dim]
            Grid points to add (normalized space)
        coords_id : list of UUID objects (version 4) [n_grid]
            Unique IDs of grid points
        """

        if n_grid > 0:
            if n_grid == 1:
                random_grid = Random(parameters_random=self.parameters_random,
                                     n_grid=self.n_grid,
                                     options=self.options)
                self.coords_norm = random_grid.coords_norm

            else:
                if self.grid_pre is None:
                    n_grid_lhs = self.n_grid
                else:
                    n_grid_lhs = self.n_grid + self.grid_pre.n_grid

                self.perc_mask = np.zeros((n_grid_lhs, self.dim)).astype(bool)
                self.coords_norm_reservoir = np.zeros([n_grid_lhs, self.dim])

                # generate LHS coordinates
                self.get_lhs_grid()

        # Denormalize grid to original parameter space
        self.coords = self.get_denormalized_coordinates(self.coords_norm)

        # Generate unique IDs of grid points
        self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    def CL2(self, array):
        """
        Calculate the L2 discrepancy of the design
        The discrepancy is a measure of the difference between the empirical cumulative distribution function
        of an experimental design and the uniform cumulative distribution function [1].

        Parameters
        ----------
        array : ndarray of float [m x n]
            Array with n rows of samples and m columns of variables/dimensions

        Returns
        -------
        cl2d_crit : float
            criterion for centered L2 discrepancy

        Notes
        -----
        .. [1] Hickernell, F. (1998). A generalized discrepancy and quadrature error bound.
           Mathematics of computation, 67(221), 299-322.
        """
        subtract = np.ones([np.shape(array)[0], np.shape(array)[1]])
        array = array - 0.5 * subtract
        prod_array_1 = np.zeros(np.shape(array)[0])
        for n in range(0, np.shape(array)[0]):
            prod_array_1[n] = (np.ones(np.shape(array)[1]) + (1 / 2 * (np.abs(array[n, :]))) - (
                    1 / 2 * ((array[n, :]) ** 2))).prod()
        prod_array_2 = np.zeros([np.shape(array)[0], np.shape(array)[0]])
        for i in range(0, np.shape(array)[0]):
            for j in range(0, np.shape(array)[0]):
                prod_array_2[i, j] = (np.ones(np.shape(array)[1]) + (1 / 2 * (np.abs(array[i, :]))) - (
                        1 / 2 * np.abs(array[j, :]) - 1 / 2 * np.abs(array[i, :] - array[j, :]))).prod()
        cl2d_crit = ((13 / 12) ** 2) - (2 / (np.shape(array)[0]) * prod_array_1.sum()) + (1 / (
                np.shape(array)[0] ** 2) * prod_array_2.sum())
        # centered L2 discrepancy criteria
        return cl2d_crit

    def log_R(self, array):
        """
        Determines the Log(R) Entropy Criterion[1]

        Parameters
        ----------
        array : ndarray of float [m x n]
            Array

        Returns
        -------
        log_R : float
            Log(R) Entropy Criterion
        Notes
        -----
        .. [1] Koehler, J.R., Owen, A.B., 1996. Computer experiments. in: Ghosh, S., Rao, C.R. (Eds.),
           Handbook of Statistics. Elsevier Science, New York, pp.261-308
        """
        # R will be [m x m]
        R = np.corrcoef(array.T)
        for i in range(0, np.shape(array)[1]):
            for j in range(0, np.shape(array)[1]):
                R[i, j] = np.exp((R[i, :] * np.abs(array[i, :] - array[j, :]) ** 2).sum())
        log_R = np.log(np.linalg.norm(R))

        return (log_R)

    def PhiP(self, x, p=10):
        """
        Calculates the Phi-p criterion of the design x with power p [1].

        Parameters
        ----------
        x : ndarray of float [n x m]
            The design to calculate Phi-p for
        p : int, optional, default: 10
            The power used for the calculation of PhiP

        Returns
        -------
        phip : float
            Phi-p criterion

        Notes
        -----
        .. [1] Morris, M. D., & Mitchell, T. J. (1995). Exploratory designs for computational experiments.
           Journal of statistical planning and inference, 43(3), 381-402.
        """

        phip = ((scipy.spatial.distance.pdist(x) ** (-p)).sum()) ** (1.0 / p)

        return phip

    def PhiP_exchange(self, P, k, Phi, p, fixed_index):
        """
        Performes a row exchange and return the altered design.

        Parameters
        ----------
        P : ndarray of float [m x n]
            The design to perform the exchange on
        k : int
            modulus of the iteration divided by the dimension to pick a row of the design repeating through the
            dimensions of the design
        Phi: float
            the PhiP criterion of the current best Design
        p: int
            The power used for the calculation of PhiP
        fixed_index: list
            an empty list to check if variables are assigned a value

        Returns
        -------
        phip : float
            Phi-p criterion
        """
        # Choose two (different) random rows to perform the exchange
        er = P.shape
        i1 = np.random.randint(P.shape[0])
        while i1 in fixed_index:
            i1 = np.random.randint(P.shape[0])

        i2 = np.random.randint(P.shape[0])
        while i2 == i1 or i2 in fixed_index:
            i2 = np.random.randint(P.shape[0])

        P_= np.delete(P, [i1, i2], axis=0)

        dist1 = scipy.spatial.distance.cdist([P[i1, :]], P_)
        dist2 = scipy.spatial.distance.cdist([P[i2, :]], P_)
        d1 = np.sqrt(dist1 ** 2 + (P[i2, k] - P_[:, k]) ** 2 - (P[i1, k] - P_[:, k]) ** 2)
        d2 = np.sqrt(dist2 ** 2 - (P[i2, k] - P_[:, k]) ** 2 + (P[i1, k] - P_[:, k]) ** 2)

        res = (Phi ** p + (d1 ** (-p) - dist1 ** (-p) + d2 ** (-p) - dist2 ** (-p)).sum()) ** (1.0 / p)

        P[i1, k], P[i2, k] = P[i2, k], P[i1, k]
        return res

    def get_lhs_grid(self):
        """
        Create samples in an m*n matrix using Latin Hypercube Sampling [1].

        Notes
        -----
        .. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). A comparison of three methods for selecting
           values of input variables in the analysis of output from a computer code. Technometrics, 42(1), 55-61.
        """

        # create sample points in icdf space using specified criteria
        if self.criterion[0] == 'corr':
            self.coords_norm_lhs = self.lhs_corr()
        elif self.criterion[0] == 'maximin' or self.criterion == 'm':
            self.coords_norm_lhs = self.lhs_maximin()
        elif self.criterion[0] == 'ese':
            self.coords_norm_lhs = self.lhs_ese()
        else:
            self.coords_norm_lhs = self.lhs_initial()

        # transform sample points from icdf to pdf space
        for i_p, p in enumerate(self.parameters_random):
            self.coords_norm_reservoir[:, i_p] = self.parameters_random[p].icdf(self.coords_norm_lhs[:, i_p])

        if self.grid_pre is not None:
            self.coords_norm_reservoir = get_different_rows_from_matrices(self.grid_pre.coords_norm,
                                                                          self.coords_norm_reservoir)

        self.coords_norm = self.coords_norm_reservoir[0:self.n_grid, :]

    def lhs_initial(self):
        """
        Construct an initial LHS grid.

        Returns
        -------
        design : ndarray of float [n, n_dim]
            LHS grid points
        """

        if self.grid_pre is not None:
            # transform normalized coordinates back to LHS-Space (0, 1)
            pre_coords_lhs = np.zeros(self.grid_pre.coords_norm.shape)

            for i, p in enumerate(self.parameters_random):
                pre_coords_lhs[:, i] = self.parameters_random[p].cdf_norm(self.grid_pre.coords_norm[:, i])

            return self.lhs_extend(pre_coords_lhs, self.n_grid)

        else:
            design = np.zeros([self.n_grid, self.dim])

            # u = matrix of uniform (0,1) that vary in n subareas
            u = np.random.rand(self.n_grid, self.dim)

            for i in range(0, self.dim):
                for j in range(0, self.n_grid):

                    if self.shift_outer:
                        if j == 0:
                            design[j, i] = j + 1
                            u[j, i] = 1 - 1/4 * u[j, i]
                        elif (j + 1) == self.n_grid:
                            design[j, i] = j + 1
                            u[j, i] = 1/4 * u[j, i]
                        elif j <= self.n_grid/2:
                            design[j, i] = j + 1 - ((j*3/4) / ((self.n_grid - 2) * self.n_grid))
                        elif j > self.n_grid/2:
                            design[j, i] = j + 1 + ((j*3/4) / ((self.n_grid - 2) * self.n_grid))
                    else:
                        design[j, i] = j + 1

            for i in range(0, self.dim):
                for j in range(0, self.n_grid):
                    design[j, i] = (design[j, i] - u[j, i]) / self.n_grid

                np.random.shuffle(design[:, i])

            return design

    def lhs_corr(self):
        """
        Create a correlation optimized LHS grid

        Parameters
        ----------
        dim : int
            Number of random variables
        n : int
            Number of sampling points
        iterations : int
            Number of iterations to optimize

        Returns
        -------
        design : ndarray of float [n, n_dim]
            LHS grid points
        """
        mincorr = np.inf

        # Minimize the components correlation coefficients
        for i in range(100):
            # Generate a random LHS
            test = self.lhs_initial()
            R = scipy.stats.spearmanr(test)[0]

            if np.max(np.abs(R)) < mincorr:
                mincorr = np.max(np.abs(R))
                design = test.copy()

        return design

    def lhs_maximin(self):
        """
        Create an optimized LHS grid with maximal minimal distance

        Parameters
        ----------
        dim : int
            Number of random variables
        n : int
            Number of sampling points
        iterations : int
            Number of iterations to optimize

        Returns
        -------
        design : ndarray of float [n, n_dim]
            LHS grid points
        """
        phi_best = max(1000, self.n_grid * 100)

        # Maximize the minimum distance between points
        for i in range(100):
            test = self.lhs_initial()
            phi = self.PhiP(test)
            if phi_best > phi:
                phi_best = phi
                design = test.copy()

        return design

    def lhs_ese(self):
        """
        Create optimized LHS grid using a enhanced stochastic evolutionary algorithm for the PhiP Maximin criterion [1]

        Returns
        -------
        design : ndarray of float [n, n_dim]
            With ESE Algorithm for minimal Phi-P optimized grid points

        Notes
        -----
        .. [1] Jin, R., Chen, W., & Sudjianto, A. (2005). An efficient algorithm for constructing optimal
           design of computer experiments. Journal of statistical planning and inference, 134(1), 268-287.
        """

        # Parameters
        t0 = None
        P0 = self.lhs_initial()
        J = 25
        tol = 1e-3
        p = 10
        outer_loop = min(int(1.5 * self.dim), 30)
        inner_loop = min(20 * self.dim, 100)

        if self.coords_norm_reservoir_perced is not None:
            fixed_index = [*range(self.coords_norm_reservoir_perced.shape[0])]
        elif self.grid_pre is not None:
            fixed_index = [*range(self.grid_pre.coords_norm.shape[0])]
        else:
            fixed_index = []

        if t0 is None:
            t0 = 0.005 * self.PhiP(P0, p=p)

        T = t0
        P_ = P0[:]  # copy of initial design
        P_best = P_[:]
        Phi = self.PhiP(P_best, p=p)
        Phi_best = Phi

        # Outer loop
        for z in range(outer_loop):
            Phi_oldbest = Phi_best
            n_acpt = 0
            n_imp = 0

            # Inner loop
            for i in range(inner_loop):
                modulo = (i + 1) % self.dim
                l_P = list()
                l_Phi = list()

                # Build J different designs with a single exchanged row
                # See PhiP_exchange
                for j in range(J):
                    l_P.append(P_.copy())
                    l_Phi.append(self.PhiP_exchange(l_P[j], k=modulo, Phi=Phi, p=p, fixed_index=fixed_index))

                l_Phi = np.asarray(l_Phi)
                k = np.argmin(l_Phi)
                Phi_try = l_Phi[k]

                # Threshold of acceptance
                if Phi_try - Phi <= T * np.random.rand(1)[0]:
                    Phi = Phi_try
                    n_acpt = n_acpt + 1
                    P_ = l_P[k]

                    # Best design retained
                    if Phi < Phi_best:
                        P_best = P_
                        Phi_best = Phi
                        n_imp = n_imp + 1

            p_accpt = float(n_acpt) / inner_loop  # probability of acceptance
            p_imp = float(n_imp) / inner_loop  # probability of improvement

            if Phi_best - Phi_oldbest < tol:
                # flag_imp = 1
                if p_accpt >= 0.1 and p_imp < p_accpt:
                    T = 0.8 * T
                elif p_accpt >= 0.1 and p_imp == p_accpt:
                    pass
                else:
                    T = T / 0.8
            else:
                # flag_imp = 0
                if p_accpt <= 0.1:
                    T = T / 0.7
                else:
                    T = 0.9 * T

        return P_best


class CO(RandomGrid):
    """
    Coherence Optimal grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    options: dict, optional, default=None
        Grid options:
        - 'seed'            : Seeding point
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.CO(parameters_random=parameters_random, n_grid=100, options={"seed": None})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, gpc, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initializes CO instance; Generates grid or copies provided content
        """
        if options is None:
            options = dict()

        if "seed" not in options.keys():
            options["seed"] = None

        if "n_warmup" not in options.keys():
            options["n_warmup"] = max(200, n_grid*2)
        else:
            options["n_warmup"] = max(options["n_warmup"], n_grid * 2)

        if "n_pool" not in options.keys():
            if 1000 > 2 * n_grid:
                options["n_pool"] = 1000
            else:
                options["n_pool"] = 2*n_grid

        self.gpc = gpc
        self.grid_pre = grid_pre
        self.coords_pool = []
        self.gpc_matrix_pool = []
        self.b2_pool = []
        self.g_pool = []
        self.f_pool = []
        self.n_pool = options["n_pool"]
        self.all_norm = []

        super(CO, self).__init__(parameters_random,
                                 n_grid=n_grid,
                                 options=options,
                                 coords=coords,
                                 coords_norm=coords_norm,
                                 coords_gradient=coords_gradient,
                                 coords_gradient_norm=coords_gradient_norm,
                                 coords_id=coords_id,
                                 coords_gradient_id=coords_gradient_id,
                                 grid_pre=grid_pre)
        if n_grid == 0:
            pass
        else:
            self.ball_volume = self.calc_ball_volume(dim=self.dim, radius=np.sqrt(2) * np.sqrt(2*self.gpc.order_max+1))
            pdf_type = [self.parameters_random[rv].pdf_type for rv in self.parameters_random]
            self.all_norm = np.array([p == "norm" for p in pdf_type]).all()
            any_norm = np.array([True for p in pdf_type if p == "norm"]).any() and not self.all_norm

            if any_norm:
                raise AssertionError("Mixed distributions of beta and normal for CO grids not implemented..."
                                     "All variables have to be either normal or beta distributed!")

            # create proposal distributed random variables
            self.parameters_random_proposal = dict()
            for rv in self.parameters_random:
                # check the order > dimension criteria
                if gpc.order_max > self.dim:

                    # uniform distributed random variables -> Chebyshev distribution
                    if self.parameters_random[rv].pdf_type == "beta" and \
                            (self.parameters_random[rv].pdf_shape == [1, 1]).all():
                        self.parameters_random_proposal[rv] = Beta(pdf_shape=[0.5, 0.5],
                                                                   pdf_limits=[-1, 1])

                    # normal distributed random variables -> sample uniformly from the d-dimensional ball of radius r
                    # here a standard normal distribution is created, the uniform sampling from the ball is considered in
                    # the method "create_pool"
                    elif self.parameters_random[rv].pdf_type == "norm":
                        self.parameters_random_proposal[rv] = Norm(pdf_shape=[0, 1],
                                                                   p_perc=self.parameters_random[rv].p_perc)

                    else:
                        NotImplementedError("Coherence optimal sampling is only possible for uniform and normal "
                                            "distributed random variables")
                else:
                    self.parameters_random_proposal[rv] = self.parameters_random[rv]
            #define number of warmup-samples
            self.n_warmup = options["n_warmup"]

            # # draw sample pool for warmup
            # self.create_pool(n_samples=2*options["n_warmup"])
            #
            # # warmup
            # self.warmup(n_warmup=options["n_warmup"])

            # draw sample pool for actual sampling
            self.create_pool(n_samples=self.n_pool + self.n_warmup)

            # get coherence optimal samples
            self.coords_norm = self.get_coherence_optimal_samples(n_grid=self.n_grid, n_warmup=self.n_warmup)

            # Denormalize grid to original parameter space
            self.coords = self.get_denormalized_coordinates(self.coords_norm)

    def create_pool(self, n_samples):
        """
        Creates a pool of samples together with the corresponding gPC matrix.

        Parameters
        ----------
        n_samples : int
            Number of samples
        """

        self.n_pool = n_samples
        self.coords_pool = np.zeros((n_samples, self.dim))

        for i_rv, rv in enumerate(self.parameters_random_proposal):
            self.coords_pool[:, i_rv] = self.parameters_random_proposal[rv].sample(n_samples=int(self.n_pool))

        if self.all_norm:
            # sample from d-dimensional ball of radius r (Hampton et al. 2015, pp. 369)
            r = np.sqrt(2)*np.sqrt(2*self.gpc.order_max+1)
            self.coords_pool = self.coords_pool / (np.linalg.norm(self.coords_pool, axis=1))[:, np.newaxis] * \
                               r * np.random.rand(1) ** (1/self.dim)

        self.gpc_matrix_pool = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=self.coords_pool)

        self.b2_pool = np.linalg.norm(self.gpc_matrix_pool, axis=1)**2
        self.g_pool = self.joint_pdf(x=self.coords_pool, parameters_random=self.parameters_random_proposal)
        self.f_pool = self.joint_pdf(x=self.coords_pool, parameters_random=self.parameters_random)

    @staticmethod
    def calc_ball_volume(dim, radius):
        """
        Volume of n-dimensional ball.

        Parameters
        ----------
        dim : int
            Number of random variables
        radius : float
            Radius

        Returns
        -------
        vol : float
            Volume of n-dimensional ball
        """
        vol = radius**dim * np.pi**(dim/2) / gamma(dim/2 + 1)

        return vol

    def joint_pdf(self, x, parameters_random):
        """
        Joint probability density function of random variables

        Parameters
        ----------
        x : ndarray of float [n_samples x n_dim]
            Samples of random variables
        parameters_random : dict of RandomVariable instances
            Random variables of proposal distribution

        Returns
        -------
        f : float
            Joint probability density
        """
        f = np.ones(x.shape[0])

        if np.array([True for p in parameters_random if parameters_random[p].pdf_type == "norm"]).all():
            f *= 1/self.ball_volume
        else:
            for i_rv, rv in enumerate(parameters_random):
                f *= parameters_random[rv].pdf_norm(x=x[:, i_rv])[1]

        return f

    def acceptance_rate(self, idx1, idx2):
        """
        Calculate acceptance rate of Metropolis Hastings algorithm

        Parameters
        ----------
        idx1 : int
            Index of sampling points of previous sample
        idx2 : int
            Index of sampling points of current sample

        Returns
        -------
        rho : float
            Acceptance rate
        """

        rho = np.min((1.,
                     (self.g_pool[idx1]*self.f_pool[idx2]*self.b2_pool[idx2]) /
                     (self.g_pool[idx2]*self.f_pool[idx1]*self.b2_pool[idx1])))

        return rho

    def warmup(self, n_warmup):
        """
        Warmup phase

        Parameters
        ----------
        n_warmup : int
            Number of warmup samples

        Returns
        -------
        coords_norm_opt : ndarray of float [n_warmup x n_dim]
            Optimal coordinates determined during warmup phase
        """

        coords_norm_opt = self.get_coherence_optimal_samples(n_grid=n_warmup)

        return coords_norm_opt

    def get_coherence_optimal_samples(self, n_grid, n_warmup):
        """
        Determine coherence optimal samples with Monte Carlo Markov Chain - Metropolis Hastings algorithm

        Parameters
        ----------
        n_grid : int
            Number of grid points
        n_warmup: int
            Number of warmup samples

        Returns
        -------
        coords_norm : ndarray of float [n_grid x n_dim]
            Coherence optimal samples
        """
        coords_norm_opt = np.zeros((n_grid, self.dim))
        coords_norm_opt[0, :] = self.coords_pool[0, :][np.newaxis, :]

        if self.grid_pre is not None:
            coords_norm_opt[:self.grid_pre.n_grid, :] = self.grid_pre.coords_norm
            i_grid_start = self.grid_pre.n_grid
        else:
            i_grid_start = 0

        # init grid_index at -n_warmup, then the number of warmup-samples are automatically drawn,
        # before index 0 is reached
        i_grid = -(n_warmup - i_grid_start)
        idx1 = 0
        idx2 = 1

        if self.all_norm:
            x_perc_norm = np.zeros(len(self.parameters_random))
            for i_rv, rv in enumerate(self.parameters_random):
                x_perc_norm[i_rv] = self.parameters_random[rv].x_perc_norm[1]

        while i_grid < n_grid:
            # create a new pool if it is empty
            if idx2 >= self.n_pool:
                self.create_pool(2*n_grid)
                idx1 = 0
                idx2 = 1

            # determine acceptance rate
            rho = self.acceptance_rate(idx1=idx1, idx2=idx2)

            # add point if acceptance rate is high
            if rho > np.random.rand(1):
                    if self.all_norm:
                        if (np.abs(self.coords_pool[idx2, :]) < x_perc_norm).all():
                            if i_grid > (i_grid_start-1):
                                coords_norm_opt[i_grid, :] = self.coords_pool[idx2, :]
                            i_grid += 1
                            idx1 = idx2
                    else:
                        if i_grid > (i_grid_start-1):
                            coords_norm_opt[i_grid, :] = self.coords_pool[idx2, :]
                        i_grid += 1
                        idx1 = idx2

            idx2 += 1

        return coords_norm_opt


class L1(RandomGrid):
    """
    L1 optimized grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    options: dict, optional, default=None
        Grid options:
        - method: "greedy", "iteration"
        - criterion: ["mc"], ["tmc", "cc"], ["D"], ["D-coh"]
        - weights: [1], [0.5, 0.5], [1]
        - n_pool: size of samples in pool to choose greedy results from
        - n_iter: number of iterations
        - seed: random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    gpc : GPC object instance
        GPC object
    grid_pre : Grid object instance, optional, default: None
        Existent grid, which will be extended.

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.L1(parameters_random=parameters_random,
    >>>                 n_grid=100,
    >>>                 options={"method": "greedy",
    >>>                          "criterion": ["mc"],
    >>>                          "weights": [1],
    >>>                          "n_pool": 1000,
    >>>                          "seed": None})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
        - method: "greedy", "iteration"
        - criterion: ["mc"], ["tmc", "cc"], ["D"], ["D-coh"]
        - weights: [1], [0.5, 0.5], [1]
        - n_pool: size of samples in pool to choose greedy results from
        - n_iter: number of iterations
        - seed: random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    gpc : GPC Object instance
        GPC object
    grid_pre : Grid object instance, optional, default: None
        Existent grid, which will be extended.
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None, gpc=None,
                 grid_pre=None):
        """
        Constructor; Initializes Grid instance; Generates grid or copies provided content
        """
        if options is None:
            options = dict()

        if type(options) is dict:
            if "method" not in options.keys():
                options["method"] = "greedy"

            if "n_pool" not in options.keys():
                options["n_pool"] = 10000

            if "n_iter" not in options.keys():
                options["n_iter"] = 1000

            if "seed" not in options.keys():
                options["seed"] = None

            if "criterion" not in options.keys():
                options["criterion"] = ["mc"]

            if "weights" not in options.keys() or options["weights"] is None:
                options["weights"] = (np.ones(len(options["criterion"])) / len(options["criterion"])).tolist()

        self.n_pool = options["n_pool"]
        self.n_iter = options["n_iter"]
        self.gpc = gpc
        self.seed = options["seed"]
        self.method = options["method"]
        self.criterion = options["criterion"]
        self.coords_norm_perced = None
        self.perc_mask = None

        if type(self.criterion) is not list:
            self.criterion = [self.criterion]

        super(L1, self).__init__(parameters_random,
                                 n_grid=n_grid,
                                 options=options,
                                 coords=coords,
                                 coords_norm=coords_norm,
                                 coords_gradient=coords_gradient,
                                 coords_gradient_norm=coords_gradient_norm,
                                 coords_id=coords_id,
                                 coords_gradient_id=coords_gradient_id,
                                 grid_pre=grid_pre)

        self.weights = options["weights"]

        if self.method == "greedy" and self.n_grid > 0 and self.coords is None:
            self.coords_norm = self.get_optimal_mu_greedy()

        elif (self.method == 'iteration' or self.method == 'iter') and self.n_grid > 0 and self.coords is None:
            self.coords_norm = self.get_optimal_mu_iteration()

        if self.n_grid > 0:
            # Denormalize grid to original parameter space
            self.coords = self.get_denormalized_coordinates(self.coords_norm)

            # Generate unique IDs of grid points
            self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    def get_optimal_mu_greedy(self):
        """
        This function computes a set of grid points with minimal mutual coherence using a greedy approach.

        Returns
        -------
        coords_norm : ndarray of float [n_grid x dim]
            Normalized sample coordinates in range [-1, 1]
        """
        n_cpu = np.min((1, multiprocessing.cpu_count()))

        # create pool (Standard random grid for D-optimal grids and CO else)
        if "D" in self.criterion:
            random_pool = Random(parameters_random=self.parameters_random,
                                 n_grid=self.n_pool,
                                 options=self.options)
        else:
            random_pool = CO(parameters_random=self.parameters_random,
                             n_grid=self.n_pool,
                             gpc=self.gpc,
                             options=self.options)
            # full gpc matrix is needed for w_matrix, maybe build a function that creates weighted pools
            # based of the coordinates
            # self.gpc.get_weight_matrix()
            # w_matrix = self.gpc.w
        index_list = []

        # project grid in case of projection approach
        if self.gpc.p_matrix is not None:
            # weight argument in create gpc matrix
            random_pool_trans = project_grid(grid=random_pool, p_matrix=self.gpc.p_matrix, mode="reduce")
            psy_pool = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=random_pool_trans.coords_norm, gradient=False,
                                                  weighted=True)
        else:
            psy_pool = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=random_pool.coords_norm, gradient=False,
                                                  weighted=True)

        m = int(self.n_grid)
        m_p = int(np.shape(psy_pool)[0])

        # set up multiprocessing
        pool = multiprocessing.Pool(n_cpu)

        # set starting point for iteration
        if self.grid_pre is None or self.grid_pre.n_grid == 0:
            # get random row of psy to start
            idx = np.random.randint(m_p)
            index_list.append(idx)
            index_list_remaining = [k for k in range(self.n_pool) if k not in index_list]
            psy_opt = np.zeros((1, psy_pool.shape[1]))
            psy_opt[0, :] = psy_pool[idx, :]
            i_start = 1

        else:
            # project grid in case of projection approach
            if self.gpc.p_matrix is not None:
                grid_pre_trans = project_grid(grid=self.grid_pre, p_matrix=self.gpc.p_matrix, mode="reduce")
                psy_opt = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=grid_pre_trans.coords_norm, gradient=False,
                                                     weighted=True)
            else:
                psy_opt = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=self.grid_pre.coords_norm, gradient=False,
                                                     weighted=True)

            index_list = []
            index_list_remaining = [k for k in range(self.n_pool) if k not in index_list]
            i_start = self.grid_pre.n_grid

        # loop over grid points
        for i in range(i_start, m):
            crit = np.ones((self.n_pool, len(self.criterion))) * 1e6

            workhorse_partial = partial(workhorse_greedy, psy_opt=psy_opt, psy_pool=psy_pool, criterion=self.criterion)
            idx_list_chunks = compute_chunks(index_list_remaining, n_cpu)

            crit_tmp = pool.map(workhorse_partial, idx_list_chunks)

            if "D" not in self.criterion and "D-coh" not in self.criterion:
                crit_tmp = np.concatenate(crit_tmp)

            else:
                sign = []
                neg_logdet = []

                for res in crit_tmp:
                    sign.append(res[0])
                    neg_logdet.append(res[1])

                sign = np.concatenate(sign)
                neg_logdet = np.concatenate(neg_logdet)
                neg_logdet_norm = neg_logdet / np.nan_to_num(np.max(np.abs(neg_logdet)))
                crit_tmp = sign * np.nan_to_num(np.exp(neg_logdet_norm))

            crit[index_list_remaining, :] = crit_tmp

            # set 1e6 dummy values to max values
            if "D" not in self.criterion and "D-coh" not in self.criterion:
                for k in range(crit.shape[1]):
                    crit[crit[:, k] == 1e6, k] = np.max(crit[crit[:, k] != 1e6, k])

            # normalize optimality criteria to [0, 1]
            crit = np.nan_to_num(crit)
            crit = (crit - np.nanmin(crit, axis=0)) / np.nan_to_num((np.nanmax(crit, axis=0) - np.nanmin(crit, axis=0)))

            # apply weights
            crit = np.sum(crit**2 * np.array(self.weights), axis=1)

            # find best index
            try:
                index_list.append(np.nanargmin(crit))
            # in very rare cases there the optimal grid point can not be determined (all nan), in this case the first
            # grid point of the remaining indices is chosen
            except ValueError:
                index_list.append(index_list_remaining[0])

            # add row with best minimal coherence and cross correlation properties to the matrix
            psy_opt = np.vstack((psy_opt, psy_pool[index_list[-1], :]))

            # create list of remaining indices
            index_list_remaining = [k for k in range(self.n_pool) if k not in index_list]

        coords_norm = random_pool.coords_norm[index_list, :]

        pool.close()
        pool.join()

        if self.grid_pre is not None:
            coords_norm = np.vstack((self.grid_pre.coords_norm, coords_norm))

        return coords_norm

    def get_optimal_mu_iteration(self):
        """
        This function computes a set of grid points with minimal mutual coherence using an iterative approach.

        Returns
        -------
        coords_norm : ndarray of float [n_grid x dim]
            Normalized sample coordinates in range [-1, 1]
        """
        n_cpu = np.min((1, multiprocessing.cpu_count()))
        coords_norm_list = []
        crit = np.ones((self.n_iter, len(self.criterion))) * 1e6

        # set up multiprocessing
        pool = multiprocessing.Pool(n_cpu)
        workhorse_partial = partial(workhorse_iteration,
                                    gpc=self.gpc,
                                    n_grid=self.n_grid,
                                    criterion=self.criterion,
                                    grid_pre=self.grid_pre,
                                    options={"seed": self.seed})
        idx_list_chunks = compute_chunks([k for k in range(self.n_iter)], n_cpu)

        res = pool.map(workhorse_partial, idx_list_chunks)

        for j in range(len(res)):
            if j == 0:
                if "D" not in self.criterion and "D-coh" not in self.criterion:
                    crit = res[j][0]
                    coords_norm_list = res[j][1]
                else:
                    sign = res[j][0]
                    neg_logdet = res[j][1]
                    neg_logdet_norm = neg_logdet / np.max(np.abs(neg_logdet))
                    crit = sign * np.exp(neg_logdet_norm)
                    coords_norm_list = res[j][2]
            else:
                if "D" not in self.criterion and "D-coh" not in self.criterion:
                    crit = np.vstack((crit, res[j][0]))
                    coords_norm_list = coords_norm_list + res[j][1]
                else:
                    sign = res[j][0]
                    neg_logdet = res[j][1]
                    neg_logdet_norm = neg_logdet / np.max(np.abs(neg_logdet))
                    crit = np.vstack((crit, sign * np.exp(neg_logdet_norm)))
                    coords_norm_list = coords_norm_list + res[j][2]

        # normalize optimality criteria to [0, 1]
        crit = (crit - np.min(crit, axis=0)) / np.nan_to_num((np.max(crit, axis=0) - np.min(crit, axis=0)))

        # apply weights
        crit = np.sum(crit**2 * np.array(self.weights), axis=1)

        coords_norm = coords_norm_list[np.argmin(crit)]

        pool.close()

        return coords_norm


class FIM(RandomGrid):
    """
    FIM D-optimal grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    seed: float
        Seeding point to replicate random grids
    options: dict, optional, default=None
        Grid options:
        - 'n_pool'   : number of random samples to determine next optimal grid point
        - 'seed'     : random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.FIM(parameters_random=parameters_random,
    >>>                  n_grid=100,
    >>>                  options={"n_pool": 1000,
    >>>                           "seed": None})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
        - method: "greedy", "iteration"
        - criterion: ["mc"], ["tmc", "cc"]
        - weights: [1], [0.5, 0.5]
        - n_pool: size of samples in pool to choose greedy results from
        - n_iter: number of iterations
        - seed: random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None, gpc=None,
                 grid_pre=None):
        """
        Constructor; Initializes Grid instance; Generates grid or copies provided content
        """
        if options is None:
            options = dict()

        if type(options) is dict:

            if "n_pool" not in options.keys():
                options["n_pool"] = 1000

            if "seed" not in options.keys():
                options["seed"] = None

        self.n_pool = options["n_pool"]
        self.gpc = copy.deepcopy(gpc)

        super(FIM, self).__init__(parameters_random,
                                  n_grid=n_grid,
                                  options=options,
                                  coords=coords,
                                  coords_norm=coords_norm,
                                  coords_gradient=coords_gradient,
                                  coords_gradient_norm=coords_gradient_norm,
                                  coords_id=coords_id,
                                  coords_gradient_id=coords_gradient_id,
                                  grid_pre=grid_pre)

        if coords_norm is not None:
            self.gpc.grid = Random(parameters_random=parameters_random,
                                   coords_norm=coords_norm,
                                   coords=coords,
                                   options=self.options)
            n_grid_add = self.gpc.grid.n_grid

            if self.gpc.p_matrix is not None:
                self.gpc.gpc_matrix = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                 x=np.matmul(coords_norm,
                                                                             self.gpc.p_matrix.transpose() /
                                                                             self.gpc.p_matrix_norm[np.newaxis, :]),
                                                                 gradient=False)
            else:
                self.gpc.gpc_matrix = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                 x=coords_norm,
                                                                 gradient=False)
        elif self.grid_pre is not None:
            self.gpc.grid = self.grid_pre
            n_grid_add = self.n_grid - self.grid_pre.n_grid

            if n_grid_add < 0:
                raise RuntimeError(f"Number of grid points to add has to be >= 0 (it is {n_grid_add}")

            if self.gpc.p_matrix is not None:
                self.gpc.gpc_matrix = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                 x=np.matmul(self.grid_pre.coords_norm,
                                                                             self.gpc.p_matrix.transpose() /
                                                                             self.gpc.p_matrix_norm[np.newaxis, :]),
                                                                 gradient=False)
            else:
                self.gpc.gpc_matrix = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                 x=self.grid_pre.coords_norm,
                                                                 gradient=False)
        else:
            n_grid_add = self.n_grid

        # add FIM optimal grid points (eventually to existing grid)
        self.coords_norm = self.add_fim_optiomal_grid_points(parameters_random=parameters_random,
                                                             n_grid_add=n_grid_add)

        # Denormalize grid to original parameter space
        self.coords = self.get_denormalized_coordinates(self.coords_norm)

        # Generate unique IDs of grid points
        self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]

    def add_fim_optiomal_grid_points(self, parameters_random, n_grid_add):
        """
        This function adds grid points (one by one) to the set of points by maximizing the Fisher-information matrix
        in a D-optimal sense.

        Parameters
        ----------
        parameters_random : OrderedDict of RandomParameter [dim]
            Random parameters (in case of projection, provide the original random parameters)
        n_grid_add : int
            Number of grid points to add

        Returns
        -------
        coords_norm : ndarray of float [n_grid x dim]
            Normalized sample coordinates in range [-1, 1]
        """

        # coords_norm_opt = np.zeros((n_grid_add, self.dim))
        #
        # for i in range(n_grid_add):
        #     fim_matrix = self.calc_fim_matrix()
        #     grid_test = Random(parameters_random=self.gpc.problem.parameters_random,
        #                        n_grid=self.n_pool,
        #                        options=self.options)
        #
        #     det = np.zeros(grid_test.coords_norm.shape[0])
        #
        #     for i_c, c in enumerate(grid_test.coords_norm):
        #         det[i_c] = self.get_det_updated_fim_matrix(fim_matrix=fim_matrix, coords_norm=c)
        #
        #     coords_norm_opt[i, :] = grid_test.coords_norm[np.argmax(det), :]
        #
        # return coords_norm_opt

        coords_norm_opt = np.zeros((n_grid_add, self.dim))
        n_cpu = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_cpu)

        grid_pool = Random(parameters_random=parameters_random,
                           n_grid=self.n_pool,
                           options={"seed": self.seed})

        if self.gpc.p_matrix is not None:
            gpc_matrix_pool = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                         x=np.matmul(self.grid_pool.coords_norm,
                                                                     self.gpc.p_matrix.transpose() /
                                                                     self.gpc.p_matrix_norm[np.newaxis, :]),
                                                         gradient=False)
        else:
            gpc_matrix_pool = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                         x=grid_pool.coords_norm,
                                                         gradient=False)

        if self.gpc.gpc_matrix is not None:
            fim_matrix = self.calc_fim_matrix()
        else:
            fim_matrix = None

        index_list = []

        for i in range(n_grid_add):
            det = np.zeros((self.n_pool))

            if self.seed is not None:
                self.seed += 1
                self.options["seed"] += 1

            # select random starting point
            if self.gpc.gpc_matrix is None:
                coords_opt = grid_pool.coords_norm[0, :][np.newaxis, ]
                self.gpc.grid = Random(parameters_random=parameters_random,
                                       options={"seed": self.seed},
                                       coords_norm=coords_opt)

                self.gpc.gpc_matrix = self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                 x=self.gpc.grid.coords_norm[-1, :][np.newaxis, ],
                                                                 gradient=False)

                index_list.append(0)

            else:
                index_list_remaining = [k for k in range(self.n_pool) if k not in index_list]
                index_list_chunks = compute_chunks(index_list_remaining, n_cpu)

                n_basis_limit = np.min((self.gpc.grid.n_grid, self.gpc.basis.n_basis))
                workhorse_partial = partial(workhorse_get_det_updated_fim_matrix,
                                            gpc_matrix_pool=gpc_matrix_pool,
                                            fim_matrix=fim_matrix,
                                            n_basis_limit=n_basis_limit)

                res = pool.map(workhorse_partial, index_list_chunks)

                sign = []
                logdet = []

                for r in res:
                    sign.append(r[0])
                    logdet.append(r[1])

                sign = np.concatenate(sign)
                logdet = np.concatenate(logdet)

                logdet_norm = logdet / np.max(np.abs(logdet))
                det[index_list_remaining] = (sign * np.exp(logdet_norm)).flatten()
                index_list.append(np.nanargmax(det))

                coords_opt = grid_pool.coords_norm[index_list[-1], :]

                # add optimal grid point
                self.gpc.grid.coords_norm = np.vstack((self.gpc.grid.coords_norm, coords_opt))

                # update gpc matrix
                self.gpc.gpc_matrix = np.vstack((self.gpc.gpc_matrix,
                                                 self.gpc.create_gpc_matrix(b=self.gpc.basis.b,
                                                                            x=self.gpc.grid.coords_norm[-1, :][np.newaxis, ],
                                                                            gradient=False)))

            # update FIM matrix
            n_basis_limit = np.min((self.gpc.grid.n_grid, self.gpc.basis.n_basis))

            if n_basis_limit == (self.gpc.basis.n_basis+1):
                fim_matrix = self.update_fim_matrix(fim_matrix=fim_matrix,
                                                    gpc_matrix_new_rows=self.gpc.gpc_matrix[-1, :][np.newaxis, ])
            else:
                fim_matrix = self.calc_fim_matrix(n_basis_limit=n_basis_limit)

        pool.close()

        return self.gpc.grid.coords_norm

    def calc_fim_matrix(self, n_basis_limit=None):
        """
        Calculates Fisher-Information matrix based on the present grid.

        Parameters
        ----------
        n_basis_limit : int
            Index of column the FIM matrix is calculated

        Returns
        -------
        fim_matrix : ndarray of float [n_basis x n_basis]
            Fisher information matrix
        """
        if n_basis_limit is None:
            n_basis_limit = self.gpc.gpc_matrix.shape[1]

        fim_matrix = np.zeros((n_basis_limit, n_basis_limit))

        for row in self.gpc.gpc_matrix:
            fim_matrix += np.outer(row[:n_basis_limit], row[:n_basis_limit])

        return fim_matrix

    @staticmethod
    def update_fim_matrix(fim_matrix, gpc_matrix_new_rows):
        """
        Updates Fisher-Information matrix based on the present grid.

        Parameters
        ----------
        fim_matrix : ndarray of float [n_basis x n_basis]
            Fisher information matrix
        gpc_matrix_new_rows : ndarray of float [n_new_rows x n_basis]
            New rows of gpc matrix to add to FIM matrix

        Returns
        -------
        fim_matrix : ndarray of float [n_basis x n_basis]
            Updated Fisher information matrix
        """
        if fim_matrix is None:
            fim_matrix = np.zeros((gpc_matrix_new_rows.shape[1], gpc_matrix_new_rows.shape[1]))

        for row in gpc_matrix_new_rows:
            fim_matrix += np.outer(row, row)

        return fim_matrix

    def get_det_updated_fim_matrix(self, fim_matrix, coords_norm):
        """
        Calculates Fisher-Information matrix based on the present grid and determined determinant.

        Parameters
        ----------
        fim_matrix : ndarray of float [n_basis x n_basis]
            Fisher information matrix
        coords_norm : ndarray of float [1 x dim]
            Candidate grid point

        Returns
        -------
        det : float
            Determinant of updated Fisher Information matrix
        """
        new_row = self.gpc.create_gpc_matrix(b=self.gpc.basis.b, x=coords_norm, gradient=False)
        fim_matrix += np.outer(new_row, new_row)

        return np.linalg.det(fim_matrix)


class L1_LHS(RandomGrid):
    """
    L1-LHS optimized grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    seed: float
        Seeding point to replicate random grids
    options: dict, optional, default=None
        Grid options:
        - 'corr'            : optimizes design points in their spearman correlation coefficients
        - 'maximin' or 'm'  : optimizes design points in their maximum minimal distance using the Phi-P criterion
        - 'ese'             : uses an enhanced evolutionary algorithm to optimize the Phi-P criterion
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.L1_LHS(parameters_random=parameters_random,
    >>>                     n_grid=100,
    >>>                     gpc=gpc,
    >>>                     options={"method": "greedy",
    >>>                              "criterion": ["mc"],
    >>>                              "weights_L1": [1],
    >>>                              "weights": [0.25, 0.75],
    >>>                              "n_pool": 1000,
    >>>                              "seed": None})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
        - method: "greedy", "iteration"
        - criterion: ["mc"], ["tmc", "cc"]
        - weights: [1], [0.5, 0.5]
        - n_pool: size of samples in pool to choose greedy results from
        - n_iter: number of iterations
        - seed: random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space), if coords are provided, no grid is generated
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space), if coords are provided, no grid is generated
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None, gpc=None,
                 grid_pre=None):
        """
        Constructor; Initializes Grid instance; Generates grid or copies provided content
        """
        if options is None:
            options = dict()

        if type(options) is dict:
            if "weights" not in options.keys():
                options["weights"] = [0.5, 0.5]

            if "method" not in options.keys():
                options["method"] = "iteration"

            if "n_pool" not in options.keys():
                options["n_pool"] = 10000

            if "n_iter" not in options.keys():
                options["n_iter"] = 1000

            if "seed" not in options.keys():
                options["seed"] = None

            if "criterion" not in options.keys():
                options["criterion"] = ["mc"]

            if "weights_L1" not in options.keys() or options["weights_L1"] is None:
                options["weights_L1"] = (np.ones(len(options["criterion"])) / len(options["criterion"])).tolist()

        self.n_pool = options["n_pool"]
        self.n_iter = options["n_iter"]
        self.gpc = gpc
        self.seed = options["seed"]
        self.method = options["method"]
        self.criterion = options["criterion"]
        self.weights_L1 = options["weights_L1"]
        self.grid_L1 = None
        self.grid_LHS = None
        self.grid_pre = grid_pre

        if type(self.criterion) is not list:
            self.criterion = [self.criterion]

        super(L1_LHS, self).__init__(parameters_random,
                                     n_grid=n_grid,
                                     options=options,
                                     coords=coords,
                                     coords_norm=coords_norm,
                                     coords_gradient=coords_gradient,
                                     coords_gradient_norm=coords_gradient_norm,
                                     coords_id=coords_id,
                                     coords_gradient_id=coords_gradient_id)

        self.weights = options["weights"]

        if coords_norm is None:
            self.n_grid_L1 = int(np.round(self.n_grid * self.weights[0]))
            self.n_grid_LHS = self.n_grid - self.n_grid_L1
        else:
            self.n_grid_L1 = None
            self.n_grid_LHS = None

        # create L1 grid
        if coords_norm is None and self.n_grid_L1 > 0:
            self.grid_L1 = L1(parameters_random=parameters_random,
                              n_grid=self.n_grid_L1,
                              gpc=gpc,
                              grid_pre=grid_pre,
                              options={"method": self.method,
                                       "criterion": self.criterion,
                                       "weights": self.weights_L1,
                                       "n_pool": self.n_pool,
                                       "n_iter": self.n_iter,
                                       "seed": self.seed})

            if self.grid_pre is not None:
                self.grid_pre.coords_norm = np.vstack((self.grid_pre.coords_norm, self.grid_L1.coords_norm))
                self.grid_pre.coords = np.vstack((self.grid_pre.coords, self.grid_L1.coords))
                self.grid_pre.n_grid = self.grid_pre.coords_norm.shape[0]
            else:
                self.grid_pre = self.grid_L1

        # create LHS (ese) grid
        if coords_norm is None and self.n_grid_LHS > 0:
            self.grid_LHS = LHS(parameters_random=parameters_random,
                                n_grid=self.n_grid_LHS,
                                grid_pre=self.grid_pre,
                                options={"criterion": ["ese"],
                                         "seed": self.seed})

        if self.grid_L1 is None and self.grid_LHS is not None:
            self.coords_norm = self.grid_LHS.coords_norm
        elif self.n_grid_L1 is not None and self.grid_LHS is None:
            self.coords_norm = self.grid_L1.coords_norm
        elif self.n_grid_L1 is not None and self.grid_LHS is not None:
            self.coords_norm = np.vstack((self.grid_L1.coords_norm, self.grid_LHS.coords_norm))

        # Denormalize grid to original parameter space
        self.coords = self.get_denormalized_coordinates(self.coords_norm)

        # Generate unique IDs of grid points
        self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]


class LHS_L1(RandomGrid):
    """
    LHS-L1 optimized grid object

    Parameters
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid: int
        Number of random samples to generate
    options: dict, optional, default=None
        Grid options:
        - 'corr'            : optimizes design points in their spearman correlation coefficients
        - 'maximin' or 'm'  : optimizes design points in their maximum minimal distance using the Phi-P criterion
        - 'ese'             : uses an enhanced evolutionary algorithm to optimize the Phi-P criterion
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points

    Examples
    --------
    >>> import pygpc
    >>> grid = pygpc.LHS_L1(parameters_random=parameters_random,
    >>>                     n_grid=100,
    >>>                     gpc=gpc,
    >>>                     options={"method": "greedy",
    >>>                              "criterion": ["mc"],
    >>>                              "weights_L1": [1],
    >>>                              "weights": [0.25, 0.75],
    >>>                              "n_pool": 1000,
    >>>                              "seed": None})

    Attributes
    ----------
    parameters_random : OrderedDict of RandomParameter instances
        OrderedDict containing the RandomParameter instances the grids are generated for
    n_grid : int or float
        Number of random samples in grid
    seed : float, optional, default=None
        Seeding point to replicate random grid
    options: dict, optional, default=None
        Grid options:
        - method: "greedy", "iteration"
        - criterion: ["mc"], ["tmc", "cc"]
        - weights: [1], [0.5, 0.5]
        - n_pool: size of samples in pool to choose greedy results from
        - n_iter: number of iterations
        - seed: random seed
    coords : ndarray of float [n_grid_add x dim]
        Grid points to add (model space)
    coords_norm : ndarray of float [n_grid_add x dim]
        Grid points to add (normalized space)
    coords_gradient : ndarray of float [n_grid x dim x dim]
        Denormalized coordinates xi
    coords_gradient_norm : ndarray of float [n_grid x dim x dim]
        Normalized coordinates xi
    coords_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    coords_gradient_id : list of UUID objects (version 4) [n_grid]
        Unique IDs of grid points
    """

    def __init__(self, parameters_random, gpc, n_grid=None, options=None, coords=None, coords_norm=None,
                 coords_gradient=None, coords_gradient_norm=None, coords_id=None, coords_gradient_id=None,
                 grid_pre=None):
        """
        Constructor; Initializes Grid instance; Generates grid or copies provided content
        """
        if options is None:
            options = dict()

        if type(options) is dict:
            if "weights" not in options.keys():
                options["weights"] = [0.5, 0.5]

            if "method" not in options.keys():
                options["method"] = "iteration"

            if "n_pool" not in options.keys():
                options["n_pool"] = 10000

            if "n_iter" not in options.keys():
                options["n_iter"] = 1000

            if "seed" not in options.keys():
                options["seed"] = None

            if "criterion" not in options.keys():
                options["criterion"] = ["mc"]

            if "weights_L1" not in options.keys() or options["weights_L1"] is None:
                options["weights_L1"] = (np.ones(len(options["criterion"])) / len(options["criterion"])).tolist()

        self.n_pool = options["n_pool"]
        self.n_iter = options["n_iter"]
        self.gpc = gpc
        self.seed = options["seed"]
        self.method = options["method"]
        self.criterion = options["criterion"]
        self.weights_L1 = options["weights_L1"]
        self.grid_L1 = None
        self.grid_LHS = None
        self.grid_pre = grid_pre

        if type(self.criterion) is not list:
            self.criterion = [self.criterion]

        super(LHS_L1, self).__init__(parameters_random,
                                     n_grid=n_grid,
                                     options=options,
                                     coords=coords,
                                     coords_norm=coords_norm,
                                     coords_gradient=coords_gradient,
                                     coords_gradient_norm=coords_gradient_norm,
                                     coords_id=coords_id,
                                     coords_gradient_id=coords_gradient_id)

        self.weights = options["weights"]

        if coords_norm is None:
            self.n_grid_LHS = int(np.round(self.n_grid * self.weights[0]))
            self.n_grid_L1 = self.n_grid - self.n_grid_LHS
        else:
            self.n_grid_LHS = None
            self.n_grid_L1 = None

        # create LHS (ese) grid
        if coords_norm is None and self.n_grid_LHS > 0:
            self.grid_LHS = LHS(parameters_random=parameters_random,
                                n_grid=self.n_grid_LHS,
                                grid_pre=grid_pre,
                                options={"criterion": ["ese"],
                                         "seed": self.seed})

            if self.grid_pre is not None:
                self.grid_pre.coords_norm = np.vstack((self.grid_pre.coords_norm, self.grid_LHS.coords_norm))
                self.grid_pre.coords = np.vstack((self.grid_pre.coords, self.grid_LHS.coords))
                self.grid_pre.n_grid = self.grid_pre.coords_norm.shape[0]
            else:
                self.grid_pre = self.grid_LHS

        # create L1 grid
        if coords_norm is None and self.n_grid_L1 > 0:
            self.grid_L1 = L1(parameters_random=parameters_random,
                              n_grid=self.n_grid_L1,
                              grid_pre=self.grid_pre,
                              gpc=gpc,
                              options={"method": self.method,
                                       "criterion": self.criterion,
                                       "weights": self.weights_L1,
                                       "n_pool": self.n_pool,
                                       "n_iter": self.n_iter,
                                       "seed": self.seed})

        if self.grid_L1 is None and self.grid_LHS is not None:
            self.coords_norm = self.grid_LHS.coords_norm
        elif self.n_grid_L1 is not None and self.grid_LHS is None:
            self.coords_norm = self.grid_L1.coords_norm
        elif self.n_grid_L1 is not None and self.grid_LHS is not None:
            self.coords_norm = np.vstack((self.grid_LHS.coords_norm, self.grid_L1.coords_norm))

        # Denormalize grid to original parameter space
        self.coords = self.get_denormalized_coordinates(self.coords_norm)

        # Generate unique IDs of grid points
        self.coords_id = [uuid.uuid4() for _ in range(self.n_grid)]


def project_grid(grid, p_matrix, mode="reduce"):
    """
    Transforms grid from original to reduced parameter space or vice versa.

    Parameters
    ----------
    grid : Grid object
        Grid object to transform
    p_matrix : ndarray of float [n_reduced, n_original]
        Projection matrix
    mode : str
        Direction of transformation ("reduce", "expand")

    Returns
    -------
    grid_trans : Grid object
        Transformed grid object
    """
    grid_trans = copy.deepcopy(grid)
    p_matrix_norm = np.sum(np.abs(p_matrix), axis=1)

    if mode == "reduce":
        p = p_matrix.transpose()
        p_n = p / p_matrix_norm[np.newaxis, :]
    elif mode == "expand":
        p = p_matrix
        p_n = p / p_matrix_norm[:, np.newaxis]
    else:
        raise ValueError("Specified mode not implemented... ('reduce', 'expand')")

    # transform variables of original grid to reduced parameter space
    grid_trans.coords = np.matmul(grid.coords, p)
    grid_trans.coords_norm = np.matmul(grid.coords_norm, p_n)

    return grid_trans


def workhorse_greedy(idx_list, psy_opt, psy_pool, criterion):
    """
    Workhorse for coherence calculation (greedy algorithm)

    Parameters
    ----------
    idx_list : list of int [n_idx]
        Indices of rows of pool matrix the coherence is calculated for
    psy_opt : ndarray of float [n_grid_current, n_basis]
        GPC matrix of previous iteration
    psy_pool : ndarray of float [n_pool, n_basis]
        GPC matrix of pool
    criterion : list of str
        Optimality criteria

    Returns
    -------
    crit : ndarray of float [n_idx, n_criterion]
        Optimality measures
    """

    crit = np.ones((len(idx_list), len(criterion))) * 1e6

    # determine gram matrix of psy_opt
    psy_opt_gram = np.matmul(psy_opt.T, psy_opt)

    if "D" in criterion or "D-coh" in criterion:
        sign = np.zeros((len(idx_list), 1))
        logdet = np.zeros((len(idx_list), 1))

    for j in range(len(idx_list)):
        psy_test = np.vstack((psy_opt, psy_pool[idx_list[j], :]))

        # update gram matrix
        psy_test_gram = psy_opt_gram + np.outer(psy_test[-1, :], psy_test[-1, :])

        if "mc" in criterion:
            crit[j, criterion.index("mc")] = mutual_coherence(psy_test)

        if "tmc" in criterion:
            crit[j, criterion.index("tmc")] = t_averaged_mutual_coherence(psy_test_gram)

        if "cc" in criterion:
            crit[j, criterion.index("cc")] = average_cross_correlation_gram(psy_test_gram)

        if "D" in criterion or "D-coh" in criterion:
            # for n_grid < n_basis only consider the first n_grid basis functions because of determinant
            n_basis_det = np.min((psy_test.shape[0], psy_test.shape[1]))

            # determinant of inverse of Gram is the inverse of the determinant
            sign[j], logdet[j] = np.linalg.slogdet(psy_test_gram[:n_basis_det, :n_basis_det])
            #sign[j], logdet[j] = np.linalg.slogdet(np.matmul(psy_test[:, :n_basis_det].T, psy_test[:, :n_basis_det]))
            logdet[j] = -logdet[j]

    if "D" not in criterion and "D-coh" not in criterion:
        return crit
    else:
        return sign, logdet


def workhorse_iteration(idx_list, gpc, n_grid, criterion, grid_pre=None, options=None):
    """
    Workhorse for coherence calculation (iterative algorithm)

    Parameters
    ----------
    idx_list : list of int [n_idx]
        Indices of iterations
    gpc : GPC object
        GPC object
    n_grid : int
        Number of grid points
    criterion : list of str
        Optimality criteria
    grid_pre : Grid object, optional, default: None
        Grid object, which is going to be extended.
    options : dict, optional, default: False
        Dictionary containing the grid options

    Returns
    -------
    crit : ndarray of float [n_idx, n_criterion]
        Optimality measures
    coords_norm_list : list [n_idx] of ndarray [n_grid x dim]
        Normalized grid coordinates of grid realizations
    """
    coords_norm_list = []
    crit = np.ones((len(idx_list), len(criterion))) * 1e6
    backend_backup = gpc.backend
    gpc.backend = "cpu"

    if "D" in criterion or "D-coh" in criterion:
        sign = np.zeros((len(idx_list), 1))
        neg_logdet = np.zeros((len(idx_list), 1))

    if grid_pre is not None and grid_pre.n_grid > 0:
        psy_pool_pre = gpc.create_gpc_matrix(b=gpc.basis.b, x=grid_pre.coords_norm, gradient=False)
    else:
        psy_pool_pre = None

    for i in range(len(idx_list)):
        # print(f"idx_list iteration: {i}")
        if gpc.p_matrix is not None:
            if "D-coh" in criterion:
                test_grid = CO(parameters_random=gpc.problem_original.parameters_random,
                               n_grid=n_grid,
                               grid_pre=grid_pre,
                               gpc=gpc,
                               options=options)
            else:
                test_grid = Random(parameters_random=gpc.problem_original.parameters_random,
                                   n_grid=n_grid,
                                   grid_pre=grid_pre,
                                   options={"seed": options["seed"]})
        else:
            if "D-coh" in criterion:
                test_grid = CO(parameters_random=gpc.problem.parameters_random,
                               n_grid=n_grid,
                               grid_pre=grid_pre,
                               gpc=gpc,
                               options=options)
            else:
                test_grid = Random(parameters_random=gpc.problem.parameters_random,
                                   n_grid=n_grid,
                                   grid_pre=grid_pre,
                                   options={"seed": options["seed"]})

        coords_norm = test_grid.coords_norm

        # save current coords norm
        coords_norm_list.append(coords_norm)

        # get the normalized gpc matrix
        if gpc.p_matrix is not None:
            psy_pool = gpc.create_gpc_matrix(b=gpc.basis.b,
                                             x=np.matmul(coords_norm, gpc.p_matrix.transpose() /
                                                         gpc.p_matrix_norm[np.newaxis, :]),
                                             gradient=False)
        else:
            psy_pool = gpc.create_gpc_matrix(b=gpc.basis.b, x=coords_norm, gradient=False)

        if psy_pool_pre is not None:
            psy_pool = np.vstack((psy_pool_pre, psy_pool))

        psy_pool_norm = psy_pool / np.abs(psy_pool).max(axis=0)

        # test current matrix
        if "mc" in criterion:
            crit[i, criterion.index("mc")] = mutual_coherence(psy_pool_norm)

        if "tmc" in criterion:
            crit[i, criterion.index("tmc")] = t_averaged_mutual_coherence(np.matmul(psy_pool_norm.T, psy_pool_norm))

        if "cc" in criterion:
            crit[i, criterion.index("cc")] = average_cross_correlation_gram(np.matmul(psy_pool_norm.T, psy_pool_norm))

        if "D" in criterion or "D-coh" in criterion:
            # for n_grid < n_basis only consider the first n_grid basis functions because of determinant
            n_basis_det = np.min((n_grid, gpc.basis.n_basis))

            # determinant of inverse of Gram is the inverse of the determinant
            sign[i], neg_logdet[i] = np.linalg.slogdet(np.matmul(psy_pool_norm[:, :n_basis_det].T, psy_pool_norm[:, :n_basis_det]))
            neg_logdet[i] = -neg_logdet[i]

    gpc.backend = backend_backup

    if "D" not in criterion and "D-coh" not in criterion:
        return crit, coords_norm_list
    else:
        return sign, neg_logdet, coords_norm_list


def workhorse_get_det_updated_fim_matrix(index_list, gpc_matrix_pool, fim_matrix, n_basis_limit):
    """
    Workhorse to determine the determinant of the Fisher Information matrix

    Parameters
    ----------
    index_list : list of int
        Indices of coordinates to test
    gpc_matrix_pool : ndarray of float [n_grid_pool x n_basis]
        Gpc matrix of large pool
    fim_matrix : ndarray of float [n_basis x n_basis]
        Fisher information matrix

    Returns
    -------
    det : float
        Determinant of updated Fisher Information matrix
    """
    sign = np.zeros(len(index_list))
    logdet = np.zeros(len(index_list))

    for i, idx in enumerate(index_list):
        fim_matrix_test = fim_matrix + np.outer(gpc_matrix_pool[idx, :n_basis_limit],
                                                gpc_matrix_pool[idx, :n_basis_limit])
        sign[i], logdet[i] = np.linalg.slogdet(fim_matrix_test)

    return sign, logdet
