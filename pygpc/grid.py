# -*- coding: utf-8 -*-
"""
Functions and classes that provide data and methods for the generation and processing of numerical grids
"""

import numpy as np
from builtins import range

from scipy.fftpack import ifft
from sklearn.utils.extmath import cartesian
from .misc import get_multi_indices
from .misc import vprint


def get_quadrature_jacobi_1d(N, b, a):
    """
    Get knots and weights of Jacobi polynomials (beta distribution).

    knots, weights = get_quadrature_jacobi_1d(N, b, a)

    Parameter
    ---------
    N: int
        number of knots
    a: float
        lower limit of quadrature coefficients
    b: float
        upper limit of quadrature coefficients

    Returns
    -------
    knots: np.ndarray
        knots of the quadratur grid
    weights: np.ndarray
        weights of the quadratur grid
    """
    # make array to count N: 0, 1, ..., N-1
    N_arr = np.arange(N)

    # compose diagonals for companion matrix
    t01 = 1.0 * (b - a) / (2 + a + b)
    t02 = 1.0 * ((b - a) * (a + b)) / ((2 * N_arr + a + b) * (2 * N_arr + 2 + a + b))
    t1 = np.append(t01, t02)
    t2 = np.sqrt((4.0 * N_arr * (N_arr + a) * (N_arr + b) * (N_arr + a + b)) / (
            (2 * N_arr - 1 + a + b) * (2 * N_arr + a + b) ** 2 * (2 * N_arr + 1 + a + b)))

    # compose companion matrix
    T = np.diag(t1) + np.diag(t2, 1) + np.diag(t2, -1)

    # evaluate roots of polynomials (the abscissas are the roots of the
    # characteristic polynomial, i.d. the eigenvalues of the companion matrix)
    # the weights can be derived from the corresponding eigenvectors.
    eigvals, eigvecs = np.linalg.eig(T)
    idx_sorted = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx_sorted]

    weights = 2.0 * eigvecs[0, idx_sorted] ** 2
    knots = eigvals_sorted

    return knots, weights


def get_quadrature_hermite_1d(N):
    """
    Get knots and weights of Hermite polynomials (normal distribution).

    knots, weights = get_quadrature_hermite_1d(N)

    Parameter
    ---------
    N: int
        number of knots

    Returns
    -------
    knots: np.ndarray
        knots of the quadratur grid
    weights: np.ndarray
        weights of the quadratur grid
    """
    N = np.int(N)
    knots, weights = np.polynomial.hermite_e.hermegauss(N)
    weights = np.array(list(2.0 * weights / np.sum(weights)))

    return knots, weights


def get_quadrature_clenshaw_curtis_1d(N):
    """
    Get the Clenshaw Curtis nodes and weights.

    knots, weights = get_quadrature_clenshaw_curtis_1d(N)

    Parameter
    ---------
    N: int
        number of knots

    Returns
    -------
    knots: np.ndarray
        knots of the clenshaw_curtis grid
    weights: np.ndarray
        weights of the clenshaw_curtis grid
    """
    N = np.int(N)

    if N == 1:
        knots = 0
        weights = 2
    else:
        n = N - 1
        C = np.zeros((N, 2))
        k = 2 * (1 + np.arange(np.floor(n / 2)))
        C[::2, 0] = 2 / np.hstack((1, 1 - k * k))
        C[1, 1] = -n
        V = np.vstack((C, np.flipud(C[1:n, :])))
        F = np.real(ifft(V, n=None, axis=0))
        knots = F[0:N, 1]
        weights = np.hstack((F[0, 0], 2 * F[1:n, 0], F[n, 0]))

    return knots, weights


def get_quadrature_fejer1_1d(N):
    """
    Computes the Fejer type 1 nodes and weights.
    
    This method uses a direct approach. The paper by Waldvogel
    exhibits a more efficient approach using Fourier transforms.

    Reference:
    Philip Davis, Philip Rabinowitz,
    Methods of Numerical Integration,
    Second Edition,
    Dover, 2007,
    ISBN: 0486453391 Titel anhand dieser ISBN in Citavi-Projekt übernehmen,
    LC: QA299.3.D28.

    Walter Gautschi,
    Numerical Quadrature in the Presence of a Singularity,
    SIAM Journal on Numerical Analysis,
    Volume 4, Number 3, 1967, pages 357-362.

    Joerg Waldvogel,
    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules,
    BIT Numerical Mathematics,
    Volume 43, Number 1, 2003, pages 1-18.

    knots, weights = get_quadrature_fejer1_1d(N)

    Parameter
    ---------
    N: int
        number of knots

    Returns
    -------
    knots: np.ndarray
        knots of the clenshaw_curtis grid
    weights: np.ndarray
        weights of the clenshaw_curtis grid
   """
    N = np.int(N)

    theta = np.zeros(N)

    for i in range(0, N):
        theta[i] = float(2 * N - 1 - 2 * i) * np.pi / float(2 * N)

    knots = np.zeros(N)

    for i in range(0, N):
        knots[i] = np.cos(theta[i])

    weights = np.zeros(N)

    for i in range(0, N):
        weights[i] = 1.0
        jhi = (N // 2)
        for j in range(0, jhi):
            angle = 2.0 * float(j + 1) * theta[i]
            weights[i] = weights[i] - 2.0 * np.cos(angle) / float(4 * (j + 1) ** 2 - 1)

    for i in range(0, N):
        weights[i] = 2.0 * weights[i] / float(N)

    return knots, weights


def get_quadrature_fejer2_1d(N):
    """
    Computes the Fejer type 2 nodes and weights (Clenshaw Curtis without boundary nodes).
        
    This method uses a direct approach. The paper by Waldvogel
    exhibits a more efficient approach using Fourier transforms.

    Reference:
    Philip Davis, Philip Rabinowitz,
    Methods of Numerical Integration,
    Second Edition,
    Dover, 2007,
    ISBN: 0486453391 Titel anhand dieser ISBN in Citavi-Projekt übernehmen,
    LC: QA299.3.D28.

    Walter Gautschi,
    Numerical Quadrature in the Presence of a Singularity,
    SIAM Journal on Numerical Analysis,
    Volume 4, Number 3, 1967, pages 357-362.

    Joerg Waldvogel,
    Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules,
    BIT Numerical Mathematics,
    Volume 43, Number 1, 2003, pages 1-18.

    knots, weights = get_quadrature_fejer2_1d(N)

    Parameter
    ---------
    N: int
        number of knots

    Returns
    -------
    knots: np.ndarray
        knots of the clenshaw_curtis grid
    weights: np.ndarray
        weights of the clenshaw_curtis grid
    """
    N = np.int(N)

    if N == 1:

        knots = np.array([0.0])
        weights = np.array([2.0])

    elif N == 2:

        knots = np.array([-0.5, +0.5])
        weights = np.array([1.0, 1.0])

    else:

        theta = np.zeros(N)

        for i in range(0, N):
            theta[i] = float(N - i) * np.pi / float(N + 1)

        knots = np.zeros(N)

        for i in range(0, N):
            knots[i] = np.cos(theta[i])

        weights = np.zeros(N)

        for i in range(0, N):

            weights[i] = 1.0
            jhi = ((N - 1) // 2)

            for j in range(0, jhi):
                angle = 2.0 * float(j + 1) * theta[i]
                weights[i] = weights[i] - 2.0 * np.cos(angle) / float(4 * (j + 1) ** 2 - 1)
                p = 2 * ((N + 1) // 2) - 1

            weights[i] = weights[i] - np.cos(float(p + 1) * theta[i]) / float(p)

        for i in range(0, N):
            weights[i] = 2.0 * weights[i] / float(N + 1)

    return knots, weights


def get_quadrature_patterson_1d(N):
    """
    Computes the nested Gauss-Patterson nodes and weights for N = 1,3,7,15,31.

    knots, weights = get_quadrature_patterson_1d(N)

    Parameter
    ---------
    N: int
        number of knots
        possible values: 1, 3, 7, 15, 31

    Returns
    -------
    knots: np.ndarray
        knots of the clenshaw_curtis grid
    weights: np.ndarray
        weights of the clenshaw_curtis grid
    """
    x = np.zeros(N)
    w = np.zeros(N)

    if N == 1:

        x = 0.0

        w = 2.0

    elif N == 3:

        x[0] = -0.77459666924148337704
        x[1] = 0.0
        x[2] = 0.77459666924148337704

        w[0] = 0.555555555555555555556
        w[1] = 0.888888888888888888889
        w[2] = 0.555555555555555555556

    elif N == 7:

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

    elif N == 15:

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

    elif N == 31:

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

    knots = x
    weights = w

    return knots, weights


def get_denormalized_coordinates(coords_norm, pdf_type, grid_shape, limits):
    """
    Denormalize grid from standardized ([-1, 1] except hermite) to original parameter space for simulations.

    coords = get_denormalized_coordinates(coords_norm, pdf_type, grid_shape, limits)

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    grid_shape: [2 x N_vars] list of list of float
        shape parameters of PDF
        beta (jacobi):  [alpha, beta]
        norm (hermite): [mean, std]
    limits: [2 x N_vars] list of list of float
        upper and lower bounds of PDF
        beta (jacobi):  [min, max]
        norm (hermite): [0, 0] (unused)
    coords_norm: [N_samples x dim] np.ndarray
        normalized [-1, 1] coordinates xi

    Returns
    -------
    coords: [N_samples x dim] np.ndarray
        denormalized coordinates xi
    """
    coords = np.zeros(coords_norm.shape)

    for i_dim in range(coords_norm.shape[1]):
        
        # if gridtype[i_dim] == 'jacobi' or gridtype[i_dim] == 'cc' or gridtype[i_dim] == 'fejer2':
        if pdf_type[i_dim] == "beta":
            coords[:, i_dim] = (coords_norm[:, i_dim] + 1) / 2 * (limits[1][i_dim] - limits[0][i_dim]) + limits[0][
                i_dim]
        # if gridtype[i_dim] == 'hermite':
        if pdf_type[i_dim] == "norm" or pdf_type[i_dim] == "normal":
            coords[:, i_dim] = coords_norm[:, i_dim] * grid_shape[1][i_dim] + grid_shape[0][i_dim]

    return coords


def get_normalized_coordinates(coords, pdf_type, grid_shape, limits):
    """
    Normalize grid from original parameter (except hermite) to standardized ([-1, 1] space for simulations.

    coords_norm = get_normalized_coordinates(coords, pdf_type, grid_shape, limits)

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    grid_shape: [2 x N_vars] list of list of float
        shape parameters of PDF
        beta (jacobi):  [alpha, beta]
        norm (hermite): [mean, std]
    limits: [2 x N_vars] list of list of float
        upper and lower bounds of PDF
        beta (jacobi):  [min, max]
        norm (hermite): [0, 0] (unused)
    coords: [N_samples x dim] np.ndarray
        denormalized coordinates xi

    Returns
    -------
    coords_norm: [N_samples x dim] np.ndarray
        normalized [-1, 1] coordinates xi
    """
    coords_norm = np.zeros(coords.shape)

    for i_dim in range(coords.shape[1]):
        
        # if gridtype[i_dim] == 'jacobi' or gridtype[i_dim] == 'cc' or gridtype[i_dim] == 'fejer2':
        if pdf_type[i_dim] == "beta":
            coords_norm[:, i_dim] = (coords[:, i_dim] - limits[0][i_dim])
            coords_norm[:, i_dim] = coords_norm[:, i_dim] / (limits[1][i_dim] - limits[0][i_dim]) * 2.0 - 1

        # if gridtype[i_dim] == 'hermite':
        if pdf_type[i_dim] == "norm" or pdf_type[i_dim] == "normal":
            coords_norm[:, i_dim] = (coords[:, i_dim] - grid_shape[0][i_dim]) / grid_shape[1][i_dim]

    return coords_norm


class TensorGrid:
    """
    Generate TensorGrid object instance.

    Initialisation
    --------------
    TensorGrid(pdf_type, grid_type, grid_shape, limits, N):

    Parameters
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
    N: [N_vars] list of int
        number of nodes in each dimension
    """

    def __init__(self, pdf_type, grid_type, grid_shape, limits, N):
        self.pdf_type = pdf_type  # 'beta', 'normal'
        self.grid_type = grid_type  # 'jacobi', 'hermite', 'cc', 'fejer2'
        self.grid_shape = grid_shape  # pdf_shape: jacobi: -> [alpha, beta] hermite: -> [mean, std]
        self.limits = limits  # limits: [min, max]
        self.N = N  # number of nodes in each dimension [dim x 1]
        self.dim = len(self.N)  # number of dimension

        # get knots and weights of polynomials in each dimension
        self.knots_dim = []
        self.weights_dim = []
        knots_temp = 0
        weights_temp = 0
        for i_dim in range(self.dim):
            
            if self.grid_type[i_dim] == 'jacobi':  # jacobi polynomials
                knots_temp, weights_temp = get_quadrature_jacobi_1d(self.N[i_dim], self.grid_shape[0][i_dim] - 1,
                                                                self.grid_shape[1][i_dim] - 1)
            if self.grid_type[i_dim] == 'hermite':  # hermite polynomials
                knots_temp, weights_temp = get_quadrature_hermite_1d(self.N[i_dim])
            if self.grid_type[i_dim] == 'cc':  # Clenshaw Curtis
                knots_temp, weights_temp = get_quadrature_clenshaw_curtis_1d(self.N[i_dim])
            if self.grid_type[i_dim] == 'fejer2':  # Fejer type 2 (Clenshaw Curtis without boundary nodes)
                knots_temp, weights_temp = get_quadrature_fejer2_1d(self.N[i_dim])
            if self.grid_type[i_dim] == 'patterson':  # Gauss-Patterson (Nested Legendre rule)
                knots_temp, weights_temp = get_quadrature_patterson_1d(self.N[i_dim])

            self.knots_dim.append(knots_temp)
            self.weights_dim.append(weights_temp)

        # combine coordinates to full tensored grid (all combinations)
        self.coords_norm = cartesian(self.knots_dim)

        # rescale normalized coordinates in case of normal distributions and "fejer2" or "cc" grids
        # +- 0.675 * sigma -> 50%
        # +- 1.645 * sigma -> 90%
        # +- 1.960 * sigma -> 95%        
        # +- 2.576 * sigma -> 99%
        # +- 3.000 * sigma -> 99.73%
        for i_dim in range(self.dim):
            if (self.pdf_type[i_dim] == "norm" or self.pdf_type[i_dim] == "normal") and (
                    not (self.grid_type[i_dim] == "hermite")):
                self.coords_norm[:, i_dim] = self.coords_norm[:, i_dim] * 1.960

        # determine combined weights of Gauss quadrature
        self.weights = np.prod(cartesian(self.weights_dim), axis=1) / (2.0 ** self.dim)

        # denormalize grid to original parameter space
        self.coords = get_denormalized_coordinates(self.coords_norm, self.pdf_type, self.grid_shape, self.limits)


# TODO: grid_shape[1] of norm is now STD. Check if code changes in sparse.
class SparseGrid:
    """
    Generate SparseGrid object instance.

    Initialisation
    --------------
    SparseGrid(pdf_type, grid_type, grid_shape, limits, level, level_max, interaction_order,
               order_sequence_type, make_grid=True, verbose=True)

    Parameters:
    -----------
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
    """

    def __init__(self, pdf_type, grid_type, grid_shape, limits, level, level_max, interaction_order,
                 order_sequence_type, make_grid=True, verbose=True):
        self.pdf_type = pdf_type  # 'beta', 'normal'
        self.grid_type = grid_type  # 'jacobi', 'hermite', 'cc', 'fejer2'
        self.grid_shape = grid_shape  # pdfshape: jacobi: -> [alpha and beta], hermite: -> [mean, variance]
        self.limits = limits  # limits: [min, max]
        self.level = level  # number of levels in each dimension [dim x 1]
        self.level_max = level_max  # global combined level maximum
        self.interaction_order = interaction_order  # interaction order of parameters and grid
        self.order_sequence_type = order_sequence_type  # 'lin', 'exp' type of order sequence (common: 'exp')
        self.verbose = verbose  # output while grid generation on/off
        self.dim = len(self.level)  # number of dimension
        self.coords = None  # coordinates of gpc model calculation in the system space
        self.coords_norm = None  # coordinates of gpc model calculation in the gpc space [-1,1]
        self.weights = None  # weights for numerical integration for every point in the coordinate space
        self.level_sequence = None  # integer sequence of levels
        self.order_sequence = None  # integer sequence of polynom order of levels

        # grid is generated during initialization or coords, coords_norm and weights are added manually
        if make_grid:
            self.calc_multi_index_lst()
            self.calc_coords_weights()
        else:
            print('Sparse grid initialized but not generated. Please add coords / coords_norm and weights manually.')

    def calc_multi_index_lst(self):
        """
        Calculate the multi index list needed for the calculation of the SparseGrid.
        """
        for i_dim in range(self.dim):
            
            if self.grid_type[i_dim] == 'fejer2':
                self.level_sequence = range(1, self.level[i_dim] + 1)
            else:
                self.level_sequence = range(self.level[i_dim] + 1)

            if self.order_sequence_type == 'exp':  # order = 2**level + 1
                if self.grid_type[i_dim] == 'fejer2':  # start with order = 1 @ level = 1
                    self.order_sequence.append(
                        (2 ** (np.linspace(1, self.level[i_dim], self.level[i_dim]).tolist()) - 1).tolist())
                    self.order_sequence[i_dim][0] = 1
                elif self.grid_type[i_dim] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    self.order_sequence.append(
                        (2 ** (np.linspace(0, self.level[i_dim], self.level[i_dim] + 1).tolist() + 1) - 1).tolist())
                else:  # start with order = 1 @ level = 0
                    self.order_sequence.append(
                        (2 ** np.linspace(0, self.level[i_dim], self.level[i_dim] + 1) + 1).tolist())
                    self.order_sequence[i_dim][0] = 1

            elif self.order_sequence_type == 'lin':  # order = level
                if self.grid_type[i_dim] == 'fejer2':  # start with level = 1 @ order = 1
                    self.order_sequence.append(np.linspace(1, self.level[i_dim] + 1, self.level[i_dim] + 1).tolist())
                elif self.grid_type[i_dim] == 'patterson':  # start with order = 1 @ level = 0 [1,3,7,15,31,...]
                    print("Not possible in case of Gauss-Patterson grid.")
                else:  # start with
                    self.order_sequence.append(np.linspace(1, self.level[i_dim] + 1, self.level[i_dim] + 1).tolist())

    def calc_l_level(self):
        """
        Calculate the l level needed for the Fejer grid type 2.

        Returns
        -------
        l_level: np.ndarray
            l level values
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
                l_level = get_multi_indices(self.dim, self.level_max)

        # filter out rows exceeding the individual level cap
        for i_dim in range(self.dim):
            l_level = l_level[l_level[:, i_dim] <= self.level[i_dim]]

        # Consider interaction order (filter out multi-indices exceeding it)
        if self.interaction_order < self.dim:
            if any("fejer2" in s for s in self.grid_type):
                l_level = l_level[np.sum(l_level > 1, axis=1) <= self.interaction_order, :]
            else:
                l_level = l_level[np.sum(l_level > 0, axis=1) <= self.interaction_order, :]

        return l_level

    def calc_grid(self):
        # make cubature lookup table for knots (dl_k) and weights (dl_w) [max(l) x dim]
        vprint("    Generating difference grids...", verbose=self.verbose)
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
        # make list of all tensor products according to multiindex list "l"
        vprint("    Generating subgrids ...", verbose=self.verbose)
        dl_k, dl_w = self.calc_grid()
        l_level = self.calc_l_level()
        dL_k = []
        dL_w = []

        for i_l_level in range(l_level.shape[0]):

            knots_temp = []
            weights_temp = []

            for i_dim in range(self.dim):
                knots_temp.append(np.asarray(dl_k[np.int(l_level[i_l_level, i_dim])][i_dim], dtype=float))
                weights_temp.append(np.asarray(dl_w[np.int(l_level[i_l_level, i_dim])][i_dim], dtype=float))

            # tensor product of knots
            dL_k.append(cartesian(knots_temp))

            # tensor product of weights
            dL_w.append(np.prod(cartesian(weights_temp), axis=1))

        # dL_w = np.hstack(dL_w)
        # dL_k = np.vstack(dL_k)
        return np.hstack(dL_k), np.vstack(dL_w)

    def calc_coords_weights(self):
        # find similar points in grid and formulate Point list
        vprint("    Merging subgrids ...", verbose=self.verbose)
        dL_w, dL_k = self.calc_tensor_products()
        point_number_list = np.zeros(dL_w.shape[0]) - 1
        point_no = 0
        epsilon_k = 1E-6
        coords_norm = []

        while any(point_number_list < 0):
            not_found = point_number_list < 0
            dL_k_nf = dL_k[not_found, :]
            point_temp = np.zeros(dL_k_nf.shape[0]) - 1
            point_temp[np.sum(np.abs(dL_k_nf - dL_k_nf[0, :]), axis=1) < epsilon_k] = point_no
            point_number_list[not_found] = point_temp
            point_no = point_no + 1
            coords_norm.append(dL_k_nf[0, :])

        coords_norm = np.array(coords_norm)
        point_number_list = np.asarray(point_number_list, dtype=int)

        weights = np.zeros(np.amax(point_number_list) + 1) - 999

        for i_point in range(np.amax(point_number_list) + 1):
            weights[i_point] = np.sum(dL_w[point_number_list == i_point])

        # filter for very small weights
        vprint("    Filter grid for very small weights ...", verbose=self.verbose)
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
        vprint("    Denormalizing grid for computations ...", verbose=self.verbose)
        self.coords_norm = coords_norm
        self.coords = get_denormalized_coordinates(coords_norm, self.pdf_type, self.grid_shape, self.limits)


class RandomGrid:
    def __init__(self, pdf_type, grid_shape, limits, N, seed=None):
        """
        Generate RandomGrid object instance

        Parameters
        ----------
        pdf_type: list of str [N_vars]
            variable specific type of pdf ("beta", "normal")
        grid_shape: list of list of float [2 x N_vars]
            shape parameters of PDF
            beta (jacobi):  [alpha, beta]
            norm (hermite): [mean, std]
        limits: list of list of float [2 x N_vars]
            Upper and lower bounds of PDF
            beta (jacobi):  [min, max]
            norm (hermite): [0, 0] (unused)
        N: int
            number of random samples to generate
        seed: float
            seeding point to replicate random grids
        """

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
        self.coords = denorm(self.coords_norm, self.pdf_type, self.grid_shape, self.limits)
