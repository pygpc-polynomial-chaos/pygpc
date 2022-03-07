import numpy as np
from scipy.fftpack import ifft
from scipy.special import roots_genlaguerre


def get_quadrature_jacobi_1d(n, p, q):
    """
    Get knots and weights of Jacobi polynomials.

    knots, weights = Grid.get_quadrature_jacobi_1d(n, p, q)

    Parameters
    ----------
    n: int
        Number of knots
    p: float
        First shape parameter
    q: float
        Second shape parameter

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
    t01 = 1.0 * (p - q) / (2 + q + p)
    t02 = 1.0 * ((p - q) * (q + p)) / ((2 * n_arr + q + p) * (2 * n_arr + 2 + q + p))
    t1 = np.append(t01, t02)
    t2 = np.sqrt((4.0 * n_arr * (n_arr + q) * (n_arr + p) * (n_arr + q + p)) / (
            (2 * n_arr - 1 + q + p) * (2 * n_arr + q + p) ** 2 * (2 * n_arr + 1 + q + p)))

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
    n = int(n)
    knots, weights = np.polynomial.hermite_e.hermegauss(n)
    weights = np.array(list(2.0 * weights / np.sum(weights)))

    return knots, weights


def get_quadrature_laguerre_1d(n, alpha):
    """
    Get knots and weights of Laguerre polynomials (gamma distribution).

    knots, weights = Grid.get_quadrature_laguerre_1d(n)

    Parameters
    ----------
    n: int
        number of knots
    alpha: float
        Parameter of Laguerre polynomial

    Returns
    -------
    knots: np.ndarray
        knots of the grid
    weights: np.ndarray
        weights of the grid
    """
    n = int(n)
    knots, weights = roots_genlaguerre(n=n, alpha=alpha)

    return knots, weights


# TODO: review this
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
    n = int(n)

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
    knots: ndarray
        Knots of the grid
    weights: ndarray
        Weights of the grid

    Notes
    -----
    .. [1] Davis, P. J., Rabinowitz, P. (2007). Methods of numerical integration.
       Courier Corporation, second edition, ISBN: 0486453391.

    .. [2] Gautschi, W. (1967). Numerical quadrature in the presence of a singularity.
       SIAM Journal on Numerical Analysis, 4(3), 357-362.

    .. [3] Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw-Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195-202.
    """
    n = int(n)

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
    .. [1] Davis, P. J., Rabinowitz, P. (2007). Methods of numerical integration.
       Courier Corporation, second edition, ISBN: 0486453391.

    .. [2] Gautschi, W. (1967). Numerical quadrature in the presence of a singularity.
       SIAM Journal on Numerical Analysis, 4(3), 357-362.

    .. [3] Waldvogel, J. (2006). Fast construction of the Fejer and Clenshaw–Curtis quadrature rules.
       BIT Numerical Mathematics, 46(1), 195-202.
    """
    n = int(n)

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
