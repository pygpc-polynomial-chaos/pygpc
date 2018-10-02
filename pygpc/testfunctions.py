"""
Testfunctions and electromagnetic field calculations
"""
import numpy as np
import scipy.special as sp
import math
import warnings

from __future__ import division

# TODO: Change SI note in docstrings


def peaks(x):
    """
    Two-dimensional peaks function.

    y = peaks(x)

    Parameters
    ----------
    x: [N x 2] np.ndarray
        input data
    Returns
    -------
    y: [N x 1] np.ndarray
        output data
    """

    y = (3.0 * (1 - x[:, 0]) ** 2. * np.exp(-(x[:, 0] ** 2) - (x[:, 1] + 1) ** 2)- 10.0 * (x[:, 0] / 5.0 - x[:, 0] ** 3
         - x[:, 1] ** 5) * np.exp(-x[:, 0] ** 2 - x[:, 1] ** 2) - 1.0 / 3 * np.exp(-(x[:, 0] + 1) ** 2 - x[:, 1] ** 2))

    return y


def lim_2002(x):
    """
    Two-dimensional test function of Lim et al

    This function is a polynomial in two dimensions, with terms up to degree
    5. It is nonlinear, and it is smooth despite being complex, which is
    common for computer experiment functions (Lim et al., 2002).

    Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design
    and analysis of computer experiments when the output is highly correlated
    over the input space. Canadian Journal of Statistics, 30(1), 109-126.

    f(x) = 9 + 5/2*x1 - 35/2*x2 + 5/2*x1*x2 + 19*x2^2 - 15/2*x1^3
           - 5/2*x1*x2^2 - 11/2*x2^4 + x1^3*x2^2

    y = lim_2002(x)
        
    Parameters
    ----------
    x: [N x 2] np.ndarray
        input data
        xi ∈ [0, 1], for all i = 1, 2

    Returns
    -------
    y: [N x 1] np.ndarray
        output data
    """

    y = (9 + 5.0 / 2 * x[:, 0] - 35.0 / 2 * x[:, 1] + 5.0 / 2 * x[:, 0] * x[:, 1] + 19 * x[:, 1] ** 2
         - 15.0 / 2 * x[:, 0] ** 3 - 5.0 / 2 * x[:, 0] * x[:, 1] ** 2 - 11.0 / 2 * x[:, 1] ** 4
         + x[:, 0] ** 3 * x[:, 1] ** 2)

    return y


def ishigami(x, a, b):
    """
    3-dimensional test function of Ishigami

    The Ishigami function of Ishigami & Homma (1990) is used as an example
    for uncertainty and sensitivity analysis methods, because it exhibits
    strong nonlinearity and nonmonotonicity. It also has a peculiar
    dependence on x3, as described by Sobol' & Levitan (1999).

    Ishigami, T., & Homma, T. (1990, December). An importance quantification
    technique in uncertainty analysis for computer models. In Uncertainty
    Modeling and Analysis, 1990. Proceedings., First International Symposium
    on (pp. 398-403). IEEE.

    Sobol', I. M., & Levitan, Y. L. (1999). On the use of variance reducing
    multipliers in Monte Carlo computations of a global sensitivity index.
    Computer Physics Communications, 117(1), 52-61.

    f(x) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)

    y = ishigami(x,a,b)

    Parameters
    ----------
    x: [N x 3] np.ndarray
        input data
        xi ~ Uniform[-π, π], for all i = 1, 2, 3
    a: float
        shape parameter
    b: float
        shape parameter

    Returns
    -------
    y: [N x 1] np.ndarray
        output data
    """

    y = (np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0]))

    return y


def sphere_zero_mean(x, a, b):
    """
    N-dimensional sphere function with zero mean.

    y = sphere_zero_mean(x,a,b)

    Parameters
    ----------
    x: [N_input x N_dims] np.ndarray
        input data
    a: [N_dims] np.ndarray
        lower bound of input data
    b: [N_dims] np.ndarray
        upper bound of input data

    Returns
    -------
    y: [N_input] np.ndarray
        output data
    """

    try:
        n = x.shape[1]
    except IndexError:
        n = 1
        x = np.array([x])

    # zero mean   
    c2 = (1.0 * n * (b ** 3 - a ** 3)) / (3 * (b - a))

    # sphere function
    y = (np.sum(np.square(x), axis=1) - c2)

    return y


def sphere(x):
    """
    N-dimensional sphere function.
        
    y = sphere(x)
        
    Parameters
    ----------
    x: [N_input x N_dims] np.ndarray
        input data

    Returns
    -------
    y [N_input x 1] np.ndarray
        output data
    """

    y = (np.sum(np.square(x), axis=1))

    return y


def g_function(x, a):
    """
    N-dimensional g-function used by Saltelli and Sobol

    this test function is used as an integrand for various numerical
    estimation methods, including sensitivity analysis methods, because it
    is fairly complex, and its sensitivity indices can be expressed
    analytically. The exact value of the integral with this function as an
    integrand is 1.

    Saltelli, Andrea; Sobol, I. M. (1995): Sensitivity analysis for nonlinear
    mathematical models: numerical experience. In: Mathematical models and
    computer experiment 7 (11), S. 16–28.
        
    y = g_function(x,a)
        
    Parameters
    ----------
    x: [N_input x N_dims] np.ndarray
        input data
    a: [N_dims] np.ndarray
        importance factor of dimensions

    Returns
    -------
        y: [N_input x 1] np.ndarray
            output data
    """

    try:
        x.shape[1]
    except IndexError:
        x = np.array([x])

    # g-function
    y = (np.prod((np.abs(4.0 * x - 2) + a) / (1.0 + a), axis=1))

    return y


def oakley_ohagan_2004(x):
    """
    15-dimensional test function of OAKLEY & O'HAGAN (2004)

    This function's a-coefficients are chosen so that 5 of the input
    variables contribute significantly to the output variance, 5 have a
    much smaller effect, and the remaining 5 have almost no effect on the
    output variance.

    Oakley, J. E., & O'Hagan, A. (2004). Probabilistic sensitivity analysis
    of complex models: a Bayesian approach. Journal of the Royal Statistical
    Society: Series B (Statistical Methodology), 66(3), 751-769.

    y = oakley_ohagan_2004(x)
        
    Parameters
    ----------
    x: [N_input x 15] np.ndarray
        input data
        xi ~ N(μ=0, σ=1), for all i = 1, …, 15.
                
    Returns
    -------
    y: [N_input x 1] np.ndarray
        output data
    """

    # load coefficients
    m = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_M.txt')
    a1 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a1.txt')
    a2 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a2.txt')
    a3 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a3.txt')

    # function
    y = (np.dot(x, a1) + np.dot(np.sin(x), a2) + np.dot(np.cos(x), a3) + np.sum(np.multiply(np.dot(x, m), x), axis=1))

    return y


def welch_1992(x):
    """
    20-dimensional test function of WELCH (1992)
        
    For input variable screening purposes, it can be found that some input
    variables of this function have a very high effect on the output,
    compared to other input variables. As Welch et al. (1992) point out,
    interactions and nonlinear effects make this function challenging.

    Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., &
    Morris, M. D. (1992). Screening, predicting, and computer experiments.
    Technometrics, 34(1), 15-25.
        
    y = welch_1992(x)
        
    Parameters
    ----------
    x: [N_input x 20] np.ndarray
        input data
        xi ~ U(-0.5, 0.5), for all i = 1, …, 20.
                
    Returns
    -------
    y: [N_input x 1] np.ndarray
        output data
    """
    y = (5.0 * x[:, 11] / (1 + x[:, 0]) + 5 * (x[:, 3] - x[:, 19]) ** 2 + x[:, 4] + 40 * x[:, 18] ** 3
         + 5 * x[:, 18] + 0.05 * x[:, 1] + 0.08 * x[:, 2] - 0.03 * x[:, 5] + 0.03 * x[:, 6]
         - 0.09 * x[:, 8] - 0.01 * x[:, 9] - 0.07 * x[:, 10] + 0.25 * x[:, 12] ** 2 - 0.04 * x[:, 13]
         + 0.06 * x[:, 14] - 0.01 * x[:, 16] - 0.03 * x[:, 17])

    return y


def wing_weight(x):
    """
    10-dimensional test function which models a light aircraft wing
        
    Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via
    surrogate modelling: a practical guide. Wiley.
        
    y  = wing_weight(x)
        
    Parameters
    ----------
    x: [N_input x 10] np.ndarray
        input data
        x1(Sw)  ∈ [150, 200]
        x2(Wfw) ∈ [220, 300]
        x3(A)   ∈ [6, 10]
        x4(Λ)   ∈ [-10, 10]
        x5(q)   ∈ [16, 45]
        x6(λ)   ∈ [0.5, 1]
        x7(tc)  ∈ [0.08, 0.18]
        x8(Nz)  ∈ [2.5, 6]
        x9(Wdg) ∈ [1700, 2500]
        x10(Wp) ∈ [0.025, 0.08]
                
    Returns
    -------
    y: [N_input x 1] np.ndarray
        output data
    """
    y = (0.036 * x[:, 0] ** 0.758 * x[:, 1] ** 0.0035
         * (x[:, 2] / np.cos(x[:, 3]) ** 2) ** 0.6 * x[:, 4] ** 0.006 * x[:, 5] ** 0.04
         * (100 * x[:, 6] / np.cos(x[:, 3])) ** -0.3 * (x[:, 7] * x[:, 8]) ** 0.49
         + x[:, 0] * x[:, 9])

    return y


def calc_potentials_3layers_surface_electrodes(conductivities, radii, anode_pos, cathode_pos, p, nbr_polynomials=50):
    """
    Calculate the electric potential in a 3-layered sphere caused by point-like electrodes.

    S.Rush, D.Driscol EEG electrode sensitivity--an application of reciprocity

    potential = calc_potentials_3layers_surface_electrodes(conductivities, radii, anode_pos, cathode_pos, p,
                                                            nbr_polynomials=50):

    Parameters
    ----------
    conductivities: [3] list
        conductivity of the 3 layers (innermost to outermost), in S/m
    radii: [3] list
        radius of each of the 3 layers (innermost to outermost), in mm
    anode_pos: [3 x 1] np.ndarray
        position of the anode_pos, in mm
    cathode_pos: [3 x 1] np.ndarray
        position of cathode_pos, in mm
    p: [N x 3] np.ndarray
        list of positions where the poteitial should be calculated, in mm
    nbr_polynomials: int, optional, default=50
        number of of legendre polynomials to use

    Returns
    -------
    potential: [N x 1] np.ndarray
        values of the electric potential, in V
    """
    assert len(radii) == 3
    assert radii[0] < radii[1] < radii[2]
    assert len(conductivities) == 3
    assert len(anode_pos) == 3
    assert len(cathode_pos) == 3
    assert p.shape[1] == 3

    b_over_s = float(conductivities[0]) / float(conductivities[1])
    s_over_t = float(conductivities[1]) / float(conductivities[2])
    radius_brain = radii[0] * 1e-3
    radius_skull = radii[1] * 1e-3
    radius_skin = radii[2] * 1e-3

    r = np.linalg.norm(p, axis=1) * 1e-3
    theta = np.arccos(p[:, 2] * 1e-3 / r)
    phi = np.arctan2(p[:, 1], p[:, 0])

    p_r = np.vstack((r, theta, phi)).T

    cathode_pos = (np.sqrt(cathode_pos[0] ** 2 + cathode_pos[1] ** 2 + cathode_pos[2] ** 2) * 1e-3,
                   np.arccos(cathode_pos[2] /
                             np.sqrt(cathode_pos[0] ** 2 + cathode_pos[1] ** 2 + cathode_pos[2] ** 2)),
                   np.arctan2(cathode_pos[1], cathode_pos[0]))

    anode_pos = (np.sqrt(anode_pos[0] ** 2 + anode_pos[1] ** 2 + anode_pos[2] ** 2) * 1e-3,
                 np.arccos(anode_pos[2] /
                           np.sqrt(anode_pos[0] ** 2 + anode_pos[1] ** 2 + anode_pos[2] ** 2)),
                 np.arctan2(anode_pos[1], anode_pos[0]))

    A = lambda n: ((2 * n + 1) ** 3 / (2 * n)) / (((b_over_s + 1) * n + 1) * ((s_over_t + 1) * n + 1) +
                                                  (b_over_s - 1) * (s_over_t - 1) * n * (n + 1) * (
                                                              radius_brain / radius_skull) ** (2 * n + 1) +
                                                  (s_over_t - 1) * (n + 1) * ((b_over_s + 1) * n + 1) * (
                                                              radius_skull / radius_skin) ** (2 * n + 1) +
                                                  (b_over_s - 1) * (n + 1) * ((s_over_t + 1) * (n + 1) - 1) * (
                                                              radius_brain / radius_skin) ** (2 * n + 1))
    # All of the bellow are modified: division by raidus_skin moved to the
    # coefficients calculations due to numerical constraints
    # THIS IS DIFFERENT FROM THE PAPER (there's a sum instead of difference)
    S = lambda n: (A(n)) * ((1 + b_over_s) * n + 1) / (2 * n + 1)
    U = lambda n: (A(n) * radius_skin) * n * (1 - b_over_s) * \
                  radius_brain ** (2 * n + 1) / (2 * n + 1)
    T = lambda n: (A(n) / ((2 * n + 1) ** 2)) * \
                  (((1 + b_over_s) * n + 1) * ((1 + s_over_t) * n + 1) +
                   n * (n + 1) * (1 - b_over_s) * (1 - s_over_t) * (radius_brain / radius_skull) ** (2 * n + 1))
    W = lambda n: ((n * A(n) * radius_skin) / ((2 * n + 1) ** 2)) * \
                  ((1 - s_over_t) * ((1 + b_over_s) * n + 1) * radius_skull ** (2 * n + 1) +
                   (1 - b_over_s) * ((1 + s_over_t) * n + s_over_t) * radius_brain ** (2 * n + 1))

    brain_region = np.where(p_r[:, 0] <= radius_brain)[0]
    skull_region = np.where(
        (p_r[:, 0] > radius_brain) * (p_r[:, 0] <= radius_skull))[0]
    skin_region = np.where((p_r[:, 0] > radius_skull)
                           * (p_r[:, 0] <= radius_skin))[0]
    inside_sphere = np.where((p_r[:, 0] <= radius_skin))[0]
    outside_sphere = np.where((p_r[:, 0] > radius_skin))[0]

    cos_theta_a = np.cos(cathode_pos[1]) * np.cos(p_r[:, 1]) + \
                  np.sin(cathode_pos[1]) * np.sin(p_r[:, 1]) * \
                  np.cos(p_r[:, 2] - cathode_pos[2])
    cos_theta_b = np.cos(anode_pos[1]) * np.cos(p_r[:, 1]) + \
                  np.sin(anode_pos[1]) * np.sin(p_r[:, 1]) * \
                  np.cos(p_r[:, 2] - anode_pos[2])

    potential = np.zeros((p.shape[0]), dtype='float64')

    coefficients = np.zeros((nbr_polynomials, p.shape[0]), dtype='float64')

    # accelerate
    for ii in range(1, nbr_polynomials):
        n = float(ii)
        coefficients[ii, brain_region] = np.nan_to_num(
            A(n) * ((p_r[brain_region, 0] / radius_skin) ** n))

        coefficients[ii, skull_region] = np.nan_to_num(S(n) * (p_r[skull_region, 0] / radius_skin) ** n +
                                                       U(n) * (p_r[skull_region, 0] * radius_skin) ** (-n - 1))

        coefficients[ii, skin_region] = np.nan_to_num(T(n) * (p_r[skin_region, 0] / radius_skin) ** n
                                                      + W(n) * (p_r[skin_region, 0] * radius_skin) ** (-n - 1))

    potential[inside_sphere] = np.nan_to_num(
        np.polynomial.legendre.legval(cos_theta_a[inside_sphere], coefficients[:, inside_sphere], tensor=False) -
        np.polynomial.legendre.legval(cos_theta_b[inside_sphere], coefficients[:, inside_sphere], tensor=False))

    potential *= 1.0 / (2 * np.pi * conductivities[2] * radius_skin)

    potential[outside_sphere] = 0.0

    return potential
    # plot_scatter(points_cart[:,0],points_cart[:,1],points_cart[:,2],potential)


def calc_potential_homogeneous_dipole(sphere_radius, conductivity, dipole_pos, dipole_moment, detector_positions):
    """
    Calculate the surface potential generated by a dipole inside a homogeneous conducting sphere.

    Dezhong Yao, Electric Potential Produced by a Dipole in a Homogeneous
    Conducting Sphere

    potential = calc_potential_homogeneous_dipole(sphere_radius, conductivity, dipole_pos, dipole_moment,
                                                  detector_positions):

    Parameters
    ----------
    sphere_radius: float
        radius of sphere, in mm
    conductivity: float
        conductivity of medium, in S/m
    dipole_pos: [3 x 1] np.ndarray
        position of dipole, in mm
    dipole_moment: [3 x 1] np.ndarray
        moment of dipole, in C.m
    detector_positions: [n x 3] np.ndarray
        position of detectors, will be projected into the sphere surface, in mm

    Returns
    -------
    potential: [n x 1] np.ndarray
       potential at the points
    """
    detector_positions = np.atleast_2d(detector_positions)
    assert detector_positions.shape[1] == 3
    assert np.linalg.norm(dipole_pos) < sphere_radius

    sphere_radius = np.float128(sphere_radius * 1e-3)
    dipole_pos = np.array(dipole_pos, dtype=np.float128) * 1e-3
    dipole_moment = np.array(dipole_moment, dtype=np.float128)
    detector_positions = np.array(detector_positions, dtype=np.float128) * 1e-3

    R = sphere_radius
    r0 = np.linalg.norm(dipole_pos)
    r = np.linalg.norm(detector_positions, axis=1)
    rp = np.linalg.norm(dipole_pos - detector_positions, axis=1)

    if not np.allclose(r, R):
        warnings.warn('Some points are not in the surface!!')

    if np.isclose(r0, 0):
        cos_phi = np.zeros(len(detector_positions), dtype=np.float128)
    else:
        cos_phi = dipole_pos.dot(detector_positions.T) / \
                  (np.linalg.norm(dipole_pos) * np.linalg.norm(detector_positions, axis=1))
    second_term = 1. / (rp[:, None] * R ** 2) * \
                  (detector_positions + (detector_positions * r0 * cos_phi[:, None] - R * dipole_pos) /
                   (R + rp - r0 * cos_phi)[:, None])
    potential = dipole_moment.dot((2 * (detector_positions - dipole_pos) / (rp ** 3)[:, None] +
                           second_term).T).T
    potential /= 4 * np.pi * conductivity
    return potential


def calc_B_field_outside_sphere(sphere_radius, dipole_pos, dipole_moment, detector_positions):
    """
    Calculate the B field outside a sphere, does not depend on conductivity.
    Dipole in SI units, positions in mm

    J.Savras - Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem

    B = calc_B_field_outside_sphere(sphere_radius, dipole_pos, dipole_moment, detector_positions)

    Parameters
    ----------
    sphere_radius: float
        radius of sphere
    dipole_pos: [3 x 1] np.ndarray
        position of dipole
    dipole_moment: [3 x 1] np.ndarray
        moment of dipole
    detector_positions: [n x 3] np.ndarray
        position of detectors, must lie outside sphere

    Returns
    -------
    B: [N x 3] np.ndarray
        array with B fields in detector positions
    """

    pos = np.array(dipole_pos, dtype=float) * 1e-3
    moment = np.array(dipole_moment, dtype=float)
    detector = np.array(detector_positions, dtype=float) * 1e-3

    assert np.all(np.linalg.norm(detector_positions, axis=1) > sphere_radius), "All points must be outside the sphere"

    assert np.all(np.linalg.norm(dipole_pos) < sphere_radius), "Dipole must be outside sphere"

    B = np.zeros(detector_positions.shape, dtype=float)

    for ii, r in enumerate(detector):
        norm_r = np.linalg.norm(r)

        r_0 = pos
        norm_r0 = np.linalg.norm(pos)

        a = r - r_0
        norm_a = np.linalg.norm(a)

        F = norm_a * (norm_r * norm_a + norm_r ** 2 - r_0.dot(r))

        grad_F = (norm_r ** (-1) * norm_a ** 2 + norm_a ** (-1) * a.dot(r) + 2 * norm_a + 2 * norm_r) * r - \
                 (norm_a + 2 * norm_r + norm_a ** (-1) * a.dot(r)) * r_0

        B[ii, :] = (4 * np.pi * 1e-7) / \
                   (4 * np.pi * F ** 2) * (F * np.cross(moment, r_0) - np.dot(np.cross(moment, r_0), r) * grad_F)

    return B


def calc_fibonacci_sphere(nr_points, R=1):
    """
    Creates N points around evenly spread through a unit sphere.

    points = calc_fibonacci_sphere(nr_points, R=1)

    Parameters
    ----------
    nr_points: int
        number of points to be spread, must be odd
    R: float, optional, default=1
        radius of sphere

    Returns
    -------
    points: [N x 3] np.ndarray
        evenly spread points through a unit sphere
    """
    assert nr_points % 2 == 1, "The number of points must be odd"
    points = []
    # The golden ratio
    phi = (1 + math.sqrt(5)) / 2.
    N = int((nr_points - 1) / 2)
    for i in range(-N, N + 1):
        lat = math.asin(2 * i / nr_points)
        lon = 2 * math.pi * i / phi
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        points.append((x, y, z))

    points = np.array(points, dtype=float)

    return points


def calc_tms_E_field(dipole_pos, dipole_moment, didt, positions):
    """
    Calculate the E field in a sphere caused by external magnetic dipoles.
    Dipole in SI units, positions in mm
    Everything should be in SI
    Independent of conductivity, see references

    L. Heller and D. van Hulsteyn, Brain stimulation using electromagnetic sources: theoretical aspects

    E = calc_tms_E_field(dipole_pos, dipole_moment, didt, positions)

    Parameters
    ----------
    dipole_pos: [M x 3] np.ndarray
        position of dipoles, must be outside sphere
    dipole_moment: [m x 3] np.ndarray
        moment of dipoles
    didt: float
        variation rate of current in the coil
    positions: [N x 3] np.ndarray
        position where fields should be calculated, must lie inside sphere

    Returns
    -------
    E: [N x 3] np.ndarray
        array with E-fields at detector positions
    """
    if dipole_pos.shape != dipole_moment.shape:
        raise ValueError('List of dipole position and moments should have the same'
                         'lengths')
    mu0_4pi = 1e-7

    E = np.zeros(positions.shape, dtype=float)
    dp = np.atleast_2d(dipole_pos)
    dm = np.atleast_2d(dipole_moment)

    r1 = positions

    for m, r2 in zip(dm, dp):
        a = r2 - r1
        norm_a = np.linalg.norm(a, axis=1)[:, None]

        norm_r1 = np.linalg.norm(r1, axis=1)[:, None]
        norm_r2 = np.linalg.norm(r2)

        r2_dot_a = np.sum(r2 * a, axis=1)[:, None]
        F = norm_a * (norm_r2 * norm_a + r2_dot_a)
        grad_F = (norm_a ** 2 / norm_r2 + 2 * norm_a + 2 * norm_r2 + r2_dot_a / norm_a) \
                 * r2 - (norm_a + 2 * norm_r2 + r2_dot_a / norm_a) * r1
        E += -didt * mu0_4pi / F ** 2 * \
             (F * np.cross(r1, m) - np.cross(np.sum(m * grad_F, axis=1)[:, None] * r1, r2))

        # Why use -didt? Take a look at the appendix 1 of the reference. It says "negative
        # time rate of change"
    return E


def calc_potential_dipole_3layers(radii, cond_brain_scalp, cond_skull, dipole_pos, dipole_moment,
                                  surface_points, nbr_polynomials=100):
    """
    Calculates the electric potential in a 3-layered sphere caused by a dipole
    Calculations assumes dimensions in SI units

    Ary, James P., Stanley A. Klein, and Derek H. Fender.
    "Location of sources of evoked scalp potentials: corrections for skull and scalp thicknesses."
    Biomedical Engineering 28.6 (1981).
    eq. 2 and 2a

    Parameters
    ----------
    radii: [3]list
        radius of each of the 3 layers (innermost to outermost), in mm
    cond_brain_scalp: float
        conductivity of the brain and scalp layers, in S/m
    cond_skull: float
        conductivity of the skull layer, in S/m
    dipole_pos: [3 x 1] np.ndarray
        position of the dipole, in mm
    dipole_moment: [3 x 1] np.ndarray
        moment of dipole, in C x m
    surface_points: [N x 3] np.ndarray
        list of positions where the poteitial should be calculated, in mm
    nbr_polynomials: int
        number of of legendre polynomials to use (default = 100)

    Returns
    -------
    potential: [N x 1] np.ndarray
        values of the electric potential, in V
    """
    assert len(radii) == 3
    assert radii[0] < radii[1] and radii[1] < radii[2]
    assert len(dipole_moment) == 3
    assert len(dipole_pos) == 3
    assert surface_points.shape[1] == 3
    assert np.linalg.norm(dipole_pos) < radii[0], "Dipole must be inside inner sphere"

    xi = float(cond_skull) / float(cond_brain_scalp)
    R = float(radii[2] * 1e-3)
    f1 = float(radii[0] * 1e-3) / R
    f2 = float(radii[1] * 1e-3) / R
    b = np.linalg.norm(dipole_pos) * 1e-3 / R

    if not np.allclose(np.linalg.norm(surface_points, axis=1), R * 1e3):
        warnings.warn('Some points are not in the surface!!')

    if np.isclose(b, 0):
        r_dir = np.array(dipole_moment, dtype=float)
        r_dir /= np.linalg.norm(r_dir)
    else:
        r_dir = dipole_pos / np.linalg.norm(dipole_pos)
    m_r = np.dot(dipole_moment, r_dir)
    cos_alpha = surface_points.dot(r_dir) / R * 1e-3

    t_dir = dipole_moment - m_r * r_dir
    # if the dipole is radial only
    if np.isclose(np.linalg.norm(dipole_moment), np.abs(np.dot(r_dir, dipole_moment))):
        # try to set an axis in x, if the dipole is not in x
        if not np.allclose(np.abs(r_dir.dot([1, 0, 0])), 1):
            t_dir = np.array([1., 0., 0.], dtype=float)
        # otherwise, set it in y
        else:
            t_dir = np.array([0., 1., 0.], dtype=float)
        t_dir = t_dir - r_dir.dot(t_dir)
    t_dir /= np.linalg.norm(t_dir)
    t2_dir = np.cross(r_dir, t_dir)
    m_t = np.dot(dipole_moment, t_dir)
    beta = np.arctan2(surface_points.dot(t2_dir), surface_points.dot(t_dir))
    cos_beta = np.cos(beta)

    def d(n):
        d_n = ((n + 1) * xi + n) * ((n * xi) / (n + 1) + 1) + \
              (1 - xi) * ((n + 1) * xi + n) * (f1 ** (2 * n + 1) - f2 ** (2 * n + 1)) - \
              n * (1 - xi) ** 2 * (f1 / f2) ** (2 * n + 1)
        return d_n

    potential = np.zeros(surface_points.shape[0], dtype='float64')

    P = np.zeros((2, nbr_polynomials + 1, surface_points.shape[0]), dtype='float64')
    for ii, ca in enumerate(cos_alpha):
        P[:, :, ii], _ = sp.lpmn(1, nbr_polynomials, ca)

    for ii in range(1, nbr_polynomials + 1):
        n = float(ii)
        potential += np.nan_to_num(
            (2 * n + 1) / n * b ** (n - 1) * ((xi * (2 * n + 1) ** 2) / (d(n) * (n + 1))) *
            (n * m_r * P[0, ii, :] - m_t * P[1, ii, :] * cos_beta))
        # Why should it be a minus there?

    potential /= 4 * np.pi * cond_brain_scalp * R ** 2

    return potential
