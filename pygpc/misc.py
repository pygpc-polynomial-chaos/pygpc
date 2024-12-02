
import sys
import math
import copy
import random
import warnings
import itertools
import numpy as np
import scipy.stats
import scipy.special
import scipy.spatial

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from scipy.spatial.distance import cdist
from .Visualization import plot_beta_pdf_fit

try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except:
    pass


def squared_exponential_kernel(x, y, lengthscale, variance):
    """
    Computes the squared exponential kernel for Gaussian Processes.

    Parameters
    ----------
    x : np.ndarray of float [N x dim]
        Input observation locations
    y : np.ndarray of float [M x dim]
        Output observation locations
    lengthscale : float
        Lengthscale parameter
    variance : float
        Output variance

    Returns
    -------
    k : np.ndarray of float [M x X]
        Kernel function values (covariance function or covariance matrix)
    """
    sqdist = cdist(x, y, 'sqeuclidean')
    k = variance * np.exp(-0.5 * sqdist * (1/lengthscale**2))
    return k


def is_instance(obj):
    """
    Tests if obj is a class instance of any type.

    Parameters
    ----------
    obj : any
        Input object

    Returns
    -------
    out : bool
        Flag if obj is class instance or not
    """
    try:
        _ = obj.__dict__
        return True

    except AttributeError:
        return False


def display_fancy_bar(text, i, n_i, more_text=None):
    """
    Display a simple progress bar. Call in each iteration and start with i=1.

    Parameters
    ----------
    text: str
       Text to display in front of actual iteration
    i: str or int
       Actual iteration
    n_i: int
       Total number of iterations
    more_text: str, optional, default=None
       Text that is displayed on an extra line above the bar.

    Examples
    --------
    fancy_bar('Run',7,10):
    Run 07 from 10 [================================        ] 70%

    fancy_bar(Run,9,10,'Some more text'):
    Some more text
    Run 09 from 10 [======================================= ] 90%
    """

    if not isinstance(i, str):
        i = str(i)

    assert isinstance(text, str)
    assert isinstance(n_i, int)

    if not text.endswith(' '):
        text += ' '

    # if i == '1':
        # sys.stdout.write('')

    sys.stdout.write('\r')
    fill_width = len(str(n_i))

    # terminal codes, working on windows as well?
    cursor_two_up = '\x1b[2A'
    erase_line = '\x1b[2K'

    if more_text:
        print(cursor_two_up + erase_line)
        print(more_text)
    sys.stdout.write(text + i.zfill(fill_width) + " from " + str(n_i))
    # this prints [50-spaces], i% * =
    # sys.stdout.write(" [%-40s] %d%%" % (
    #     '=' * int((float(i) + 0) / n_i * 100 / 2.5), float(i) / n_i * 100.))
    sys.stdout.write(" [{}{}] {:.1f}%".format('=' * int(int(i) / n_i * 40),
                                              " " * (40 - int(int(i) / n_i * 40)),
                                              float(int(i) / n_i * 100)))
    sys.stdout.flush()
    # if int(i) == n_i:
    #     print("")
    print("")


def get_cartesian_product(array_list):
    """
    Generate a cartesian product of input arrays (all combinations).

    cartesian_product = get_cartesian_product(array_list)

    Parameters
    ----------
    array_list : list of 1D ndarray of float
        Arrays to compute the cartesian product with

    Returns
    -------
    cartesian_product : ndarray of float
        Array containing the cartesian products (all combinations of input vectors)
        (M, len(arrays))

    Examples
    --------
    >>> import pygpc
    >>> out = pygpc.get_cartesian_product(([1, 2, 3], [4, 5], [6, 7]))
    >>> out
    """

    cartesian_product = [element for element in itertools.product(*array_list)]
    return np.array(cartesian_product)


def get_rotation_matrix(theta):
    """
    Generate rotation matrix from euler angles.

    rotation_matrix = get_rotation_matrix(theta)

    Parameters
    ----------
    theta : list of float [3]
        Euler angles

    Returns
    -------
    rotation_matrix : ndarray of float [3, 3]
        Rotation matrix computed from euler angles
    """

    r_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    r_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    r_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    rotation_matrix = np.dot(r_z, np.dot(r_y, r_x))

    return rotation_matrix


def get_different_rows_from_matrices(a, b):
    """
    Compares rows from matrix a with rows from matrix b. It is assumed that b contains rows from a.
    The function returns the rows of b, which are not included in a.

    Parameters
    ----------
    a : ndarray of float [m1 x n]
        First matrix (usually the smaller one), where rows are part of b
    b : ndarray of float [m2 x n]
        Second matrix (usually the larger one), containing rows of a

    Returns
    -------
    b_diff : ndarray of float [m3 x n]
        Rows from b differing from a
    """
    idx_in = []

    for _a in a:
        for i_b, _b in enumerate(b):
            if (np.isclose(_a, _b, atol=1e-6)).all():
                idx_in.append(i_b)

    idx = np.arange(b.shape[0])
    idx_not_in = [i for i in idx if i not in idx_in]

    return b[idx_not_in, :]


def get_list_multi_delete(input_list, index):
    """
    Delete multiple entries from list.

    input_list = get_list_multi_delete(input_list, index)

    Parameters
    ----------
    input_list : list
        Simple list
    index : list of integer
        List of indices to delete

    Returns
    -------
    input_list : list
        Input list without entries specified in index
    """

    indices = sorted(index, reverse=True)
    for list_index in indices:
        del input_list[list_index]
    return input_list


def get_array_unique_rows(array):
    """
    Compute unique rows of 2D array and delete rows that are redundant.

    unique = get_array_unique_rows(array)

    Parameters
    ----------
    array: ndarray of float
        Matrix with k redundant rows

    Returns
    -------
    unique: ndarray of float
        Matrix without k redundant rows
    """

    _, index = np.unique(array, axis=0, return_index=True)
    index = np.sort(index)
    return array[index]


def nrmsd(array, array_ref, error_norm="relative", x_axis=False):
    """
    Determine the normalized root mean square deviation between input data and reference data.

    normalized_rms = get_normalized_rms(array, array_ref)

    Parameters
    ----------
    array: np.ndarray
        input data [ (x), y0, y1, y2 ... ]
    array_ref: np.ndarray
        reference data [ (x_ref), y0_ref, y1_ref, y2_ref ... ]
        if array_ref is 1D, all sizes have to match
    error_norm: str, optional, default="relative"
        Decide if error is determined "relative" or "absolute"
    x_axis: boolean, optional, default=False
        If True, the first column of array and array_ref is interpreted as the x-axis, where the data points are
        evaluated. If False, the data points are assumed to be at the same location.

    Returns
    -------
    normalized_rms: ndarray of float [array.shape[1]]
        Normalized root mean square deviation between the columns of array and array_ref
    """

    n_points = array.shape[0]

    if x_axis:
        # handle different array lengths
        if len(array_ref.shape) == 1:
            array_ref = array_ref[:, None]
        if len(array.shape) == 1:
            array = array[:, None]

        # determine number of input arrays
        if array_ref.shape[1] == 2:
            n_data = array.shape[1]-1
        else:
            n_data = array.shape[1]

        # interpolate array on array_ref data if necessary
        if array_ref.shape[1] == 1:
            data = array
            data_ref = array_ref
        else:
            # crop reference if it is longer than the axis of the data
            data_ref = array_ref[(array_ref[:, 0] >= min(array[:, 0])) & (array_ref[:, 0] <= max(array[:, 0])), 1]
            array_ref = array_ref[(array_ref[:, 0] >= min(array[:, 0])) & (array_ref[:, 0] <= max(array[:, 0])), 0]

            data = np.zeros([len(array_ref), n_data])
            for i_data in range(n_data):
                data[:, i_data] = np.interp(array_ref, array[:, 0], array[:, i_data+1])
    else:
        data_ref = array_ref
        data = array

    # determine "absolute" or "relative" error
    if error_norm == "relative":
        # max_min_idx = np.isclose(np.max(data_ref, axis=0), np.min(data_ref, axis=0))
        delta = np.max(data_ref, axis=0) - np.min(data_ref, axis=0)

        # if max_min_idx.any():
        #     delta[max_min_idx] = max(data_ref[max_min_idx])
    else:
        delta = 1

    # filter nan
    row_idx_data, col_idx_data = np.where(np.isnan(data))
    row_idx_data_ref, col_idx_data_ref = np.where(np.isnan(data_ref))
    row_idx = np.hstack((row_idx_data, row_idx_data_ref))
    col_idx = np.hstack((col_idx_data, col_idx_data_ref))

    if row_idx_data.size > 0:
        warnings.warn("nan in input dataset found at rows={} cols={} (ignored)".format(row_idx, col_idx))

    if row_idx_data_ref.size > 0:
        warnings.warn("nan in reference dataset found at rows={} cols={} (ignored)".format(row_idx_data_ref,
                                                                                            col_idx_data_ref))

    if row_idx.size > 0:
        data = np.delete(data, np.unique(row_idx), axis=0)
        data_ref = np.delete(data_ref, np.unique(row_idx), axis=0)

    # determine normalized rms deviation and return
    normalized_rms = np.sqrt(1.0/n_points * np.sum((data - data_ref)**2, axis=0)) / delta

    return normalized_rms


def get_beta_pdf_fit(data, beta_tolerance=0, uni_interval=0, fn_plot=None):
    """
    Fit data to a beta distribution in the interval [a, b].

    beta_parameters, moments, p_value, uni_parameters = get_beta_pdf_fit(data, beta_tolerance=0, uni_interval=0)

    Parameters
    ----------
    data: ndarray of float
        Data to fit beta distribution on
    beta_tolerance: float or list of float, optional, default=0
        Specifies bounds of beta distribution. If float, it calculates the tolerance
        from observed data, e.g. 0.2 (+-20% tolerance on observed max and min value).
        If list, it takes [min, max] as bounds [a, b].
    uni_interval: float or list of float, optional, default=0
        uniform distribution interval. If float, the bounds are defined as a fraction of the beta distribution
        interval (e.g. 0.95 (95%)). If list, it takes [min, max] as bounds [a, b].
    fn_plot: str
        Filename of plot so save (.pdf and .png)
    
    Returns
    -------
    beta_parameters: list of float [4]
        Two shape parameters and lower and upper limit [p, q, a, b]
    moments: list of float [4]
        Mean and std of raw data and fitted beta distribution [data_mean, data_std, beta_mean, beta_std]
    p_value: float
        p-value of the Kolmogorov Smirnov test
    uni_parameters: list of float [2]
        Lower and upper limits of uniform distribution [a, b]
    """
    
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    # fit beta distribution to data
    if type(beta_tolerance) is list:
        a_beta = beta_tolerance[0]
        b_beta = beta_tolerance[1]
        p_beta, q_beta, a_beta, ab_beta = scipy.stats.beta.fit(data, floc=a_beta, fscale=b_beta - a_beta)
    elif beta_tolerance > 0:
        # use user beta_tolerance of to set limits of distribution
        data_range = data.max()-data.min()
        a_beta = data.min()-beta_tolerance*data_range
        b_beta = data.max()+beta_tolerance*data_range
        p_beta, q_beta, a_beta, ab_beta = scipy.stats.beta.fit(data, floc=a_beta, fscale=b_beta-a_beta)
    else:
        # let scipy.stats.beta.fit determine the limits
        p_beta, q_beta, a_beta, ab_beta = scipy.stats.beta.fit(data)
        b_beta = a_beta + ab_beta
    
    beta_mean, beta_var = scipy.stats.beta.stats(p_beta, q_beta, loc=a_beta, scale=(b_beta-a_beta), moments='mv')
    beta_std = np.sqrt(beta_var)
    
    moments = np.array([data_mean, data_std, beta_mean, beta_std])

    # perform Kolmogorov Smirnov test
    _, p_value = scipy.stats.kstest(data, "beta", [p_beta, q_beta, a_beta, ab_beta])

    beta_parameters = np.array([p_beta, q_beta, a_beta, b_beta])

    # determine limits of uniform distribution [a_uni, b_uni] covering the
    # interval uni_interval of the beta distribution
    if type(uni_interval) is list:
        a_uni = uni_interval[0]
        b_uni = uni_interval[1]
        uni_parameters = np.array([a_uni, b_uni])
    elif uni_interval > 0:
        a_uni = scipy.stats.beta.ppf((1 - uni_interval) / 2, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)
        b_uni = scipy.stats.beta.ppf((1 + uni_interval) / 2, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)
        uni_parameters = np.array([a_uni, b_uni])
    else:
        a_uni = None
        b_uni = None
        uni_parameters = None

    if fn_plot is not None:
        plot_beta_pdf_fit(data=data,
                          a_beta=a_beta, b_beta=b_beta, p_beta=p_beta, q_beta=q_beta,
                          a_uni=a_uni, b_uni=b_uni,
                          interactive=True, fn_plot=fn_plot, xlabel="$x$", ylabel="$p(x)$")

    return beta_parameters, moments, p_value, uni_parameters


def mutual_coherence(array):
    """
    Calculate the mutual coherence of a matrix A. It can also be referred as the cosine of the smallest angle
    between two columns.

    mutual_coherence = mutual_coherence(array)

    Parameters
    ----------
    array: ndarray of float
        Input matrix

    Returns
    -------
    mutual_coherence: float
        Mutual coherence
    """

    array = array / np.linalg.norm(array, axis=0)[np.newaxis, :]
    t = np.dot(array.conj().T, array)
    np.fill_diagonal(t, 0.0)
    mu = np.max(t)

    return mu


def RIP(A, x):
    """
    Calculate the restricted isometric property constant delta of a matrix A for a given sparse vector x.

    delta = RIP(A, x)

    Parameters
    ----------
    A: ndarray of float
        Input matrix
    x: ndarray of float
        applied vector
    Returns
    -------
    delta: float
        restricted isometric property constant
    """

    norm_Ax = np.linalg.norm(np.dot(A, x))
    norm_x = np.linalg.norm(x)
    delta = (norm_Ax**2 / norm_x**2) - 1

    return delta


def wrap_function(fn, x, args):
    """
    Function wrapper to call anonymous function with variable number of arguments (tuple).

    wrap_function(fn, x, args)

    Parameters
    ----------
    fn: function
        anonymous function to call
    x: tuple
        parameters of function
    args: tuple
        arguments of function

    Returns
    -------
    function_wrapper: function
        wrapped function
    """

    def function_wrapper(*wrapper_args):
        return fn(*(wrapper_args + x + args))

    return function_wrapper


def get_num_coeffs(order, dim):
    """
    Calculate the number of gPC coefficients by the maximum order and the number of random variables.

    num_coeffs = (order+dim)! / (order! * dim!)

    num_coeffs = get_num_coeffs(order , dim)

    Parameters
    ----------
    order: int
        Maximum order of expansion
    dim: int
        Number of random variables

    Returns
    -------
    num_coeffs: int
        Number of gPC coefficients and polynomials
    """

    return scipy.special.factorial(order + dim) / (scipy.special.factorial(order) * scipy.special.factorial(dim))


def get_num_coeffs_sparse(order_dim_max, order_glob_max, order_inter_max, dim, order_inter_current=None,
                          order_glob_max_norm=1):
    """
    Calculate the number of gPC coefficients for a specific maximum order in each dimension "order_dim_max",
    global maximum order "order_glob_max" and the interaction order "order_inter_max".

    num_coeffs_sparse = get_num_coeffs_sparse(order_dim_max, order_glob_max, order_inter_max, dim)

    Parameters
    ----------
    order_dim_max: ndarray of int or list of int [dim]
        Maximum order in each dimension
    order_glob_max: int
        Maximum global order of interacting polynomials
    order_inter_max: int
        Interaction order
    dim: int
        Number of random variables
    order_inter_current : int

    order_glob_max_norm: float
        Norm, which defines how the orders are accumulated to derive the total order (default: 1-norm).
        Values smaller than one restrict higher orders and shrink the basis.

    Returns
    -------
    num_coeffs_sparse: int
        Number of gPC coefficients and polynomials
    """

    if order_inter_current is None:
        order_inter_current = order_inter_max

    if type(order_dim_max) is list:
        order_dim_max = np.array(order_dim_max)

    if order_dim_max.size == 1:
        order_dim_max = order_dim_max * np.ones(dim)

    # generate multi-index list up to maximum order
    if dim == 1:
        poly_idx = np.array([np.linspace(0, order_dim_max[0], int(order_dim_max[0] + 1))]).astype(int).transpose()
    else:
        poly_idx = get_multi_indices(order=order_dim_max,
                                     order_max=order_glob_max,
                                     interaction_order=order_inter_max,
                                     order_max_norm=order_glob_max_norm,
                                     interaction_order_current=order_inter_current)

    for i_dim in range(dim):
        # add multi-indexes to list when not yet included
        if order_dim_max[i_dim] > order_glob_max:
            poly_add_dim = np.linspace(order_glob_max + 1,
                                       order_dim_max[i_dim],
                                       order_dim_max[i_dim] - (order_glob_max + 1) + 1)
            poly_add_all = np.zeros([poly_add_dim.shape[0], dim])
            poly_add_all[:, i_dim] = poly_add_dim
            poly_idx = np.vstack([poly_idx, poly_add_all.astype(int)])

        # delete multi-indexes from list when they exceed individual max order of parameter
        elif order_dim_max[i_dim] < order_glob_max:
            poly_idx = poly_idx[poly_idx[:, i_dim] <= order_dim_max[i_dim], :]

    # Consider interaction order (filter out multi-indices exceeding it)
    poly_idx = poly_idx[np.sum(poly_idx > 0, axis=1) <= order_inter_max, :]

    return poly_idx.shape[0]


def get_pdf_beta(x, p, q, a, b):
    """
    Calculate the probability density function of the beta distribution in the interval [a, b].

    pdf = (gamma(p)*gamma(q)/gamma(p+q).*(b-a)**(p+q-1))**(-1) *
              (x-a)**(p-1) * (b-x)**(q-1);

    pdf = get_pdf_beta(x, p, q, a, b)

    Parameters
    ----------
    x: ndarray of float
        Values of random variable
    a: float
        lower boundary
    b: float
        upper boundary
    p: float
        First shape parameter defining the distribution
    q: float
        Second shape parameter defining the distribution

    Returns
    -------
    pdf: ndarray of float
        Probability density
    """
    return (scipy.special.gamma(p) * scipy.special.gamma(q) / scipy.special.gamma(p + q)
            * (b - a) ** (p + q - 1)) ** (-1) * (x - a) ** (p - 1) * (b - x) ** (q - 1)


def get_all_combinations(array, number_elements):
    """
    Compute all k-tuples (e_1, e_2, ..., e_k) of combinations of the set of elements of the input array where
    e_n+1 > e_n.
    combinations = get_all_combinations(array, number_elements)
    Parameters
    ----------
    array: np.ndarray
        Array to perform the combinatorial problem with
    number_elements: int
        Number of elements in tuple
    Returns
    -------
    combinations: np.ndarray
        Array of combination vectors
    """

    combinations = itertools.combinations(array, number_elements)
    return np.array([c for c in combinations])


def get_multi_indices(order, order_max, interaction_order, order_max_norm=1., interaction_order_current=None):
    """
    Computes all multi-indices with a maximum overall order of max_order considering a certain maximum order norm.

    multi_indices = get_multi_indices(length, max_order)

    Parameters
    ----------
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
        of polynomials in the expansion such that interaction terms are penalized more.
        sum(a_i^q)^1/q <= p, where p is order_max and q is order_max_norm (for more details see eq (11) in [1]).
    interaction_order : int
        Number of random variables, which can interact with each other
    interaction_order_current : int, optional, default: interaction_order
        Number of random variables currently interacting with respect to the highest order.
        (interaction_order_current <= interaction_order)
        The parameters for lower orders are all interacting with interaction_order.

    Returns
    -------
    multi_indices: ndarray [n_basis x dim]
        Multi-indices for a maximum order gPC assuming a certain order norm.
    """

    dim = len(order)

    order_max = int(order_max)
    order = [int(o) for o in order]

    if interaction_order_current is None or interaction_order_current > interaction_order:
        interaction_order_current = interaction_order
    else:
        interaction_order_current = interaction_order_current

    multi_indices = []
    for i_order_max in range(order_max + 1):
        s = get_all_combinations(np.arange(dim + i_order_max - 1) + 1, dim - 1)

        m = s.shape[0]

        s1 = np.zeros([m, 1])
        s2 = (dim + i_order_max) + s1

        v = np.diff(np.hstack([s1, s, s2]))
        v = v - 1

        if i_order_max == 0:
            multi_indices = v
        else:
            multi_indices = np.vstack([multi_indices, v])

    # remove polynomials exceeding order_max considering max_order_norm
    if order_max_norm != 1:
        multi_indices = multi_indices[np.linalg.norm(multi_indices, ord=order_max_norm, axis=1) <=
                                      (order_max + 1e-6), :]

    # add or delete monomials specified in order
    for i_dim in range(dim):
        # add multi-indexes to list when not yet included
        if order[i_dim] > order_max:
            multi_indices_add_dim = np.linspace(order_max + 1,
                                                order[i_dim],
                                                order[i_dim] - (order_max + 1) + 1)
            multi_indices_add_all = np.zeros([multi_indices_add_dim.shape[0], dim])
            multi_indices_add_all[:, i_dim] = multi_indices_add_dim
            multi_indices = np.vstack([multi_indices, multi_indices_add_all.astype(int)])

        # delete multi-indexes from list when they exceed individual max order of parameter
        elif order[i_dim] < order_max:
            multi_indices = multi_indices[multi_indices[:, i_dim] <= order[i_dim], :]

    # Consider interaction order (filter out multi-indices exceeding it)
    if interaction_order < dim:
        multi_indices = multi_indices[np.sum(multi_indices > 0, axis=1) <= interaction_order, :]

    # if interaction_order_current is smaller than interaction_order, delete those basis functions of highest order
    if interaction_order_current < interaction_order:
        mask_order_max = np.sum(multi_indices, axis=1) == order_max
        mask_interaction_order = np.sum(multi_indices > 0, axis=1) > interaction_order_current
        mask = np.logical_not(np.logical_and(mask_order_max, mask_interaction_order))
        multi_indices = multi_indices[mask]

    return multi_indices.astype(int)


def sample_sphere(n_points, r):
    """
    Creates n_points evenly spread in a sphere of radius r.

    Parameters
    ----------
    n_points: int
        Number of points to be spread, must be odd
    r: float
        Radius of sphere

    Returns
    -------
    points: ndarray of float [N x 3]
        Evenly spread points in a unit sphere
    """

    assert n_points % 2 == 1, "The number of points must be odd"
    points = []

    # The golden ratio
    phi = (1 + math.sqrt(5)) / 2.
    n = int((n_points - 1) / 2)

    for i in range(-n, n + 1):
        lat = math.asin(2 * i / n_points)
        lon = 2 * math.pi * i / phi
        x = r * math.cos(lat) * math.cos(lon)
        y = r * math.cos(lat) * math.sin(lon)
        z = r * math.sin(lat)
        points.append((x, y, z))

    points = np.array(points, dtype=float)

    return points


def mat2ten(mat, incr):
    """
    Transforms gPC gradient matrix or gradient grid points from matrix to tensor form

    Parameters
    ----------
    mat : ndarray of float [n_grid*incr, m]
        Matrix to transform
    incr : int
        Increment after every row, a new tensor slice is created

    Returns
    -------
    ten : ndarray of float [n_grid, m, incr]
        Tensor

    Notes
    -----
    Builds chunks after every "incr" row and writes it in a new slice [i, :, :]
    """

    ten = np.zeros((int(mat.shape[0]/incr), mat.shape[1], incr))
    idx = np.arange(0, mat.shape[0], incr)

    for i in range(incr):
        ten[:, :, i] = mat[idx + i, :]

    return ten


def ten2mat(ten):
    """
    Transforms gPC gradient tensor or gradient grid points from tensor to matrix form

    Parameters
    ----------
    ten : ndarray of float [n_grid, m, incr]
        Tensor to transform

    Returns
    -------
    mat : ndarray of float [n_grid*incr, m]
        Matrix

    Notes
    -----
    Stacks slices [i, :, :] vertically
    """
    mat = np.vstack([ten[i, :, :].transpose() for i in range(ten.shape[0])])

    return mat


def list2dict(l):
    """
    Transform list of dicts with same keys to dict of list

    Parameters
    ----------
    l : list of dict
        List containing dictionaries with same keys

    Returns
    -------
    d : dict of lists
        Dictionary containing the entries in a list
    """

    n = len(l)
    keys = l[0].keys()
    d = dict()

    for key in keys:
        d[key] = [0 for _ in range(n)]
        for i in range(n):
            d[key][i] = l[i][key]

    return d


def get_gradient_idx_domain(domains, d, gradient_idx):
    """
    Determine local gradient_idx in domain "d" from global gradient_idx

    Parameters
    ----------
    domains : ndarray of float [n_grid_global]
        Array containing the domain IDs
    d : int
        Domain ID for which the gradient index has to be computed for
    gradient_idx : ndarray of int [n_grid_global]
        Indices of grid points (global) where the gradient in gradient_results is provided

    Returns
    -------
    gradient_idx_local : ndarray of int [len(domains[domains==d])]
        Indices of grid points (local) where the gradient in gradient_results is provided
    """
    arr = np.arange(len(domains))
    gradient_idx_local = np.array([i
                                   for i, c in enumerate(arr[domains == d])  # local
                                   for cr in arr[gradient_idx]               # global
                                   if (c == cr).all()])

    return gradient_idx_local


def determine_projection_matrix(gradient_results, lambda_eps=0.95):
    """
    Determines projection matrix [P].

    .. math:: \\eta = [\\mathbf{P}] \\xi

    Parameters
    ----------
    gradient_results : ndarray of float [n_grid x n_out x dim]
        Gradient of model function in grid points
    lambda_eps : float, optional, default: 0.95
        Bound of principal components in %. All eigenvectors are included until lambda_eps of total sum of all
        eigenvalues is included in the system.

    Returns
    -------
    p_matrix : ndarray of float [dim_reduced x dim]
        Reduced projection matrix for QOI.
    p_matrix_complete : ndarray of float [dim_reduced x dim]
        Complete projection matrix for QOI.
    """

    # Determine projection matrices by SVD of gradients
    u, s, v = np.linalg.svd(gradient_results)

    # determine dominant eigenvalues up to lambda_eps * s_sum
    s_mask = [False]*len(s)
    s_sum_part = 0
    s_sum = np.sum(s)
    i_s = 0

    while s_sum_part <= lambda_eps*s_sum:
        s_sum_part += s[i_s]
        s_mask[i_s] = True
        i_s += 1

    s_filt = s[s_mask]
    v_filt = v[np.append(s_mask, [False]*(v.shape[0]-u.shape[1])) > 0, :]
    p_matrix = v_filt
    p_matrix_complete = v

    return p_matrix, p_matrix_complete


def get_indices_of_k_smallest(arr, k):
    """
    Find indices of k smallest elements in ndarray

    Parameters
    ----------
    arr : ndarray of float
        Array
    k : int
        Number of smallest values to extract

    Returns
    -------
    idx : tuple of ndarray [k]
        Indices of k smallest elements in array
    """
    # index = np.argpartition(arr.ravel(), k)
    # idx = tuple(np.array(np.unravel_index(index, arr.shape))[:, range(min(k, 0), max(k, 0))])
    #
    # return idx

    idx = np.argpartition(arr.ravel(), arr.size - k)[:k]

    return tuple(np.unravel_index(idx, arr.shape))


def get_coords_discontinuity(classifier, x_min, x_max, n_coords_disc=10, border_sampling="structured"):
    """
    Determine n_coords_disc grid points close to discontinuity

    Parameters
    ----------
    classifier : Classifier object
        Classifier object to predict classes from coordinates (needs to contain a classifier.predict() method)
    x_min : ndarray of float [n_dim]
        Minimal values in parameter space to search discontinuity
    x_max : ndarray of float [n_dim]
        Maximal values in parameter space to search discontinuity
    n_coords_disc : int, optional, default: 10
        Number of grid points to determine close to discontinuity
    border_sampling : str, optional, default: "structured"
        Sampling method to determine location of discontinuity


    Returns
    -------
    coords_disc : ndarray of float [n_coords_disc]
    """

    # if border_sampling == "random":
    #     coords_border_det = grid_learn_cluster.coords
    #     domains = model_kmeans.labels_
    dim = len(x_min)

    # create tensored mesh to find discontinuity
    if border_sampling == "structured":
        n_samples = int(1E4**(1./dim))
        eval_str = "np.array(np.meshgrid("

        for i in range(dim):
            eval_str += "np.linspace(x_min[{}], x_max[{}], {})".format(i, i, n_samples)

            if i < dim-1:
                eval_str += ", "

        eval_str += ")).T.reshape(-1, {})".format(dim)

        coords_border_det = eval(eval_str)

        domains = classifier.predict(coords_border_det)
    else:
        raise NotImplementedError("Please use valid border sampling method (""structured"")")

    # determine mask of not equaling domain points
    # dom_mat = np.tile(domains[:, np.newaxis], (1, domains.shape[0]))
    dom_mat = np.broadcast_to(domains, (len(domains), len(domains))).T
    mask = dom_mat != dom_mat.transpose()
    mask = np.tril(mask) * False + np.triu(mask)

    # determine distances between grid points in different domains
    distance_matrix = np.ones(mask.shape) * np.nan
    for i_c, c in enumerate(coords_border_det):
        distance_matrix[i_c, mask[i_c, :]] = np.linalg.norm(coords_border_det[mask[i_c, :], :] - c, axis=1)

    np.fill_diagonal(distance_matrix, np.nan)

    # find n_smallest distances and determine midpoints between those points
    n_smallest = 1000
    idx = get_indices_of_k_smallest(distance_matrix, n_smallest)
    coords_border = (coords_border_det[idx[0], :] + coords_border_det[idx[1], :]) / 2

    # resample n_coords_disc equally distributed points from n_smallest points on discontinuity
    n_reps = 1000  # number of repetitions
    idx = np.zeros((n_coords_disc, n_reps))
    distance_mean = np.zeros(n_reps)

    # select n_coords_disc points randomly and determine average minimal distance between points
    for i in range(n_reps):
        # sample n_resample points from coords_border
        idx[:, i] = random.sample(list(range(coords_border.shape[0])), n_coords_disc)

        # determine all to all distances
        distance_matrix_resample = np.ones((n_coords_disc, n_coords_disc)) * np.nan

        for i_c, c in enumerate(coords_border[idx[:, i].astype(int), :]):
            distance_matrix_resample[i_c, :] = np.linalg.norm(coords_border[idx[:, i].astype(int), :] - c, axis=1)

        np.fill_diagonal(distance_matrix_resample, 1000)
        distance_mean[i] = np.mean(np.min(distance_matrix_resample, axis=1))

    coords_disc = coords_border[idx[:, np.argmax(distance_mean)].astype(int), :]

    return coords_disc


def increment_basis(order_current, interaction_order_current, interaction_order_max, incr):
    """
    Increments basis

    Parameters
    ----------
    order_current: int
        Maximum global expansion order.
        The maximum expansion order considers the sum of the orders of combined polynomials together with the
        chosen norm "order_max_norm". Typically this norm is 1 such that the maximum order is the sum of all
        monomial orders.
    interaction_order_current : int
        Current number of random variables, which can interact with each other
        All polynomials are ignored, which have an interaction order greater than specified
    interaction_order_max : int
        Maximum number of random variables, which can interact with each other
        All polynomials are ignored, which have an interaction order greater than specified
    incr : int
        Number of sub-iteration increments

    Returns
    -------
    order : int
        Updated order
    interaction_order : int
        Updated interaction order
    """
    while incr > 0:
        if interaction_order_current + 1 <= min(order_current, interaction_order_max):
            interaction_order_current += 1
        else:
            order_current += 1
            interaction_order_current = 1

        incr -= 1

    return order_current, interaction_order_current


def compute_chunks(seq, num):
    """
    Splits up a sequence _seq_ into _num_ chunks of similar size.
    If len(seq) < num, (num-len(seq)) empty chunks are returned so that len(out) == num

    Parameters
    ----------
    seq : list of something [N_ele]
        List containing data or indices, which is divided into chunks
    num : int
        Number of chunks to generate

    Returns
    -------
    out : list of num sublists
        num sub-lists of seq with each of a similar number of elements (or empty).
    """
    assert len(seq) > 0
    assert num > 0
    # assert isinstance(seq, list), f"{type(seq)} can't be chunked. Provide list."

    avg = len(seq) / float(num)
    n_empty = 0  # if len(seg) < num, how many empty lists to append to return?

    if avg < 1:
        avg = 1
        n_empty = num - len(seq)

    out = []
    last = 0.0

    while last < len(seq):
        # if only one element would be left in the last run, add it to the current
        if (int(last + avg) + 1) == len(seq):
            last_append_idx = int(last + avg) + 1
        else:
            last_append_idx = int(last + avg)

        out.append(seq[int(last):last_append_idx])

        if (int(last + avg) + 1) == len(seq):
            last += avg + 1
        else:
            last += avg

    # append empty lists if len(seq) < num
    out += [[]] * n_empty

    return out


def t_averaged_mutual_coherence(array, t=0.2):
    """
    Computes the t-averaged mutual coherence.

    Parameters
    ----------
    array : ndarray of float [m x n]
        Matrix
    t : float
        Threshold

    Returns
    -------
    res : float
        t-averaged mutual coherence
    """
    array = np.abs(array)
    mask = array > t

    return np.sum(array[mask]) / np.sum(mask)


def average_cross_correlation_gram(array):
    """
    Computes the average cross correlation of the gram matrix.

    Parameters
    ----------
    array : ndarray of float [m x n]
        Gram matrix

    Returns
    -------
    res : float
        cross correlation
    """
    k = array.shape[1]
    n = k * (k - 1)

    return (1 / n) * (np.linalg.norm(np.identity(k) - array) ** 2)


def PhiP(x, p=10):
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


def poly_expand(current_set, to_expand, order_max, interaction_order):
    """
    Algorithm by Gerstner and Griebel to expand polynomial basis [1] according to two criteria:
        (1) The basis function may not be completely enclosed by already existing basis functions. In this case the added
            basis would be already included in any case.
        (2) The basis function is not a candidate if adding any basis would have no predecessors. The new basis must have
            predecessors in all decreasing direction and may not "float".

    Parameters
    ----------
    current_set : ndarray of int [n_basis, dim]
        Array of multi-indices of basis functions
    to_expand : ndarray of int [dim]
        Selected basis function (with highest gPC coefficient), which will be expanded in all possible direction
    order_max : int
        Maximal accumulated order (sum over all parameters)
    interaction_order : int
        Allowed interaction order between variables (<= dim)

    Returns
    -------
    expand : ndarray of int [n_basis, dim]
        Array of multi-indices, which will be added to the set of basis functions

    Notes
    -----
    .. [1] T Gerstner and M Griebel. Dimension adaptive tensor product quadrature. Computing, 71:65–87, 2003.
    """

    if type(current_set) is not list:
        current_set = current_set.tolist()

    expand = []
    for e in range(len(to_expand)):
        forward = copy.deepcopy(to_expand)
        forward[e] += 1
        has_predecessors = True
        for e2 in range(len(to_expand)):
            if forward[e2] > 0:
                predecessor = forward.copy()
                predecessor[e2] -= 1
                has_predecessors *= list(predecessor) in current_set
        if has_predecessors and (np.sum(np.abs(forward)) <= order_max) and (np.sum(forward > 0) <= interaction_order):
            expand += [tuple(forward)]

    return np.array(expand)


def get_non_enclosed_multi_indices(multi_indices, interaction_order):
    """
    Extract possible multi-indices from a given set which are potential candidates for anisotropic basis extension.
    Two criteria must be met:
    (1) The basis function may not be completely enclosed by already existing basis functions. In this case the added
        basis would be already included in any case.
    (2) The basis function is not a candidate if adding any basis would have no predecessors. The new basis must have
        predecessors in all decreasing direction and may not "float".

    Parameters
    ----------
    multi_indices : ndarray of float [n_basis, dim]
        Array of multi-indices of basis functions
    interaction_order : int
        Allowed interaction order between variables (<= dim)

    Returns
    -------
    multi_indices_non_enclosed : ndarray of float [n_basis_candidates, dim]
        Array of possible multi-indices of basis functions
    poly_indices_non_enclosed : list of int
        Indices of selected basis functions in global "multi-indices" array
    """
    multi_indices_non_enclosed = []

    if type(multi_indices) is not list:
        multi_indices = multi_indices.tolist()

    dim = len(multi_indices[0])

    for m in np.array(multi_indices):
        for i_dim in range(dim):
            m_test = copy.deepcopy(np.array(m))
            m_test[i_dim] = m_test[i_dim] + 1
            has_predecessors = True

            if np.sum(m_test > 0) <= interaction_order:
                if list(m_test) not in multi_indices:
                    for e2 in range(dim):
                        if m_test[e2] > 0:
                            predecessor = copy.deepcopy(m_test)
                            predecessor[e2] = predecessor[e2] - 1
                            has_predecessors *= list(predecessor) in multi_indices
                    if has_predecessors:
                        multi_indices_non_enclosed.append(m)

    multi_indices_non_enclosed = np.unique(multi_indices_non_enclosed, axis=0)
    poly_indices_non_enclosed = [np.where((multi_indices == m).all(axis=1))[0][0] for m in multi_indices_non_enclosed]

    return multi_indices_non_enclosed, poly_indices_non_enclosed


def plot_basis_by_multiindex(a):
    """

    Parameters
    ----------
    a

    Returns
    -------

    """
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(a)):
        e = 0.3
        x = a[i][0] + e
        y = a[i][1] + e
        z = a[i][2] + e
        vertices = [[(x + e, y + e, z + e), (x + e, y + e, z - e), (x + e, y - e, z - e), (x + e, y - e, z + e)],
                    [(x - e, y + e, z + e), (x - e, y + e, z - e), (x - e, y - e, z - e), (x - e, y - e, z + e)],
                    [(x + e, y + e, z + e), (x + e, y + e, z - e), (x - e, y + e, z - e), (x - e, y + e, z + e)],
                    [(x + e, y - e, z + e), (x + e, y - e, z - e), (x - e, y - e, z - e), (x - e, y - e, z + e)],
                    [(x + e, y + e, z + e), (x + e, y - e, z + e), (x - e, y - e, z + e), (x - e, y + e, z + e)],
                    [(x + e, y + e, z - e), (x + e, y - e, z - e), (x - e, y - e, z - e), (x - e, y + e, z - e)]]
        poly = Poly3DCollection(vertices, alpha=0.5)
        ax.add_collection3d(poly)

        for j in range(6):
            line_xs = np.hstack((np.array(vertices[j]).T[0], np.array(vertices[j]).T[0][0]))
            line_ys = np.hstack((np.array(vertices[j]).T[1], np.array(vertices[j]).T[1][0]))
            line_zs = np.hstack((np.array(vertices[j]).T[2], np.array(vertices[j]).T[2][0]))
            plt.plot(line_xs, line_ys, line_zs, color='k')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    plt.show()


def poly_expand_SimNIBS(active_set, old_set, to_expand, order_max, interaction_max):
    '''Algorithm by Guiherme B. Saturnino'''
    active_set  = [tuple(a) for a in active_set]
    old_set = [tuple(o) for o in old_set]
    to_expand = tuple(to_expand)
    active_set.remove(to_expand)
    old_set += [to_expand]
    expand = []

    for e1 in range(len(to_expand)):
        forward = np.asarray(to_expand, dtype=int)
        forward[e1] += 1
        has_predecessors = True
        for e2 in range(len(to_expand)):
            if forward[e2] > 0:
                predecessor = forward.copy()
                predecessor[e2] -= 1
                predecessor = tuple(predecessor)
                has_predecessors *= predecessor in old_set
        if has_predecessors and np.sum(np.abs(forward)) <= order_max \
            and np.sum(forward > 0) <= interaction_max:
            expand += [tuple(forward)]
            active_set += [tuple(forward)]

    return active_set, old_set, np.array(expand)

def choose_to_expand(multi_indices, active_set, old_set, coeffs, order_max, interaction_max):

    active_set = [tuple(a) for a in active_set]
    coeffs = np.linalg.norm(coeffs, axis=1)
    for idx in multi_indices[coeffs.argsort()[::-1]]:
        if tuple(idx) in active_set:
            _, _, expand = poly_expand_SimNIBS(active_set=active_set,
                                               old_set=old_set,
                                               to_expand=tuple(idx),
                                               order_max=order_max,
                                               interaction_max=interaction_max)
            if len(expand) > 0:
                return tuple(idx)

    raise ValueError('Could not find a polynomial in the expansion and in the active set')
