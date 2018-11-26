# -*- coding: utf-8 -*-
"""
Functions and classes that provide data and methods with general usage in the pygpc package
"""

import numpy as np
import scipy.special
import scipy.stats
import sys
import math
import itertools
from .Visualization import plot_beta_pdf_fit


def display_fancy_bar(text, i, n_i, more_text=None):
    # TODO: improve?
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

    if i == '1':
        sys.stdout.write('\n')

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
    sys.stdout.write(" [%-40s] %d%%" % (
        '=' * int((float(i) + 0) / n_i * 100 / 2.5), float(i) / n_i * 100))
    sys.stdout.flush()
    if int(i) == n_i:
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


def get_normalized_rms_deviation(array, array_ref, x_axis=False):
    """
    Determine the normalized root mean square deviation between input data and reference data in [%].

    normalized_rms = get_normalized_rms(array, array_ref)

    Parameters
    ----------
    array: np.ndarray
        input data [ (x), y0, y1, y2 ... ]
    array_ref: np.ndarray
        reference data [ (x_ref), y0_ref, y1_ref, y2_ref ... ]
        if array_ref is 1D, all sizes have to match
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

    max_min_idx = np.isclose(np.max(data_ref, axis=0), np.min(data_ref, axis=0))
    delta = np.max(data_ref, axis=0) - np.min(data_ref, axis=0)

    if max_min_idx.any():
        delta[max_min_idx] = max(data_ref[max_min_idx])
    
    # determine normalized rms deviation and return
    normalized_rms = 100 * np.sqrt(1.0/n_points * np.sum((data - data_ref)**2, axis=0)) / delta

    return normalized_rms


def get_beta_pdf_fit(data, beta_tolerance=0, uni_interval=0, fn_plot=None):
    """
    Fit data to a beta distribution in the interval [a, b].

    beta_parameters, moments, p_value, uni_parameters = get_beta_pdf_fit(data, beta_tolerance=0, uni_interval=0)

    Parameters
    ----------
    data: ndarray of float
        Data to fit beta distribution on
    beta_tolerance: float, optional, default=0
        Tolerance interval to calculate the bounds of beta distribution
        from observed data, e.g. 0.2 (+-20% tolerance on observed max and min value)
    uni_interval: float, optional, default=0
        uniform distribution interval defined as fraction of beta distribution interval (e.g. 0.95 (95%))
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
    if beta_tolerance > 0:
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
    if uni_interval > 0:
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

    t = np.dot(array.conj().T, array)
    s = np.sqrt(np.diag(t))
    s_sqrt = np.diag(s)
    c = np.max(1.0*(t-s_sqrt)/np.outer(s, s))
    
    return c


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


def get_num_coeffs_sparse(order_dim_max, order_glob_max, order_inter_max, dim):
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

    Returns
    -------
    num_coeffs_sparse: int
        Number of gPC coefficients and polynomials
    """

    if type(order_dim_max) is list:
        order_dim_max = np.array(order_dim_max)

    if order_dim_max.size == 1:
        order_dim_max = order_dim_max * np.ones(dim)

    # generate multi-index list up to maximum order
    if dim == 1:
        poly_idx = np.array([np.linspace(0, order_dim_max, order_dim_max + 1)]).astype(int).transpose()
    else:
        poly_idx = get_multi_indices_max_order(int(dim), order_glob_max)

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


def get_multi_indices_max_order(dim, max_order):
    """
    Computes all multi-indices with a maximum overall order of max_order.

    multi_indices = get_multi_indices_max_order(length, max_order)

    Parameters
    ----------
    dim : int
        Number of random parameters (length of multi-index tuples)
    max_order : int
        Maximum order (over all parameters)

    Returns
    -------
    multi_indices: np.ndarray [n_basis x dim]
        Multi-indices for a classical maximum order gPC
    """

    multi_indices = []
    for i_max_order in range(max_order + 1):
        s = get_all_combinations(np.arange(dim + i_max_order - 1) + 1, dim - 1)

        m = s.shape[0]

        s1 = np.zeros([m, 1])
        s2 = (dim + i_max_order) + s1

        v = np.diff(np.hstack([s1, s, s2]))
        v = v - 1

        if i_max_order == 0:
            multi_indices = v
        else:
            multi_indices = np.vstack([multi_indices, v])

    return multi_indices.astype(int)
