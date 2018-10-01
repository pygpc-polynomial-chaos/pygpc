# -*- coding: utf-8 -*-
"""
Functions and classes that provide data and methods with general usage in the pygpc package
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os
import sys
import math
import multiprocessing
from builtins import range
from multiprocessing import pool


class NoDaemonProcess(multiprocessing.Process):
    """
    Helper class to create a non daemonic process.
    From https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
        make 'daemon' attribute always return False
    """

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonicPool(pool.Pool):
    """
    Helper class to create a non daemonic pool.
    We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    because the latter is only a wrapper function, not a proper class.
    """

    Process = NoDaemonProcess


def display_fancy_bar(text, i, n_i, more_text=None):
    """
    Display a simple progess bar.
    Call for each iteration and start with i=1.

    Parameters
    ----------
    text: str
       text to display in front of actual iteration
    i: str or int
       actual iteration
    n_i: int
       number of iterations
    more_text: str, optional, default=None
       text that displayed at an extra line.

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
        print((cursor_two_up + erase_line))
        print(more_text)
    sys.stdout.write(text + i.zfill(fill_width) + " from " + str(n_i))
    # this prints [50-spaces], i% * =
    sys.stdout.write(" [%-40s] %d%%" % (
        '=' * int((float(i) + 1) / n_i * 100 / 2.5), float(i) / n_i * 100))
    sys.stdout.flush()
    if int(i) == n_i:
        print("")


def get_cartesian_product(array_list, cartesian_product=None):
    """
    Generate a cartesian product of input arrays.

    cartesian_product = get_cartesian_product(array_list, cartesian_product=None)

    Parameters
    ----------
    array_list : list of np.ndarray
        arrays to form the cartesian product with
    cartesian_product : np.ndarray
        array to write the cartesian product

    Returns
    -------
    cartesian_product : np.ndarray
        array to write the cartesian product
        (M, len(arrays))

    Examples
    --------
    cartesian(([1, 2, 3], [4, 5], [6, 7])) =

    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    array_list = [np.asarray(x) for x in array_list]
    dtype = array_list[0].dtype

    n = np.prod([x.size for x in array_list])
    if cartesian_product is None:
        cartesian_product = np.zeros([n, len(array_list)], dtype=dtype)

    m = n / array_list[0].size
    cartesian_product[:, 0] = np.repeat(array_list[0], m)
    if array_list[1:]:
        cartesian(array_list[1:], cartesian_product=cartesian_product[0:m, 1:])
        for j in range(1, array_list[0].size):
            cartesian_product[j * m:(j + 1) * m, 1:] = cartesian_product[0:m, 1:]
    return cartesian_product


def get_rotation_matrix(theta):
    """
    Generate rotation matrix from euler angles.

    rotation_matrix = get_rotation_matrix(theta)

    Parameters
    ----------
    theta : list of float
        list of euler angles

    Returns
    -------
    rotation_matrix : [3,3] np.ndarray
        rotation matrix computed from euler angles
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
        simple list
    index : list of integer
        list of indices to delete

    Returns
    -------
    input_list : list
        input list without entries specified in index
    """

    indices = sorted(index, reverse=True)
    for list_index in indices:
        del input_list[list_index]
    return input_list


def get_array_unique_rows(array):
    """
    Compute unique rows of array and delete rows that are linearly dependent.

    unique = get_array_unique_rows(array)

    Parameters
    ----------
    array: np.ndarray
        matrix with k linearly dependent rows

    Returns
    -------
    unique: np.ndarray
        matrix without k linearly dependent rows
    """

    unique, idx = np.unique(array.view(array.dtype.descr * array.shape[1]), return_index=True)
    return unique[np.argsort(idx)].view(array.dtype).reshape(-1, array.shape[1])


def get_set_combinations(array, number_elements):
    """
    Computes all k-tuples (e_1, e_2, ..., e_k) of combinations of the set of elements of the first row of the
    input matrix where e_n+1 > e_n

    combination_vectors = get_set_combinations(array, number_elements)

    Parameters
    ----------
    array: np.ndarray
        matrix containing a first row of input elements
    number_elements: int
        number of elements in tuple

    Returns
    -------
    combination_vectors : np.ndarray
        matrix of combination vectors
    """

    # array is numpy array [1 x nv]
    # number_elements is scalar
    nv = array.shape[1]
    if nv == number_elements:
        return array
    if nv < number_elements:
        return []

    d = nv - number_elements
    ny = d + 1

    for i in range(2, number_elements + 1):
        ny = ny + (1.0 * ny * d) / i

    combination_vectors = np.zeros([int(ny), int(number_elements)])

    index = np.linspace(1, number_elements, number_elements).astype(int)
    limit = np.append(np.linspace(nv - number_elements + 1, nv - 1, number_elements - 1), 0)
    a = int(1)

    while 1:
        b = int(a + nv - index[number_elements - 1])  # Write index for last column
        for i in range(1, number_elements):  # Write the left number_elements-1 columns
            combination_vectors[(a - 1):b, i - 1] = array[0, index[i - 1] - 1]

        # Write the number_elements.th column
        combination_vectors[(a - 1):b, number_elements - 1] = array[0, index[number_elements - 1] - 1:nv]
        a = b + 1  # Move the write pointer

        new_loop = np.sum(index < limit)
        if new_loop == 0:  # All columns are filled:
            break  # Ready!
        index[(new_loop - 1):number_elements] = index[new_loop - 1] + np.linspace(1, number_elements - new_loop + 1,
                                                                                     number_elements - new_loop + 1)
    return combination_vectors


def get_multi_indices(length, max_order):
    """
    Computes all multi-indices with a maximum overall order of max_order.

    multi_indices = get_multi_indices(length, max_order)

    Parameters
    ----------
    length : int
        length of multi-index tuples
    max_order : int
        maximum overall interaction order

    Returns
    -------
    multi_indices : np.ndarray
        matrix of multi-indices
    """

    multi_indices = []
    for i_max_order in range(max_order + 1):
        # Chose (length-1) the splitting points of the array [0:(length+max_order)]
        # 1:length+max_order-1
        s = get_set_combinations(np.linspace(1, length + i_max_order - 1, length + i_max_order - 1) *
                                 np.ones([1, length + i_max_order - 1]), length - 1)

        m = s.shape[0]

        s1 = np.zeros([m, 1])
        s2 = (length + i_max_order) + s1

        v = np.diff(np.hstack([s1, s, s2]))
        v = v - 1

        if i_max_order == 0:
            multi_indices = v
        else:
            multi_indices = np.vstack([multi_indices, v])

    return multi_indices.astype(int)


def get_normalized_rms(array, ref):
    """
    Determine the normalized root mean square deviation between input data and reference data in [%].

    normalized_rms = get_normalized_rms(array, ref)

    Parameters
    ----------
    array: np.ndarray
        input data [ (x), y0, y1, y2 ... ]
    ref: np.ndarray
        reference data [ (xref), yref ]
        if ref is 1D, all sizes have to match

    Returns
    -------
    normalized_rms: float
        normalized root mean square deviation
    """

    N_points = array.shape[0]
    
    # determine number of input arrays
    if ref.shape[1] == 2:
        N_data = array.shape[1]-1
    else:
        N_data = array.shape[1]
    
    # interpolate array on ref data if necessary
    if ref.shape[1] == 1:
        data = array
        data_ref = ref
    else:
        # crop reference if it is longer than the axis of the data
        array_ref = ref[(ref[:, 0] >= min(array[:, 0])) & (ref[:, 0] <= max(array[:, 0])), 0]
        data_ref = ref[(ref[:, 0] >= min(array[:, 0])) & (ref[:, 0] <= max(array[:, 0])), 1]
        
        data = np.zeros([len(array_ref), N_data])
        for i_data in range(N_data):
            data[:, i_data] = np.interp(array_ref, array[:, 0], array[:, i_data+1])

    if (max(data_ref) - min(data_ref)) == 0: 
        delta = max(data_ref)
    else:
        delta = max(data_ref) - min(data_ref)
    
    # determine normalized rms deviation and return
    normalized_rms = 100 * np.sqrt(1.0/N_points * np.sum((data - data_ref)**2, axis=0)) / delta
    return normalized_rms


def get_betapdf_fit(data, beta_tolerance=0, uni_intervall=0):
    """
    Fit data to a beta distribution in the interval [a, b].

    beta_parameters, moments, p_value, uni_parameters = get_betapdf_fit(data, beta_tolerance=0, uni_intervall=0)

    Parameters
    ----------
    data: np.ndarray
        data to fit
    beta_tolerance: float, optional, default=0
        tolerance interval to calculate the bounds of beta distribution
        from observed data, e.g. 0.2 (+-20% tolerance)
    uni_intervall: float, optional, default=0
        uniform distribution interval defined as fraction of
        beta distribution interval
        range: [0...1], e.g. 0.90 (90%)
    
    Returns
    -------
    beta_parameters: [4] list of float
        2 shape parameters and limits
        [p, q, a, b]
    moments: [4] list of float
        [data_mean, data_std, beta_mean, beta_std]
    p_value: float
        p-value of the Kolmogorov Smirnov test
    uni_parameters: [2] list of float
        limits a and b
        [a, b]
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

    # determine kernel density estimates using Gaussian kernel
    kde = scipy.stats.gaussian_kde(data, bw_method=0.05/data.std(ddof=1))
    kde_x = np.linspace(a_beta, b_beta, 100)
    kde_y = kde(kde_x)
    
    # perform Kolmogorov Smirnov test
    _, p_value = scipy.stats.kstest(data, "beta", [p_beta, q_beta, a_beta, ab_beta])

    beta_parameters = np.array([p_beta, q_beta, a_beta, b_beta])

    # determine limits of uniform distribution [a_uni, b_uni] covering the
    # interval uni_intervall of the beta distribution
    if uni_intervall > 0:
        a_uni = scipy.stats.beta.ppf((1 - uni_intervall) / 2, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)
        b_uni = scipy.stats.beta.ppf((1 + uni_intervall) / 2, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)
        uni_parameters = np.array([a_uni, b_uni])
    else:
        uni_parameters = None

    return beta_parameters, moments, p_value, uni_parameters


def mutcoh(array):
    """
    Calculate the mutual coherence of a matrix A. It can also be referred as the cosine
    of the smallest angle between two columns.
      
    mutual_coherence = mutcoh(array)
 
    Parameters
    ----------
    array: np.ndarray
        input matrix

    Returns
    -------
    mutual_coherence: float
    """

    t = np.dot(array.conj().T, array)
    s = np.sqrt(np.diag(t))
    s_sqrt = np.diag(s)
    mutual_coherence = np.max(1.0*(t-s_sqrt)/np.outer(s, s))
    
    return mutual_coherence


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


def vprint(message, verbose=True):
    """
    Function that prints out a message if verbose argument is true.

    vprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """
    if verbose:
        print(message)


def get_num_coeffs(order, dim):
    """
    Calculate the number of PCE coefficients by the used order and dimension.

    num_coeffs = (order+dim)! / (order! * dim!)

    num_coeffs = get_num_coeffs(order , dim)

    Parameters
    ----------
    order: int
        global order of expansion
    dim: int
        number of random variables

    Returns
    -------
    num_coeffs: int
        number of coefficients and polynomials
    """

    return scipy.special.factorial(order + dim) / (scipy.special.factorial(order) * scipy.special.factorial(dim))


def get_num_coeffs_sparse(order_dim_max, order_glob_max, order_inter_max, dim):
    """
    Calculate the number of PCE coefficients for a specific maximum order in each dimension order_dim_max,
    maximum order of interacting polynomials order_glob_max and the interaction order order_inter_max.

    num_coeffs_sparse = get_num_coeffs_sparse(order_dim_max, order_glob_max, order_inter_max, dim)

    Parameters
    ----------
    order_dim_max: int or np.ndarray
        maximum order in each dimension
    order_glob_max: int
        maximum global order of interacting polynomials
    order_inter_max: int
        interaction order
    dim: int
        number of random variables

    Returns
    -------
    num_coeffs_sparse: int
        number of coefficients and polynomials
    """

    order_dim_max = np.array(order_dim_max)

    if order_dim_max.size == 1:
        order_dim_max = order_dim_max * np.ones(dim)

    # generate multi-index list up to maximum order
    if dim == 1:
        poly_idx = np.array([np.linspace(0, order_dim_max, order_dim_max + 1)]).astype(int).transpose()
    else:
        poly_idx = get_multi_indices(int(dim), order_glob_max)

    for i_dim in range(dim):
        # add multi-indexes to list when not yet included
        if order_dim_max[i_dim] > order_glob_max:
            poly_add_dim = np.linspace(order_glob_max + 1, order_dim_max[i_dim], order_dim_max[i_dim] - (order_glob_max + 1) + 1)
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
    Calculate the probability density function of the beta distribution in the interval [a,b].

    pdf = (gamma(p)*gamma(q)/gamma(p+q).*(b-a)**(p+q-1))**(-1) *
              (x-a)**(p-1) * (b-x)**(q-1);

    pdf = get_pdf_beta(x, p, q, a, b)

    Parameters
    ----------
    x: np.ndarray
        values of random variable
    a: float
        min boundary
    b: float
        max boundary
    p: float
        parameter defining the distribution shape
    q: float
        parameter defining the distribution shape

    Returns
    -------
    pdf: np.ndarray
        probability density
    """
    return (scipy.special.gamma(p) * scipy.special.gamma(q) / scipy.special.gamma(p + q)
            * (b - a) ** (p + q - 1)) ** (-1) * (x - a) ** (p - 1) * (b - x) ** (q - 1)

# TODO:Refactor
# def get_reg_obj(fname, results_folder):
#     # if .yaml does exist: load from .yaml file
#     if os.path.exists(fname):
#         print(results_folder + ": Loading reg_obj from file: " + fname)
#         reg_obj = read_gpc_obj(fname)
#
#     # if not: create reg_obj, save to .yaml file
#     else:
#         # re-initialize reg object with appropriate number of grid-points
#         reg_obj = Reg(pdf_type,
#                       pdf_shape,
#                       limits,
#                       order * np.ones(dim),
#                       order_max=order,
#                       interaction_order=interaction_order_max,
#                       grid=grid_init,
#                       random_vars=random_vars)
#
#         write_gpc_obj(reg_obj, fname)
#
#     return reg_obj

# def plot(interactive=True, filename=None, xlabel="$x$", ylabel="$p(x)$"):
#     if not interactive:
#         plt.ioff()
#     else:
#         plt.ion()
#
#     plt.figure(1)
#     plt.clf()
#     plt.rc('text', usetex=True)
#     plt.rc('font', size=18)
#     ax = plt.gca()
#     # legendtext = [r"e-pdf", r"$\beta$-pdf"]
#     legendtext = [r"$\beta$-pdf"]
#
#     # plot histogram of data
#     n, bins, patches = plt.hist(data, bins=16, normed=1, color=[1, 1, 0.6], alpha=0.5)
#
#     # plot beta pdf (kernel density estimate)
#     # plt.plot(kde_x, kde_y, 'r--', linewidth=2)
#
#     # plot beta pdf (fitted)
#     beta_x = np.linspace(a_beta, b_beta, 100)
#     beta_y = scipy.stats.beta.pdf(beta_x, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)
#
#     plt.plot(beta_x, beta_y, linewidth=2, color=[0, 0, 1])
#
#     # plot uniform pdf
#     uni_y = 0
#     if uni_intervall > 0:
#         uni_x = np.hstack([a_beta, a_uni - 1E-6 * (b_uni - a_uni),
#                            np.linspace(a_uni, b_uni, 100), b_uni + 1E-6 * (b_uni - a_uni), b_beta])
#         uni_y = np.hstack([0, 0, 1.0 / (b_uni - a_uni) * np.ones(100), 0, 0])
#         plt.plot(uni_x, uni_y, linewidth=2, color='r')
#         legendtext.append("u-pdf")
#
#         # configure plot
#     plt.legend(legendtext, fontsize=18, loc="upper left")
#     plt.grid(True)
#     plt.xlabel(xlabel, fontsize=22)
#     plt.ylabel(ylabel, fontsize=22)
#     ax.set_xlim(a_beta - 0.05 * (b_beta - a_beta), b_beta + 0.05 * (b_beta - a_beta))
#     ax.set_ylim(0, 1.1 * max([max(n), max(beta_y[np.logical_not(beta_y == np.inf)]), max(uni_y)]))
#
#     if interactive > 0:
#         plt.show()
#
#     # save plot
#     if filename:
#         plt.savefig(filename + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.01 * 4)
#         plt.savefig(filename + ".png", format='png', bbox_inches='tight', pad_inches=0.01 * 4, dpi=600)
