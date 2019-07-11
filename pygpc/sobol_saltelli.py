# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import h5py
from scipy.stats import norm
from scipy.special import binom

##############################################################################
# The following code is based on the Sobol sequence generator by Frances
# Y. Kuo and Stephen Joe. The license terms are provided below.
#
# Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the names of the copyright holders nor the names of the
# University of New South Wales and the University of Waikato
# and its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
##############################################################################


def sobol_sampling(n, dim):
    """Generate (N x D) numpy array of Sobol sequence samples"""
    scale = 31
    result = np.zeros([n, dim])

    # load directions
    fn = os.path.join(os.path.dirname(__file__), "sobol_saltelli_directions.hdf5")

    with h5py.File(fn, "r") as f:
        directions_raw = []
        directions = []

        for key in f.keys():
            directions_raw.append(f[key][:].tolist())

        for i in range(len(directions_raw)):
            for d in directions_raw[i]:
                directions.append(d)

    if dim > len(directions) + 1:
        raise ValueError("Error in Sobol sequence: not enough dimensions")

    ll = int(math.ceil(math.log(n) / math.log(2)))

    if ll > scale:
        raise ValueError("Error in Sobol sequence: not enough bits")

    for i in range(dim):
        v = np.zeros(ll + 1, dtype=int)

        if i == 0:
            for j in range(1, ll + 1):
                v[j] = 1 << (scale - j)  # all m's = 1
        else:
            m = np.array(directions[i - 1], dtype=int)
            a = m[0]
            s = len(m) - 1

            # The following code discards the first row of the ``m`` array
            # Because it has floating point errors, e.g. values of 2.24e-314
            if ll <= s:
                for j in range(1, ll + 1):
                    v[j] = m[j] << (scale - j)
            else:
                for j in range(1, s + 1):
                    v[j] = m[j] << (scale - j)
                for j in range(s + 1, ll + 1):
                    v[j] = v[j - s] ^ (v[j - s] >> s)
                    for k in range(1, s):
                        v[j] ^= ((a >> (s - 1 - k)) & 1) * v[j - k]

        x = int(0)
        for j in range(1, n):
            x ^= v[index_of_least_significant_zero_bit(j - 1)]
            result[j][i] = float(x / math.pow(2, scale))

    return result


def index_of_least_significant_zero_bit(value):
    index = 1
    while (value & 1) != 0:
        value >>= 1
        index += 1

    return index


def saltelli_sampling(n_samples, dim, calc_second_order=True):
    """Generates model inputs using Saltelli's extension of the Sobol sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme.  Saltelli's scheme extends the Sobol sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations.  If
    calc_second_order is False, the resulting matrix has N * (D + 2)
    rows, where D is the number of parameters.  If calc_second_order is True,
    the resulting matrix has N * (2D + 2) rows.  These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate
    dim : int
        The number of dimensions
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    """
    dim = int(dim)
    n_samples = int(n_samples)
    dg = dim

    # How many values of the Sobol sequence to skip
    skip_values = 1000

    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sampling(n_samples + skip_values, 2 * dim)

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * dg + 2) * n_samples, dim])
    else:
        saltelli_sequence = np.zeros([(dg + 2) * n_samples, dim])
    index = 0

    for i in range(skip_values, n_samples + skip_values):

        # Copy matrix "A"
        for j in range(dim):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(dg):
            for j in range(dim):
                if j == k:
                    saltelli_sequence[index, j] = base_sequence[i, j + dim]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(dg):
                for j in range(dim):
                    if j == k:
                        saltelli_sequence[index, j] = base_sequence[i, j]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j + dim]

                index += 1

        # Copy matrix "B"
        for j in range(dim):
            saltelli_sequence[index, j] = base_sequence[i, j + dim]

        index += 1
    return saltelli_sequence


def get_sobol_indices_saltelli(y, dim, calc_second_order=True, num_resamples=100,
                               conf_level=0.95):
    """Perform Sobol Analysis on model outputs.

    Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.  If calc_second_order is
    True, the dictionary also contains keys 'S2' and 'S2_conf'.

    Parameters
    ----------
    y : numpy.array
        A NumPy array containing the model outputs
    dim : int
        Number of dimensions
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)

    References
    ----------
    .. [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
           mathematical models and their Monte Carlo estimates."  Mathematics
           and Computers in Simulation, 55(1-3):271-280,
           doi:10.1016/S0378-4754(00)00270-6.
    .. [2] Saltelli, A. (2002).  "Making best use of model evaluations to
           compute sensitivity indices."  Computer Physics Communications,
           145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.
    .. [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
           S. Tarantola (2010).  "Variance based sensitivity analysis of model
           output.  Design and estimator for the total sensitivity index."
           Computer Physics Communications, 181(2):259-270,
           doi:10.1016/j.cpc.2009.09.018.
    """

    if calc_second_order and y.shape[0] % (2 * dim + 2) == 0:
        n = int(y.shape[0] / (2 * dim + 2))
    elif not calc_second_order and y.shape[0] % (dim + 2) == 0:
        n = int(y.shape[0] / (dim + 2))
    else:
        raise RuntimeError("""
        Incorrect number of samples in model output file.
        Confirm that calc_second_order matches option used during sampling.""")

    if conf_level < 0 or conf_level > 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # normalize the model output
    y = (y - y.mean(axis=0)) / y.std(axis=0)

    a, b, ab, ba = separate_output_values(y, dim, n, calc_second_order)
    r = np.random.randint(n, size=(n, num_resamples))
    z = norm.ppf(0.5 + conf_level / 2)

    n_sobol = int(dim + binom(dim, 2))
    sobol = np.zeros((n_sobol, y.shape[1]))
    sobol_total = np.zeros((dim, y.shape[1]))
    sobol_total_conf = np.zeros((dim, y.shape[1]))
    sobol_conf = np.zeros((n_sobol, y.shape[1]))
    sobol_idx = [np.nan for _ in range(n_sobol)]
    sobol_idx_bool = np.zeros((n_sobol, dim)).astype(bool)

    # first and total order (+ confidence interval)
    i_sobol = 0
    for j in range(dim):
        sobol[j, :] = first_order(a, ab[:, j, :], b)
        sobol_conf[j, :] = z * first_order(a[r], ab[r, j], b[r]).std(ddof=1)
        sobol_total[j, :] = z * total_order(a, ab[:, j], b)
        sobol_total_conf[j, :] = z * total_order(a[r], ab[r, j], b[r]).std(ddof=1)
        sobol_idx[j] = np.array([j])
        sobol_idx_bool[j, j] = True
        i_sobol += 1

    # Second order (+ confidence interval)
    if calc_second_order:
        for j in range(dim):
            for k in range(j + 1, dim):
                sobol[i_sobol, :] = second_order(a, ab[:, j, :], ab[:, k, :], ba[:, j, :], b)
                sobol_conf[i_sobol, :] = z * second_order(a[r], ab[r, j], ab[r, k], ba[r, j], b[r]).std(ddof=1)
                sobol_idx[i_sobol] = np.array([j, k])
                sobol_idx_bool[i_sobol, [j, k]] = True
                i_sobol += 1

    return sobol, sobol_idx, sobol_idx_bool


def first_order(a, ab, b):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return np.mean(b * (ab - a), axis=0) / np.var(np.r_[a, b], axis=0)


def total_order(a, ab, b):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return 0.5 * np.mean((a - ab) ** 2, axis=0) / np.var(np.r_[a, b], axis=0)


def second_order(a, abj, abk, baj, b):
    # Second order estimator following Saltelli 2002
    vjk = np.mean(baj * abk - a * b, axis=0) / np.var(np.r_[a, b], axis=0)
    sj = first_order(a, abj, b)
    sk = first_order(a, abk, b)

    return vjk - sj - sk


def create_si_dict(dim, calc_second_order):
    # initialize empty dict to store sensitivity indices
    s = dict((k, np.zeros(dim)) for k in ('S1', 'S1_conf', 'ST', 'ST_conf'))

    if calc_second_order:
        s['S2'] = np.zeros((dim, dim))
        s['S2'][:] = np.nan
        s['S2_conf'] = np.zeros((dim, dim))
        s['S2_conf'][:] = np.nan

    return s


def separate_output_values(y, dim, n, calc_second_order):
    ab = np.zeros((n, dim, y.shape[1]))
    ba = np.zeros((n, dim, y.shape[1])) if calc_second_order else None
    step = 2 * dim + 2 if calc_second_order else dim + 2

    a = y[0:y.shape[0]:step, :]
    b = y[(step - 1):y.shape[0]:step, :]
    for j in range(dim):
        ab[:, j, :] = y[(j + 1):y.shape[0]:step, :]
        if calc_second_order:
            ba[:, j, :] = y[(j + 1 + dim):y.shape[0]:step, :]

    return a, b, ab, ba
