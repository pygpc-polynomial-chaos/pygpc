# -*- coding: utf-8 -*-
from pygpc.AbstractModel import AbstractModel
from pyrates.utility import grid_search
from pyrates.utility.visualization import plot_psd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from numba import njit
import sys
import io
import os


# auxiliary function to determine QOI
def get_psd(data, tmin=0.):
    # Compute spectrum
    try:
        dt = data.index[1] - data.index[0]
        n = data.shape[0]
        # Get closest power of 2 that includes n for zero padding
        n_two = 1 if n == 0 else 2 ** (n - 1).bit_length()

        freqs = np.linspace(0, 1 / dt, n_two)
        spec = np.fft.fft(data.loc[tmin:, :] - np.mean(data.loc[tmin:, :]), n=n_two, axis=0)
        return freqs, spec

    except IndexError:
        return np.NaN, np.NaN


def psd(data, tmin=0., tmax=None, **kwargs):
    # prepare data frame
    dt = data.index[1] - data.index[0]
    tmin = int(tmin / dt)
    tmax = data.shape[0] + 1 if tmax is None else max([int(tmax/dt), data.shape[0] + 1])
    if len(data.shape) > 1:
        data = data.iloc[tmin:tmax, :]
    else:
        data = data.iloc[tmin:tmax]

    # Compute power spectral density
    try:
        from scipy.signal import welch
        return welch(data.values, fs=1/dt, axis=0, **kwargs)
    except IndexError:
        return np.NaN, np.NaN


class PyRates_CNS_Model(AbstractModel):
    """
    PyRates example model

    Parameters
    ----------
    self.p['w_ein_pc'] : ndarray of float [n_grid]
        Excitatory connectivity weight
    self.p['w_iin_pc'] : ndarray of float [n_grid]
        Inhibitory connectivity weight

    Returns
    -------
    y: ndarray of float [n_grid x n_out]
        Results of the n_out quantities of interest the gPC is conducted for
    additional_data: dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
        If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id, matlab_engine=None):
        T = 10.
        dt = 1e-3
        dts = 1e-2
        ext_input = np.random.uniform(3., 5., (int(T / dt), 1))

        # sys.stdout = io.StringIO()

        # run PyRates with parameter combinations
        results, result_map = grid_search(circuit_template=f'{os.path.dirname(__file__)}/PyRates_model_template/JRC',
                                          param_grid={'w_ep': self.p['w_ein_pc'], 'w_ip': self.p['w_iin_pc']},
                                          param_map={'w_ep': {'vars': ['weight'],
                                                              'edges': [('EIN', 'PC')]},
                                                     'w_ip': {'vars': ['weight'],
                                                              'edges': [('IIN', 'PC')]}},
                                          simulation_time=T, dt=dt, sampling_step_size=dts,
                                          inputs={'PC/Op_exc_syn/I_ext': ext_input},
                                          outputs={'r': 'PC/Op_exc_syn/r'},
                                          init_kwargs={'vectorization': True},
                                          permute_grid=False,
                                          backend="numpy",  # "tensorflow"
                                          decorator=njit)

        y = np.zeros((len(self.p['w_ein_pc']), 1))

        # extract QOI
        for idx, (we, wi) in enumerate(zip(self.p['w_ein_pc'], self.p['w_iin_pc'])):
            res_idx = result_map.loc[(result_map == (we, wi)).all(1), :].index
            # plot_psd
            # plot_psd(results[we][wi], tmin=30.0, show=False)
            # p = plt.gca().get_lines()[-1].get_ydata()
            # f = plt.gca().get_lines()[-1].get_xdata()
            # max_idx = np.argmax(p)
            # y[idx, 0] = f[max_idx]
            # # y[idx, 1] = p[max_idx]
            # plt.close(plt.gcf())

            # welch
            # f, p = psd(data=results[we][wi], nperseg=4096, tmin=30.0)
            # max_idx = np.argmax(p)
            # y[idx, 0] = f[max_idx]
            # y[idx, 1] = p[max_idx]

            # fft
            # f, p = psd(data=results[res_idx], nperseg=4096, tmin=30.0)
            f, p = get_psd(data=results[res_idx], tmin=30.0)
            p = p[:int(len(p) / 2)]
            f = f[np.argmax(np.abs(p))]
            y[idx, 0] = f
            # y[idx, 1] = p

        sys.stdout = sys.__stdout__

        return y

