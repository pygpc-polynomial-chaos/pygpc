# -*- coding: utf-8 -*-
from pygpc.AbstractModel import AbstractModel
from pyrates.utility import grid_search
from pyrates.utility.visualization import plot_psd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from numba import njit


class PyRates_CNS_Model(AbstractModel):
    """
    MyModel evaluates something. The parameters of the model (constants and random parameters) are stored in the
    dictionary p. Their type is defined during the problem definition.

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        Parameter 1
    p["x2"]: float or ndarray of float [n_grid]
        Parameter 2
    p["x3"]: float or ndarray of float [n_grid]
        Parameter 3

    Returns
    -------
    y: ndarray of float [n_grid x n_out]
        Results of the n_out quantities of interest the gPC is conducted for
    additional_data: dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
        If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
    """

    def __init__(self, p, context):
        super(PyRates_CNS_Model, self).__init__(p, context)
        from pyrates.frontend import OperatorTemplate
        from pyrates.frontend import NodeTemplate, CircuitTemplate

        exc_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v) /tau',
                   'd/dt * v = (v^2 + eta + I_ext + (I_exc - I_inh)*tau - (PI*r*tau)^2) /tau',
                   'd/dt * I_exc = J*r + r_exc - I_exc/tau_exc',
                   'd/dt * I_inh =  r_inh - I_inh/tau_inh'
                   ]
        inh_syn = ['d/dt * r = (delta/(PI*tau) + 2.*r*v) /tau',
                   'd/dt * v = (v^2 + eta + I_ext + (I_exc - I_inh)*tau - (PI*r*tau)^2) /tau',
                   'd/dt * I_exc = r_exc - I_exc/tau_exc',
                   'd/dt * I_inh = J*r + r_inh - I_inh/tau_inh'
                   ]
        variables = {'delta': {'default': 1.0},
                     'tau': {'default': 1.0},
                     'eta': {'default': -2.5},
                     'J': {'default': 0.0},
                     'tau_exc': {'default': 1.0},
                     'tau_inh': {'default': 2.0},
                     'r': {'default': 'output'},
                     'v': {'default': 'variable'},
                     'I_exc': {'default': 'variable'},
                     'I_inh': {'default': 'variable'},
                     'I_ext': {'default': 'input'},
                     'r_exc': {'default': 'input'},
                     'r_inh': {'default': 'input'},
                     }

        op_exc_syn = OperatorTemplate(name='Op_exc_syn', path=None, equations=exc_syn, variables=variables)
        op_inh_syn = OperatorTemplate(name='Op_inh_syn', path=None, equations=inh_syn, variables=variables)

        pcs = NodeTemplate(name='PCs', path=None, operators=[op_exc_syn])
        eins = NodeTemplate(name='EINs', path=None, operators={op_exc_syn: {'eta': -0.5}})
        iins = NodeTemplate(name='IINs', path=None, operators={op_inh_syn: {'tau': 2.0, 'eta': -0.5}})

        self.jrc_template = CircuitTemplate(name='jrc_template', path=None,
                                            nodes={'PCs': pcs, 'EINs': eins, 'IINs': iins},
                                            edges=[('PCs/Op_exc_syn/r', 'EINs/Op_exc_syn/r_exc',
                                                    None, {'weight': 13.5}),
                                                   ('EINs/Op_exc_syn/r', 'PCs/Op_exc_syn/r_exc',
                                                    None, {'weight': 0.8 * 13.5}),
                                                   ('PCs/Op_exc_syn/r', 'IINs/Op_inh_syn/r_exc',
                                                    None, {'weight': 0.25 * 13.5}),
                                                   ('IINs/Op_inh_syn/r', 'PCs/Op_exc_syn/r_inh',
                                                    None, {'weight': 1.75 * 13.5})]
                                            )

    def validate(self):
        pass

    def simulate(self, process_id):

        # w_ein_pc = np.linspace(0.5, 2, 10) * 0.8 * 13.5
        # w_iin_pc = np.linspace(0.5, 2, 10) * 1.75 * 13.5

        T = 100.
        dt = 1e-3
        dts = 1e-2
        ext_input = np.random.uniform(3., 5., (int(T / dt), 1))

        # run PyRates with parameter combinations
        results = grid_search(deepcopy(self.jrc_template),
                              param_grid={'w_ep': self.p['w_ein_pc'], 'w_ip': self.p['w_iin_pc']},
                              param_map={'w_ep': {'var': [(None, 'weight')],
                                                  'edges': [('EINs.0', 'PCs.0', 0)]},
                                         'w_ip': {'var': [(None, 'weight')],
                                                  'edges': [('IINs.0', 'PCs.0', 0)]}},
                              simulation_time=T, dt=dt, sampling_step_size=dts,
                              inputs={('PCs.0', 'Op_exc_syn.0', 'I_ext'): ext_input},
                              outputs={'r': ('PCs.0', 'Op_exc_syn.0', 'r')},
                              init_kwargs={'vectorization': 'nodes', 'build_in_place': False},
                              permute_grid=False,
                              backend="numpy",  # "tensorflow"
                              decorator=njit)

        y = np.zeros((len(self.p['w_ein_pc']), 1))

        # extract QOI
        for idx, (we, wi) in enumerate(zip(self.p['w_ein_pc'], self.p['w_iin_pc'])):
            plot_psd(results[we][wi], tmin=30.0, show=False)
            p = plt.gca().get_lines()[-1].get_ydata()
            f = plt.gca().get_lines()[-1].get_xdata()
            max_idx = np.argmax(p)
            y[idx, 0] = f[max_idx]
            # y[idx, 1] = p[max_idx]
            plt.close(plt.gcf())

        return y


