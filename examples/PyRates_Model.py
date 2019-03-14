# -*- coding: utf-8 -*-
from pygpc.AbstractModel import AbstractModel
from pyrates.utility import grid_search
import numpy as np
import scipy.signal as sp


class MyModel(AbstractModel):
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
        super(MyModel, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        param_map = {'k_e': {'var': [('Op_e.0', 'k_ee'), ('Op_i.0', 'k_ie')],
                             'nodes': ['E.0', 'I.0']},
                     'k_i': {'var': [('Op_e.0', 'k_ei'), ('Op_i.0', 'k_ii')],
                             'nodes': ['E.0', 'I.0']}
                     }

        if type(self.p['k_e']) == float:
            self.p['k_e'] = [self.p['k_e']]
        if type(self.p['k_i']) == float:
            self.p['k_i'] = [self.p['k_i']]
        param_grid = {'k_e': self.p['k_e'], 'k_i': self.p['k_i']}

        simulation_time = 2.
        results = grid_search(circuit_template="/data/hu_salomon/PycharmProjects/PyRates/models/Montbrio/Montbrio.EI_Circuit",
                              param_map=param_map,
                              param_grid=param_grid,
                              dt=1e-5,
                              simulation_time=simulation_time,
                              inputs={},
                              outputs={"r": ("E", "Op_e.0", "r")},
                              sampling_step_size=1e-3, permute_grid=False)

        y = np.zeros((len(self.p["k_e"]), 1))
        for i_grid, (k_e, k_i) in enumerate(zip(self.p['k_e'], self.p['k_i'])):
            data = np.array(results[k_e][k_i])
            peaks = sp.argrelextrema(data, np.greater)
            y[i_grid, 0] = int(len(peaks[0]) / simulation_time)

        additional_data = [{"additional_data/info_1": [1, 2, 3],
                            "additional_data/info_2": ["some additional information"]}]
        additional_data = y.shape[0] * additional_data

        return y, additional_data
