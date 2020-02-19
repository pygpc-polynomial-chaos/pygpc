import numpy as np
import inspect
from pygpc.AbstractModel import AbstractModel


class MyModel(AbstractModel):
    """
    MyModel evaluates something. The parameters of the model (constants and random parameters) are stored in the
    dictionary p. Their type is defined during the problem definition.

    Parameters
    ----------
    p["x1"] : float or ndarray of float [n_grid]
        Parameter 1
    p["x2"] : float or ndarray of float [n_grid]
        Parameter 2
    p["x3"] : float or ndarray of float [n_grid]
        Parameter 3

    Returns
    -------
    y : ndarray of float [n_grid x n_out]
        Results of the n_out quantities of interest the gPC is conducted for
    additional_data : dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
        If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
    """

    def __init__(self):
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = self.p["x1"] * self.p["x2"] * self.p["x3"]
        y = y[:, np.newaxis]

        additional_data = [{"additional_data/info_1": [1, 2, 3],
                            "additional_data/info_2": ["some additional information"]}]
        additional_data = y.shape[0] * additional_data

        return y, additional_data
