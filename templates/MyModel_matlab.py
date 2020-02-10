import inspect
import numpy as np
import matlab.engine
from pygpc.AbstractModel import AbstractModel


class MyModel_matlab(AbstractModel):
    """
    MyModel evaluates something using Matlab. The parameters of the model (constants and random parameters)
    are stored in the dictionary p. Their type is defined during the problem definition.

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

    def __init__(self, fun_path):
        self.fun_path = fun_path
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, matlab_engine, process_id=None):

        # add path of Matlab function
        matlab_engine.addpath(self.fun_path, nargout=0)

        # convert input parameters to matlab format (only lists can be converted)
        x1 = matlab.double(np.array(self.p["x1"]).tolist())
        x2 = matlab.double(np.array(self.p["x2"]).tolist())
        x3 = matlab.double(np.array(self.p["x3"]).tolist())
        a = matlab.double(np.array(self.p["a"]).tolist())
        b = matlab.double(np.array(self.p["b"]).tolist())

        # call Matlab function
        y = matlab_engine.Ishigami(x1, x2, x3, a, b)

        # convert the output back to numpy and ensure that the output is [n_grid x n_out]
        y = np.array(y).transpose()

        if y.ndim == 0:
            y = np.array([[y]])
        elif y.ndim == 1:
            y = y[:, np.newaxis]

        # delete matlab engine after simulations because it can not be saved in the gpc object
        # del self.matlab_engine

        return y
