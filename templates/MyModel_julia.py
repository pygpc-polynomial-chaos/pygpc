import inspect
import numpy as np
from pygpc.AbstractModel import AbstractModel

try:
    from julia import Main
except ImportError:
    pass

class MyModel_julia(AbstractModel):
    """
    MyModel evaluates something by loading a julia file that contains a function. The parameters of the model
     (constants and random parameters) are stored in the dictionary p. Their type is defined during the problem
      definition.

    Parameters
    ----------
    fname_julia : str
        Filename of julia function
    p["x1"] : float or ndarray of float [n_grid]
        Parameter 1
    p["x2"] : float or ndarray of float [n_grid]
        Parameter 2
    p["x3"] : float or ndarray of float [n_grid]
        Parameter 3
    p["a"] : float
        shape parameter (a=7)
    p["b"] : float
        shape parameter (b=0.1)

    Returns
    -------
    y : ndarray of float [n_grid x n_out]
        Results of the n_out quantities of interest the gPC is conducted for
    additional_data : dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
        If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
    """

    def __init__(self, fname_julia=None):
        if fname_julia is not None:
            self.fname_julia = fname_julia                          # filename of julia function
        self.fname = inspect.getfile(inspect.currentframe())        # filename of python function

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        # pass parameters to julia function
        x1 = self.p["x1"]
        x2 = self.p["x2"]
        x3 = self.p["x3"]
        a = self.p["a"]
        b = self.p["b"]

        # access .jl file
        Main.fname_julia = self.fname_julia
        Main.include(Main.fname_julia)

        # call julia function
        y = Main.Ishigami(x1, x2, x3, a, b)

        if y.ndim == 0:
            y = np.array([[y]])
        elif y.ndim == 1:
            y = y[:, np.newaxis]

        return y
