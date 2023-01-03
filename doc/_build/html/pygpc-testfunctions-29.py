import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-1.5, 4, 100)
parameters["x2"] = np.linspace(-3, 4, 100)

constants = None

plot("McCormickFunction", parameters, constants, plot_3d=False)