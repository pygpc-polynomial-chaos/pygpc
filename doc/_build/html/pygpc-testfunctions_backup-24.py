import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-np.pi, np.pi, 100)
parameters["x2"] = np.linspace(-np.pi, np.pi, 100)

constants = OrderedDict()
constants["a"] = 7.
constants["b"] = 0.1
constants["x3"] = 0.

plot("Ishigami", parameters, constants, plot_3d=False)