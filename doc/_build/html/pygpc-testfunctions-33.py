import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-2, 2, 100)
parameters["x2"] = np.linspace(-2, 2, 100)

constants = OrderedDict()
constants["b"] = 10

plot("PermFunction", parameters, constants, plot_3d=False)