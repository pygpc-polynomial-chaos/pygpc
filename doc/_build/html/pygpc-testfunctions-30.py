import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(0, np.pi, 100)
parameters["xi"] = np.linspace(0, np.pi, 100)

constants = OrderedDict()
constants["m"] = 10.

plot("MichalewiczFunction", parameters, constants, plot_3d=False)