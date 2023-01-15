import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(0.5, 2.5, 100)
parameters["x2"] = np.linspace(0.5, 2.5, 100)

constants = None

plot("GramacyLeeFunction", parameters, constants, plot_3d=False)