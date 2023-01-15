import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-5, 10, 100)
parameters["xi"] = np.linspace(-5, 10, 100)

constants = None

plot("ZakharovFunction", parameters, constants, plot_3d=False)