import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-100, 100, 100)
parameters["x2"] = np.linspace(-100, 100, 100)

constants = None

plot("SchafferFunction4", parameters, constants, plot_3d=False)