import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-65.536, 65.536, 100)
parameters["x2"] = np.linspace(-65.536, 65.536, 100)

constants = None

plot("RotatedHyperEllipsoid", parameters, constants, plot_3d=False)