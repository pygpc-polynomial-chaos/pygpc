import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(0, 1, 250)
parameters["x2"] = np.linspace(0, 1, 250)

plot("DiscontinuousRidgeManufactureDecay", parameters)