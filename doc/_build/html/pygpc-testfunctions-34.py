import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-4, 4, 100)
parameters["x2"] = np.linspace(-4, 4, 100)

plot("Ridge", parameters)