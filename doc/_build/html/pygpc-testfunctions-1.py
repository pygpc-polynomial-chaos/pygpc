import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-32.768, 32.768, 100)
parameters["x2"] = np.linspace(-32.768, 32.768, 100)

constants = OrderedDict()
constants["a"] = 20.
constants["b"] = 0.2
constants["c"] = 0.5*np.pi

plot("Ackley", parameters, constants, plot_3d=False)