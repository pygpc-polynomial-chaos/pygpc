import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["rho_0"] = np.linspace(0, 1, 100)
parameters["beta"] = np.linspace(0, 20, 100)

constants = OrderedDict()
constants["alpha"] = 1.

plot("SurfaceCoverageSpecies", parameters, constants, plot_3d=False)