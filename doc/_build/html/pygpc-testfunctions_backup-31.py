import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["xi"] = np.linspace(-1, 1, 100)

plot("MovingParticleFrictionForce", parameters)