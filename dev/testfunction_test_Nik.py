# homepage of testfunctions:
# https://www.sfu.ca/~ssurjano/optimization.html
# to implement a testfunction go to: .../pygpc/pygpc/testfunctions/testfunctions.py
# Please implement:
# - Ackley Function
# - Bukin Function N. 6
# - Bohachevsky Functions
# - Perm Function 0, d, β
# - Rotated Hyper-Ellipsoid Function
# - Sum of Different Powers Function
# - Zakharov Function
# - Six-Hump Camel Function
# - Dixon-Price Function
# - Rosenbrock Function
# - De Jong Function N. 5
# - Michalewicz Function

import pygpc
import numpy as np
from collections import OrderedDict

testfunction_name = "Ackley"

parameters = OrderedDict()
parameters["x1"] = np.linspace(-32.768, 32.768, 1000)
parameters["x2"] = np.linspace(-32.768, 32.768, 1000)

constants = OrderedDict()
constants["a"] = 20.
constants["b"] = 0.2
constants["c"] = 0.5*np.pi

pygpc.testfunctions.plot_testfunction(testfunction_name=testfunction_name,
                                      parameters=parameters,
                                      constants=constants,
                                      output_idx=0)
