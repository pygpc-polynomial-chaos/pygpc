"""
L1 optimal sampling
===================
Add a description here about the criteria ["mc"] and ["tmc", "cc"]
and the 2 methods to create such a grid "greedy", "iter"

Example
-------
In order to create a grid of sampling points, we have to define the random parameters and create a gpc object.
"""

import pygpc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

# define model
model = pygpc.testfunctions.RosenbrockFunction()

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

# define problem
problem = pygpc.Problem(model, parameters)

# create gpc object
gpc = pygpc.Reg(problem=problem,
                order=[5]*problem.dim,
                order_max=5,
                order_max_norm=1,
                interaction_order=2,
                interaction_order_current=2,
                options=None,
                validation=None)

###############################################################################
# L1 designs with different optimization criteria can be created using the "criterion" argument in the options
# dictionary. In the following, we are going to create different L1 designs for 2 random variables with 200
# sampling points:

grid_mc = pygpc.L1(parameters_random=parameters,
                   n_grid=200,
                   gpc=gpc,
                   options={"criterion": ["mc"],
                            "method": "greedy",
                            "n_pool": 1000,
                            "seed": None})

grid_mc_cc = pygpc.L1(parameters_random=parameters,
                      n_grid=100,
                      gpc=gpc,
                      options={"criterion": ["tmc", "cc"],
                               "method": "greedy",
                               "n_pool": 1000,
                               "seed": None})

grid_tmc_cc = pygpc.L1(parameters_random=parameters,
                       n_grid=100,
                       gpc=gpc,
                       options={"criterion": ["tmc", "cc"],
                                "method": "greedy",
                                "n_pool": 1000,
                                "seed": None})


###############################################################################
# The following options are available for L1-optimal grids:
#
# - seed: set a seed to reproduce the results (default: None)
# - method:
#    - "greedy": greedy algorithm (default, recommended)
#    - "iter": iterative algorithm (faster but does not perform as good as "greedy")
# - criterion:
#    - ["mc"]: mutual coherence optimal#
#    - ["mc", "cc"]: mutual coherence and cross correlation optimal
#    - ["tmc", "cc"]: t-averaged mutual coherence and cross correlation optimal
# - n_pool: number of grid points in overall pool to select optimal points from (default: 10.000)
#
# The grid points are distributed as follows (in the normalized space):

fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(9.53, 3.2))

ax[0].scatter(grid_mc.coords_norm[:, 0], grid_mc.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[0])
ax[1].scatter(grid_mc_cc.coords_norm[:, 0], grid_mc_cc.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[1])
ax[2].scatter(grid_tmc_cc.coords_norm[:, 0], grid_tmc_cc.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[2])

title = ['L1 (mc)', 'L1 (mc, cc)', 'L1 (tmc-cc)']

for i in range(len(ax)):
    ax[i].set_xlabel("$x_1$", fontsize=12)
    ax[i].set_ylabel("$x_2$", fontsize=12)
    ax[i].set_xticks(np.linspace(-1, 1, 5))
    ax[i].set_yticks(np.linspace(-1, 1, 5))
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_title(title[i])
    ax[i].grid()

plt.tight_layout()

###############################################################################
# The sampling method can be selected accordingly for each gPC algorithm by setting the following options
# when setting up the algorithm:
options = dict()
...
options["grid"] = pygpc.L1
options["grid_options"] = {"seed": None,
                           "method": "greedy",
                           "criterion": ["mc", "cc"],
                           "n_pool": 1000}
...

# When using Windows you need to encapsulate the code in a main function and insert an
# if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively:
#
# if __name__ == '__main__':
#     main()
