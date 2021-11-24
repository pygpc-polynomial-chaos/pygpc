"""
Hybrid L1-LHS sampling
======================
DESCRIBE GRIDS HERE

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
# L1-LHS designs with different optimization criteria can be created using the "criterion" argument in the options
# dictionary. Additionally it is possible to define the ratio between the number of L1 and LHS optimal sampling points
# by the weights. In the following, we are going to create different L1-LHS designs for 2 random variables with 200
# sampling points:

grid_025_075 = pygpc.L1_LHS(parameters_random=parameters,
                            n_grid=200,
                            gpc=gpc,
                            options={"weights": [0.25, 0.75],
                                     "criterion": ["mc", "cc"],
                                     "method": "greedy",
                                     "n_pool": 1000,
                                     "seed": None})

grid_050_050 = pygpc.L1_LHS(parameters_random=parameters,
                            n_grid=200,
                            gpc=gpc,
                            options={"weights": [0.50, 0.50],
                                     "criterion": ["mc", "cc"],
                                     "method": "greedy",
                                     "n_pool": 1000,
                                     "seed": None})

grid_075_025 = pygpc.L1_LHS(parameters_random=parameters,
                            n_grid=200,
                            gpc=gpc,
                            options={"weights": [0.75, 0.25],
                                     "criterion": ["mc", "cc"],
                                     "method": "greedy",
                                     "n_pool": 1000,
                                     "seed": None})

###############################################################################
# The following options are available for L1-LHS-optimal grids:
#
# - seed: set a seed to reproduce the results (default: None)
# - weights: weights between L1 and LHS optimal grid points
# - method:
#    - "greedy": greedy algorithm (default, recommended)
#    - "iter": iterative algorithm (faster but does not perform as good as "greedy")
# - criterion:
#    - ["mc"]: mutual coherence optimal
#    - ["mc", "cc"]: mutual coherence and cross correlation optimal
#    - ["tmc", "cc"]: t-averaged mutual coherence and cross correlation optimal
# - n_pool: number of grid points in overall pool to select optimal points from (default: 10.000)
#
# The grid points are distributed as follows (in the normalized space):

fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(9.53, 3.2))

ax[0].scatter(grid_025_075.coords_norm[:grid_025_075.grid_L1.n_grid, 0],
              grid_025_075.coords_norm[:grid_025_075.grid_L1.n_grid, 1],
              color=sns.color_palette("bright", 5)[0])
ax[0].scatter(grid_025_075.coords_norm[grid_025_075.grid_L1.n_grid:, 0],
              grid_025_075.coords_norm[grid_025_075.grid_L1.n_grid:, 1],
              color=sns.color_palette("pastel", 5)[0], edgecolor="k", alpha=0.75)
ax[1].scatter(grid_050_050.coords_norm[:grid_050_050.grid_L1.n_grid, 0],
              grid_050_050.coords_norm[:grid_050_050.grid_L1.n_grid, 1],
              color=sns.color_palette("bright", 5)[1])
ax[1].scatter(grid_050_050.coords_norm[grid_050_050.grid_L1.n_grid:, 0],
              grid_050_050.coords_norm[grid_050_050.grid_L1.n_grid:, 1],
              color=sns.color_palette("pastel", 5)[1], edgecolor="k", alpha=0.75)
ax[2].scatter(grid_075_025.coords_norm[:grid_075_025.grid_L1.n_grid, 0],
              grid_075_025.coords_norm[:grid_075_025.grid_L1.n_grid, 1],
              color=sns.color_palette("bright", 5)[2])
ax[2].scatter(grid_075_025.coords_norm[grid_075_025.grid_L1.n_grid:, 0],
              grid_075_025.coords_norm[grid_075_025.grid_L1.n_grid:, 1],
              color=sns.color_palette("pastel", 5)[2], edgecolor="k", alpha=0.75)

title = ['L1-LHS (weights: [0.25, 0.75])', 'L1-LHS (weights: [0.50, 0.50])', 'L1-LHS (weights: [0.75, 0.25])']

for i in range(len(ax)):
    ax[i].set_xlabel("$x_1$", fontsize=12)
    ax[i].set_ylabel("$x_2$", fontsize=12)
    ax[i].set_xticks(np.linspace(-1, 1, 5))
    ax[i].set_yticks(np.linspace(-1, 1, 5))
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_title(title[i])
    ax[i].grid()
    ax[i].legend(["L1", "LHS"], loc=1, fontsize=9, framealpha=1, facecolor=[0.95, 0.95, 0.95])

plt.tight_layout()

###############################################################################
# The sampling method can be selected accordingly for each gPC algorithm by setting the following options
# when setting up the algorithm:
options = dict()
...
options["grid"] = pygpc.L1_LHS
options["grid_options"] = {"seed": None,
                           "weights": [0.25, 0.75],
                           "method": "greedy",
                           "criterion": ["mc", "cc"],
                           "n_pool": 1000}
...

# When using Windows you need to encapsulate the code in a main function and insert an
# if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively:
#
# if __name__ == '__main__':
#     main()
