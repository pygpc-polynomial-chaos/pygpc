"""
Hybrid LHS-L1 sampling
======================
Before we are going to introduce the different LHS grids, we are going to define a test problem.
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

###############################################################################
# Hybrid grids (LHS/L1)
# ---------------------
# Describe here that it makes a difference in which order the grids are generated and introduce the weighting factor
#
# L1-LHS grids
# ^^^^^^^^^^^^
# Describe L1-LHS grids

gpc = pygpc.Reg(problem=problem, order_max=5)

# weighting factor between L1 and LHS grid
weights = [0.5, 0.5]

grid_l1lhs_mc_greedy = pygpc.L1_LHS(parameters_random=parameters,
                                    n_grid=100,
                                    gpc=gpc,
                                    options={"weights": weights,
                                             "criterion": ["mc"],
                                             "method": "greedy",
                                             "n_pool": 1000,
                                             "seed": None})

grid_l1lhs_mc_iter = pygpc.L1_LHS(parameters_random=parameters,
                                  n_grid=100,
                                  gpc=gpc,
                                  options={"weights": weights,
                                           "criterion": ["mc"],
                                           "method": "iter",
                                           "n_iter": 1000,
                                           "seed": None})

grid_l1lhs_tmccc_greedy = pygpc.L1_LHS(parameters_random=parameters,
                                       n_grid=100,
                                       gpc=gpc,
                                       options={"weights": weights,
                                                "criterion": ["tmc", "cc"],
                                                "method": "greedy",
                                                "n_pool": 1000,
                                                "seed": None})

grid_l1lhs_tmccc_iter = pygpc.L1_LHS(parameters_random=parameters,
                                     n_grid=100,
                                     gpc=gpc,
                                     options={"weights": weights,
                                              "criterion": ["tmc", "cc"],
                                              "method": "iter",
                                              "n_iter": 1000,
                                              "seed": None})

# plot
fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(12.7, 3.2))

n_grid_l1 = grid_l1lhs_tmccc_iter.grid_L1.n_grid
n_grid_lhs = grid_l1lhs_tmccc_iter.grid_LHS.n_grid

ax[0].scatter(grid_l1lhs_mc_greedy.coords_norm[:n_grid_l1, 0], grid_l1lhs_mc_greedy.coords_norm[:n_grid_l1, 1],
              color=sns.color_palette("bright", 5)[0])
ax[0].scatter(grid_l1lhs_mc_greedy.coords_norm[n_grid_l1:, 0], grid_l1lhs_mc_greedy.coords_norm[n_grid_l1:, 1],
              color=sns.color_palette("pastel", 5)[0], edgecolor="k", alpha=0.75)
ax[1].scatter(grid_l1lhs_mc_iter.coords_norm[:n_grid_l1, 0], grid_l1lhs_mc_iter.coords_norm[:n_grid_l1, 1],
              color=sns.color_palette("bright", 5)[1])
ax[1].scatter(grid_l1lhs_mc_iter.coords_norm[n_grid_l1:, 0], grid_l1lhs_mc_iter.coords_norm[n_grid_l1:, 1],
              color=sns.color_palette("pastel", 5)[1], edgecolor="k", alpha=0.75)
ax[2].scatter(grid_l1lhs_tmccc_greedy.coords_norm[:n_grid_l1, 0], grid_l1lhs_tmccc_greedy.coords_norm[:n_grid_l1, 1],
              color=sns.color_palette("bright", 5)[2])
ax[2].scatter(grid_l1lhs_tmccc_greedy.coords_norm[n_grid_l1:, 0], grid_l1lhs_tmccc_greedy.coords_norm[n_grid_l1:, 1],
              color=sns.color_palette("pastel", 5)[2], edgecolor="k", alpha=0.75)
ax[3].scatter(grid_l1lhs_tmccc_iter.coords_norm[:n_grid_l1, 0], grid_l1lhs_tmccc_iter.coords_norm[:n_grid_l1, 1],
              color=sns.color_palette("bright", 5)[3])
ax[3].scatter(grid_l1lhs_tmccc_iter.coords_norm[n_grid_l1:, 0], grid_l1lhs_tmccc_iter.coords_norm[n_grid_l1:, 1],
              color=sns.color_palette("pastel", 5)[3], edgecolor="k", alpha=0.75)

title = ['L1-LHS-mc (greedy)', 'L1-LHS-mc (iter)', 'L1-LHS-tmc-cc (greedy)', 'L1-LHS-tmc-cc (iter)']

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

# LHS-L1 grids
# ^^^^^^^^^^^^
# Describe LHS-L1 grids

gpc = pygpc.Reg(problem=problem, order_max=5)

# weighting factor between L1 and LHS grid
weights = [0.5, 0.5]

grid_lhsl1_mc_greedy = pygpc.LHS_L1(parameters_random=parameters,
                                    n_grid=100,
                                    gpc=gpc,
                                    options={"weights": weights,
                                             "criterion": ["mc"],
                                             "method": "greedy",
                                             "n_pool": 1000,
                                             "seed": None})

grid_lhsl1_mc_iter = pygpc.LHS_L1(parameters_random=parameters,
                                  n_grid=100,
                                  gpc=gpc,
                                  options={"weights": weights,
                                           "criterion": ["mc"],
                                           "method": "iter",
                                           "n_iter": 1000,
                                           "seed": None})

grid_lhsl1_tmccc_greedy = pygpc.LHS_L1(parameters_random=parameters,
                                       n_grid=100,
                                       gpc=gpc,
                                       options={"weights": weights,
                                                "criterion": ["tmc", "cc"],
                                                "method": "greedy",
                                                "n_pool": 1000,
                                                "seed": None})

grid_lhsl1_tmccc_iter = pygpc.LHS_L1(parameters_random=parameters,
                                     n_grid=100,
                                     gpc=gpc,
                                     options={"weights": weights,
                                              "criterion": ["tmc", "cc"],
                                              "method": "iter",
                                              "n_iter": 1000,
                                              "seed": None})

# plot
fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(12.7, 3.2))

n_grid_lhs = grid_lhsl1_tmccc_iter.grid_LHS.n_grid
n_grid_l1 = grid_lhsl1_tmccc_iter.grid_L1.n_grid

ax[0].scatter(grid_lhsl1_mc_greedy.coords_norm[:n_grid_lhs, 0], grid_lhsl1_mc_greedy.coords_norm[:n_grid_lhs, 1],
              color=sns.color_palette("pastel", 5)[0], edgecolor="k", alpha=0.75)
ax[0].scatter(grid_lhsl1_mc_greedy.coords_norm[n_grid_lhs:, 0], grid_lhsl1_mc_greedy.coords_norm[n_grid_lhs:, 1],
              color=sns.color_palette("bright", 5)[0])
ax[1].scatter(grid_lhsl1_mc_iter.coords_norm[:n_grid_lhs, 0], grid_lhsl1_mc_iter.coords_norm[:n_grid_lhs, 1],
              color=sns.color_palette("pastel", 5)[1], edgecolor="k", alpha=0.75)
ax[1].scatter(grid_lhsl1_mc_iter.coords_norm[n_grid_lhs:, 0], grid_lhsl1_mc_iter.coords_norm[n_grid_lhs:, 1],
              color=sns.color_palette("bright", 5)[1])
ax[2].scatter(grid_lhsl1_tmccc_greedy.coords_norm[:n_grid_lhs, 0], grid_lhsl1_tmccc_greedy.coords_norm[:n_grid_lhs, 1],
              color=sns.color_palette("pastel", 5)[2], edgecolor="k", alpha=0.75)
ax[2].scatter(grid_lhsl1_tmccc_greedy.coords_norm[n_grid_lhs:, 0], grid_lhsl1_tmccc_greedy.coords_norm[n_grid_lhs:, 1],
              color=sns.color_palette("bright", 5)[2])
ax[3].scatter(grid_lhsl1_tmccc_iter.coords_norm[:n_grid_lhs, 0], grid_lhsl1_tmccc_iter.coords_norm[:n_grid_lhs, 1],
              color=sns.color_palette("pastel", 5)[3], edgecolor="k", alpha=0.75)
ax[3].scatter(grid_lhsl1_tmccc_iter.coords_norm[n_grid_lhs:, 0], grid_lhsl1_tmccc_iter.coords_norm[n_grid_lhs:, 1],
              color=sns.color_palette("bright", 5)[3])

title = ['LHS-L1-mc (greedy)', 'LHS-L1-mc (iter)', 'LHS-L1-tmc-cc (greedy)', 'LHS-L1-tmc-cc (iter)']

for i in range(len(ax)):
    ax[i].set_xlabel("$x_1$", fontsize=12)
    ax[i].set_ylabel("$x_2$", fontsize=12)
    ax[i].set_xticks(np.linspace(-1, 1, 5))
    ax[i].set_yticks(np.linspace(-1, 1, 5))
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_title(title[i])
    ax[i].grid()
    ax[i].legend(["LHS", "L1"], loc=1, fontsize=9, framealpha=1, facecolor=[0.95, 0.95, 0.95])

plt.tight_layout()


# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
