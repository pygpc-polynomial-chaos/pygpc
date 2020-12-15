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
# To increase the information of each individual sampling point and to prevent undersampling, LHS is a simple
# alternative to enhance the space-filling properties of the sampling scheme first established by
# McKay et al. (2000).
#
# .. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). A comparison of three methods for selecting
#    values of input variables in the analysis of output from a computer code. Technometrics, 42(1), 55-61.
#
# To draw :math:`n` independent samples from a number of :math:`d`-dimensional parameters
# a matrix :math:`\Pi` is constructed with
#
# .. math::
#
#     \pi_{ij} = \frac{p_{ij} - u}{n}
#
# where :math:`P` is a :math:`d \times n` matrix of randomly perturbed integers
# :math:`p_{ij} \in \mathbb{N}, {1,...,n}` and u is uniform random number :math:`u \in [0,1]`.
#
# LHS Designs can further be improved upon, since the pseudo-random sampling procedure
# can lead to samples with high spurious correlation and the space filling capability
# in itself leaves room for improvement, some optimization criteria have been found to
# be adequate for compensating the initial designs shortcomings.
# 
# Optimization Criteria of LHS designs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Spearman Rank Correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# For a sample size of :math:`n` the scores of each variable are converted to their Ranks :math:`rg_{X_i}`
# the Spearman Rank Correlation Coefficient is then the Pearson Correlation Coefficient applied to the rank
# variables :math:`rg_{X_i}`:
# 
# .. math::
#
#     r_s = \rho_{rg_{X_i}, rg_{X_j}} = \frac{cov(rg_{X_i}, rg_{X_j})}{\sigma_{rg_{X_i}} \sigma_{rg_{X_i}}}
#
# where :math:`\rho` is the pearson correlation coefficient, :math:`\sigma` is the standard deviation
# and :math:`cov` is the covariance of the rank variables
# 
# Maximum-Minimal-Distance
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For creating a so called maximin distance design that maximizes the minimum inter-site distance, proposed by
# Johnson et al.
# 
# .. math::
#
#     \min_{1 \leqslant i, j \leqslant n, i \neq j} d(x_i,x_j),
# 
# where :math:`d` is the distance between two samples :math:`x_i` and :math:`x_j` and
# :math:`n` is the number of samples in a sample design.
# 
# .. math::
#
#     d(x_i,x_j) = d_ij = [ \sum_{k=1}^{m}|x_ik - x_jk| ^ t]^\frac{1}{t}, t \in {1,2}
#
# There is however a more elegant way of computing this optimization criterion as shown by Morris and Mitchell (1995),
# called the :math:`\varphi_P` criterion.
# 
# .. math::
#
#     \min\varphi_P \quad \text{subject to} \quad \varphi_P = [ \sum_{k = 1} ^ {s} J_id_i  ^ p]^\frac{1}{p},
# 
# where :math:`s` is the number of distinct distances, :math:`J` is an vector of indices of the distances
# and :math:`p` is an integer. With a very large :math:`p` this criterion is equivalent to the maximin criterion
#
# .. Morris, M. D. and Mitchell, T. J. ( (1995). Exploratory Designs for Computer Experiments.J. Statist. Plann.
#    Inference 43, 381-402.
# 
# LHS with enhanced stochastic evolutionary algorithm (ESE)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To achieve optimized designs with a more stable method and possibly quicker then by simply evaluating
# the criteria over a number of repetitions **pygpc** can use an ESE for achieving sufficient
# :math:`\varphi_P`-value. This algorithm is more appealing in its efficacy and proves to
# [sth about the resulting error or std in a low sample size].
# This method originated from Jin et al. (2005).
#
# .. Jin, R., Chen, W., Sudjianto, A. (2005). An efficient algorithm for constructing optimal
#    design of computer experiments. Journal of statistical planning and inference, 134(1), 268-287.
#
# LHS designs with different optimization criteria can be created using the "criterion" argument in the options
# dictionary. In the following, we are going to create different LHS designs for 2 random variables with 100
# sampling points:

# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():

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

grid_lhs_std = pygpc.LHS(parameters_random=parameters, n_grid=7, options={"criterion": None,      "seed": None})
grid_lhs_cor = pygpc.LHS(parameters_random=parameters, n_grid=7, options={"criterion": "corr",    "seed": None})
grid_lhs_max = pygpc.LHS(parameters_random=parameters, n_grid=7, options={"criterion": "maximin", "seed": None})
grid_lhs_ese = pygpc.LHS(parameters_random=parameters, n_grid=7, options={"criterion": "ese",     "seed": None})

# plot
fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(12.7, 3.2))

ax[0].scatter(grid_lhs_std.coords_norm[:, 0], grid_lhs_std.coords_norm[:, 1], color=sns.color_palette("bright", 5)[0])
ax[1].scatter(grid_lhs_cor.coords_norm[:, 0], grid_lhs_cor.coords_norm[:, 1], color=sns.color_palette("bright", 5)[1])
ax[2].scatter(grid_lhs_max.coords_norm[:, 0], grid_lhs_max.coords_norm[:, 1], color=sns.color_palette("bright", 5)[2])
ax[3].scatter(grid_lhs_ese.coords_norm[:, 0], grid_lhs_ese.coords_norm[:, 1], color=sns.color_palette("bright", 5)[3])

title = ['LHS (standard)', 'LHS (corr opt)', 'LHS (Phi-P opt)', 'LHS (ese)']

for i in range(len(ax)):
    ax[i].set_xlabel("$x_1$", fontsize=12)
    ax[i].set_ylabel("$x_2$", fontsize=12)
    # ax[i].set_xticks(np.linspace(-1, 1, 5))
    # ax[i].set_yticks(np.linspace(-1, 1, 5))
    ax[i].set_xticks([-1] + np.linspace(-1+2/grid_lhs_std.n_grid/4, 1-2/grid_lhs_std.n_grid/4, grid_lhs_std.n_grid).tolist() + [1])
    ax[i].set_yticks([-1] + np.linspace(-1+2/grid_lhs_std.n_grid/4, 1-2/grid_lhs_std.n_grid/4, grid_lhs_std.n_grid).tolist() + [1])
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_title(title[i])
    ax[i].grid()

plt.tight_layout()

###############################################################################
# L1 optimal grids
# ----------------
# Add a description here about the criteria ["mc"] and ["tmc", "cc"]
# and the 2 methods to create such a grid "greedy", "iter"

gpc = pygpc.Reg(problem=problem, order_max=5)

grid_l1_mc_greedy = pygpc.L1(parameters_random=parameters,
                             n_grid=100,
                             gpc=gpc,
                             options={"criterion": ["mc"],
                                      "method": "greedy",
                                      "n_pool": 1000,
                                      "seed": None})

grid_l1_mc_iter = pygpc.L1(parameters_random=parameters,
                           n_grid=100,
                           gpc=gpc,
                           options={"criterion": ["mc"],
                                    "method": "iter",
                                    "n_iter": 1000,
                                    "seed": None})

grid_l1_tmccc_greedy = pygpc.L1(parameters_random=parameters,
                                n_grid=100,
                                gpc=gpc,
                                options={"criterion": ["tmc", "cc"],
                                         "method": "greedy",
                                         "n_pool": 1000,
                                         "seed": None})

grid_l1_tmccc_iter = pygpc.L1(parameters_random=parameters,
                              n_grid=100,
                              gpc=gpc,
                              options={"criterion": ["tmc", "cc"],
                                       "method": "iter",
                                       "n_iter": 1000,
                                       "seed": None})

# plot
fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(12.7, 3.2))

ax[0].scatter(grid_l1_mc_greedy.coords_norm[:, 0],    grid_l1_mc_greedy.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[0])
ax[1].scatter(grid_l1_mc_iter.coords_norm[:, 0],      grid_l1_mc_iter.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[1])
ax[2].scatter(grid_l1_tmccc_greedy.coords_norm[:, 0], grid_l1_tmccc_greedy.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[2])
ax[3].scatter(grid_l1_tmccc_iter.coords_norm[:, 0],   grid_l1_tmccc_iter.coords_norm[:, 1],
              color=sns.color_palette("bright", 5)[3])

title = ['L1-mc (greedy)', 'L1-mc (iter)', 'L1-tmc-cc (greedy)', 'L1-tmc-cc (iter)']

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

###############################################################################
# Grid comparison
# ^^^^^^^^^^^^^^^
# Add a figure (.png) showing some results from our grid paper







# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
