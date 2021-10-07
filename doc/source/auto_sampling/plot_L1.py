"""
L1 optimal sampling
===================
Before we are going to introduce the different L1 grids, we are going to define a test problem.
"""

import pygpc
import numpy as np
from collections import OrderedDict

# define model
model = pygpc.testfunctions.Ishigami()

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = 0
parameters["a"] = 7
parameters["b"] = 0.1

# define problem
problem = pygpc.Problem(model, parameters)



###############################################################################
# L1 optimal grids use methods from the field of compressive sampling to enhance the information value of the drawn
# samples. In the field of signal processing compressive samples aims to recover signals with considerably less samples
# then standard methods by exploiting sparsity of the signal. In the gPC approximation with pygpc this is done by using
# L1 minimization, thus the grids tailored for the most efficient recovery using L1 minimization are called L1 optimal
# grids.
#
# Optimization Criteria of L1 optimal grids
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Mutual coherence
# ^^^^^^^^^^^^^^^^
# The mutual coherence (MC) of a matrix measures the cross-correlations between its columns by evaluating the largest
# absolute and normalized inner product between different columns. The mutual coherence for a matrix :math:
# '\mathbf{\Psi}' is calculated by:
#
# .. math::
#
#     \mu(\mathbf{\Psi}) = \max_ {1 \leq i, j\leq k \quad \textsf{and}
#     \quad j\neq i} \quad \frac{|\psi_i^T \psi_j|}{||\psi_i||.||\psi_j||}
#
#
# The objective is to minimize $\mu$ for a desired $\ell_1$ optimal design, which has been proven to increase the
# efficiency of compressive sampling.
#

# T-averaged-Mutual-Coherence (TMC)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The T-averaged-Mutual-Coherence (TMC) is also defined as the average of cross-correlations larger then a value t.
# A low TMC improves the recovery accuracy compared to random measurements. However TMC optimized sampling suffers from
# reduced robustness measurements with noise attached, that are not exactly sparse any more. It is defined by:
#
# .. math::
#
#     \mu_t(\mathbf{\Psi}) = \frac{\sum_{1 \leq i, j \leq K, i\neq j}\mathbbm{1}(g_{i,j} \geq t |g_{i, j}| )}
#     {\sum_{1 \leq i, j \leq K, i\neq j}\mathbbm{1}(g_{i,j})}
#
# .. M. Elad, Optimized Projections for Compressed Sensing, IEEE Transactions
# on Signal Processing 55 (12) (2007) 5695-5702.
#
# Where :math: '\mathbbm{1}' is the the indicator function.
#
# Average-cross correlation (CC)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The average-cross correlation (CC) is a measure that can also be used for a robuster recovery. It aims to increase
# the distance between the Gramian matrix :math: '\mathbf{[G_\mathbf{\Psi}]}' and the corresponding identity matrix
# :math: '[I_K]' It has proven to yield very accurate representations and also showed a superior performance handling
# noisy inputs
#
# .. N. Alemazkoor, H. Meidani, A near-optimal sampling strategy for sparse
# recovery of polynomial chaos expansions, Journal of Computational Physics
# 900 371 (2018) 137-151.
#
# .. L. Gang, Z. Zhu, D. Yang, L. Chang, H. Bai, On Projection Matrix Optimization
# for Compressive Sensing Systems, IEEE Transactions on Signal
# Processing 61 (11) (2013) 2887-2898.
#
# This measure can be expressed as:
#
#  .. math::
#      \gamma(\mathbf{\Psi}) = \frac{1}{N} \min_{[\mathbf{\Psi}] \in R^{M \times K}} ||[I_K] -
#      [\mathbf{G_\mathbf{\Psi}}]||^2_F
#
# where :math:'N := K \times (K - 1)' is the total number of column pairs, K is the order of the gPC and
# :math: '\gamma' is the optimization criteria abbreviated as average-cross correlation (CC) that is averaged
# over each sample location represented by the matrix iteration :math: '\mathbf{\Psi}'.
#
# As the optimization of only the average-cross correlation shows a possibly large mutual coherence and is regularly
# prone to an inaccurate recovery, a hybrid optimization criteria can be inspected. For this case the average-cross
# correlation and the mutual coherence will both be evaluated in a method following the canon of the findings of
# Alemazkoor and Meidani.
#
# .. math::
#     \argmin(f(\mathbf{\Psi})) = \argmin((\frac{\mu_{i} -\min(\boldsymbol\mu)}{\max(\boldsymbol\mu) -
#     \min(\boldsymbol\mu)})^2 + (\frac{\gamma_i -\min(\boldsymbol\gamma)}{max(\boldsymbol\gamma) -
#     \min(\boldsymbol\gamma)})^2)
#
# with :math: '\boldsymbol\mu = (\mu_{1}, \mu_{2}, ..., \mu_{i})$ and $ \boldsymbol\gamma =
# (\gamma_1, \gamma_2, ..., \gamma_i)'
#
# .. N. Alemazkoor, H. Meidani, A near-optimal sampling strategy for sparse
# recovery of polynomial chaos expansions, Journal of Computational Physics
# 900 371 (2018) 137-151.
#
# Optimization Method for L1 optimal grids
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Additionally to the optimization criteria specification the algorithm or the method for the optimization objective
# can also be picked independently. Pygpc provides an iterative and a greedy algorithm for finding the optimal set of
# samples.
#
# Iterative algorithm
# ^^^^^^^^^^^^^^^^^^^
# For the iterative method a number of iterations needs to be defined and until this number is reached the
# algorithm will generate full sampling sets and evaluate their optimization criteria to finally select the best
# contender among the given number of iterations.
#
# Greedy algorithm
# ^^^^^^^^^^^^^^^^
# The greedy method of creating L1 optimal sampling sets uses a large pool of initially drawn samples, out of this pool
# a randomly selected first sample is picked. After the each following samples is then picked by selecting the best fit
# to the selected samples out of the remaining samples in the pool according to the selected optimallity criterion.
# The algorithm then stops once the desired sample size is reached.



#
# L1 designs with different optimization criteria can be created using the "criterion" argument in the options
# dictionary.
# methods, weights...
#
# In the following, we are going to create different LHS designs for 2 random variables with 100
# sampling points:

# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():

import seaborn as sns
import matplotlib.pyplot as plt

# define gpc first, since the gpc matrix needs to be evaluated for the optimality criteria
gpc = pygpc.Reg(problem=problem, order_max=5)

grid_l1_mc = pygpc.L1(parameters_random=parameters, n_grid=7, options={"criterion": "mc",      "seed": None})
grid_l1_tmc = pygpc.L1(parameters_random=parameters, n_grid=7, options={"criterion": "tmc",    "seed": None})
grid_l1_cc = pygpc.L1(parameters_random=parameters, n_grid=7, options={"criterion": "cc", "seed": None})
grid_l1_mc_cc = pygpc.L1(parameters_random=parameters, n_grid=7, options={"criterion": "mc-cc",     "seed": None})

# plot
fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(12.7, 3.2))

ax[0].scatter(grid_l1_mc.coords_norm[:, 0], grid_l1_mc.coords_norm[:, 1], color=sns.color_palette("bright", 5)[0])
ax[1].scatter(grid_l1_tmc.coords_norm[:, 0], grid_l1_tmc.coords_norm[:, 1], color=sns.color_palette("bright", 5)[1])
ax[2].scatter(grid_l1_cc.coords_norm[:, 0], grid_l1_cc.coords_norm[:, 1], color=sns.color_palette("bright", 5)[2])
ax[3].scatter(grid_l1_mc_cc.coords_norm[:, 0], grid_l1_mc_cc.coords_norm[:, 1], color=sns.color_palette("bright", 5)[3])

title = ['LHS (standard)', 'LHS (corr opt)', 'LHS (Phi-P opt)', 'LHS (ese)']

for i in range(len(ax)):
    ax[i].set_xlabel("$x_1$", fontsize=12)
    ax[i].set_ylabel("$x_2$", fontsize=12)
    # ax[i].set_xticks(np.linspace(-1, 1, 5))
    # ax[i].set_yticks(np.linspace(-1, 1, 5))
    ax[i].set_xticks([-1] + np.linspace(-1+2/grid_l1_mc.n_grid/4, 1-2/grid_l1_mc.n_grid/4, grid_l1_mc.n_grid).tolist() + [1])
    ax[i].set_yticks([-1] + np.linspace(-1+2/grid_l1_mc.n_grid/4, 1-2/grid_l1_mc.n_grid/4, grid_l1_mc.n_grid).tolist() + [1])
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([-1, 1])
    ax[i].set_title(title[i])
    ax[i].grid()

plt.tight_layout()




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

title = ['L1-mc (greedy)', 'L1-mc (iter)', 'L1-mc-cc (greedy)', 'L1-mc-cc (iter)']

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
#

# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
