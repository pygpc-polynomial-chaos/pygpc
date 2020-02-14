"""
Polynomial basis functions
==========================
"""

#%%
# Test problem
# ------------

import pygpc
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from collections import OrderedDict

# define model
model = pygpc.testfunctions.Ishigami()

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

# define problem
problem = pygpc.Problem(model, parameters)

#%%
# Total-order gPC
# ^^^^^^^^^^^^^^^
# In general, the set :math:`\mathcal{A}(\mathbf{p})` of multi-indices can be freely chosen according
# to the problem under investigation. In the following figures, the blue boxes correspond to polynomials
# included in the gPC expansion. The coordinates of the boxes correspond to the multi-indices
# :math:`\mathbf{\alpha}`, which correspond to the polynomial degrees of the individual basis functions
# forming the joint basis functions. For a total-order gPC, the number of basis functions, and hence,
# coefficients to determine, increases exponentially in this case :math:`N_c=(P+1)^d`

basis = pygpc.Basis()
basis.init_basis_sgpc(problem=problem,
                      order=[5, 5, 5],
                      order_max=15,
                      order_max_norm=1,
                      interaction_order=3)

basis.plot_basis(dims=[0, 1, 2])

#%%
# Maximum-order gPC
# ^^^^^^^^^^^^^^^^^
# In practical applications, the more economical maximum total order gPC is preferably used.
# In this case, the set :math:`\mathcal{A}(p_g)` includes all polynomials whose total order
# does not exceed a predefined value :math:`P_g`.
#
# .. math::
#
#     \mathcal{A}(p_g) = \left\{ \mathbf{\alpha} \, : \, \sum_{i=1}^{d} \alpha_i \leq p_g \right\} =
#     \left\{ \mathbf{\alpha} \, : \lVert \mathbf{\alpha} \rVert_1  \leq p_g \right\}
#
# This results in a reduced set of basis functions and is termed maximum order gPC. The number of multi-indices,
# and hence, the dimension of the space spanned by the polynomials, is:
#
# .. math::
#     N_c = \binom{d+p_g}{d} = \frac{(d+p_g)!}{d!p_g!}.

basis = pygpc.Basis()
basis.init_basis_sgpc(problem=problem,
                      order=[5, 5, 5],
                      order_max=5,
                      order_max_norm=1,
                      interaction_order=3)

basis.plot_basis(dims=[0, 1, 2])

#%%
# Reduced-basis gPC
# -----------------
# The concept of the *maximum-order* gPC is extended by introducing three new parameters:
# - the *univariate* expansion order :math:`\mathbf{p}_u = (p_{u,1},...,p_{u,d})` with
# :math:`p_{u,i}>p_g \forall i={1,...,d}`
# - the *interaction order* :math:`p_i`, limits the number of interacting parameters and it reflects the
# dimensionality, i.e. the number of random variables (independent variables) appearing in the
# basis function :math:`\Psi_{\mathbf{\alpha}}({\xi})`: :math:`\lVert\mathbf{\alpha}\rVert_0 \leq p_i`
# - the *maximum order norm* :math:`q` additionally truncates the included basis functions
# in terms of the maximum order :math:`p_g` such that
# :math:`\lVert \mathbf{\alpha} \rVert_{q}=\sqrt[q]{\sum_{i=1}^d \alpha_i^{q}} \leq p_g`
#
# Those parameters define the set
# :math:`\mathcal{A}(\mathbf{p})` with :math:`\mathbf{p} = (\mathbf{p}_u,p_i,p_g, q)`
#
# The reduced set :math:`\mathcal{A}(\mathbf{p})` is then constructed by the following rule:
#
# .. math::
#     \mathcal{A}(\mathbf{p}) := \left\{ \mathbf{\alpha} \in \mathbb{N}_0^d\, :
#     (\lVert \mathbf{\alpha} \rVert_q  \leq p_g \wedge \lVert\mathbf{\alpha}\rVert_0 \leq p_i)
#     \vee (\lVert \mathbf{\alpha} \rVert_1  \leq p_{u,i} \wedge \lVert\mathbf{\alpha}\rVert_0 = 1,
#     \forall i \in \{1,...,d\}) \right\}
#
# It includes all elements from a total order gPC with the restriction of the interaction order
# :math:`P_i`. Additionally, univariate polynomials of higher orders specified in :math:`\mathbf{P}_u`
# may be added to the set of basis functions.

# reduced basis gPC
basis = pygpc.Basis()
basis.init_basis_sgpc(problem=problem,
                      order=[10, 12, 3],
                      order_max=7,
                      order_max_norm=0.8,
                      interaction_order=3)

basis.plot_basis(dims=[0, 1, 2])

#%%
# Adaptive basis
# ^^^^^^^^^^^^^^
# The basic problem in gPC is to find a suitable basis while reducing the number of necessary forward
# simulations to determine the gPC coefficients!

basis_order = np.array([-1, 0])
interaction_order = 2
order_max_norm = 1
n_iter = 10

# define model
model = pygpc.testfunctions.Ishigami()

# define parameters
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

# define problem
problem = pygpc.Problem(model, parameters)
basis = pygpc.Basis()

for i in range(n_iter):
    # increment basis
    basis_order[0], basis_order[1] = pygpc.increment_basis(order_current=basis_order[0],
                                                           interaction_order_current=basis_order[1],
                                                           interaction_order_max=interaction_order,
                                                           incr=1)

    # set basis
    basis.init_basis_sgpc(problem=problem,
                          order=[basis_order[0]] * problem.dim,
                          order_max=basis_order[0],
                          order_max_norm=order_max_norm,
                          interaction_order=interaction_order,
                          interaction_order_current=basis_order[1])

    # plot basis
    basis.plot_basis(dims=[0, 1, 2], dynamic_plot_update=True)

    time.sleep(0.5)
    display.display(plt.gcf())

    if i != (n_iter-1):
        display.clear_output(wait=True)
        plt.close()
