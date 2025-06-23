"""
Polynomial basis functions
==========================
"""
#%%
# Total-order gPC
# ^^^^^^^^^^^^^^^
# In general, the set :math:`\mathcal{A}(\mathbf{p})` of multi-indices can be freely chosen according
# to the problem under investigation. In the following figures, the blue boxes correspond to polynomials
# included in the gPC expansion. The coordinates of the boxes correspond to the multi-indices
# :math:`\mathbf{\alpha}`, which correspond to the polynomial degrees of the individual basis functions
# forming the joint basis functions. For a total-order gPC, the number of basis functions, and hence,
# coefficients to determine, increases exponentially in this case :math:`N_c=(p+1)^d`

# sphinx_gallery_thumbnail_number = 2

# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
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

# define basis
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
# does not exceed a predefined value :math:`p_g`.
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
# :math:`p_i`. Additionally, univariate polynomials of higher orders specified in :math:`\mathbf{p}_u`
# may be added to the set of basis functions.

# reduced basis gPC
basis = pygpc.Basis()
basis.init_basis_sgpc(problem=problem,
                      order=[7, 9, 3],
                      order_max=7,
                      order_max_norm=0.8,
                      interaction_order=3)

basis.plot_basis(dims=[0, 1, 2])

#%%
# Isotropic adaptive basis
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The basic problem in gPC is to find a suitable basis while reducing the number of necessary forward
# simulations to determine the gPC coefficients! To do this two basis increment strategies exist. This first is called
# isotropic and is the default option for the gpc. It determines which multi-indices are picked to be added to the
# existing set of basis functions in terms of their order and dimension. The boundary conditions for this expansion
# are given by the maximum order and the interaction order in the gpc options. The maximum order sets a limit to how
# high any index may grow and the interaction order limits the maximal dimension in which multi-indices can be chosen.
# In action isotropic basis incrementation chooses the new basis multi-indices equally in each direction decreasing the
# dimension in every step. If the interaction order is set as 3 this means that the first indices to be increased is
# along the axes (shown in orange in the figure below), then the indices that span the area between the axes are
# chosen and finally the indices that create the volume contained by the surrounding area are added. After that the
# axes are extended again and the cycle is repeated until the error is sufficient. For an interaction order of higher
# than three the expansion continues until the final dimension is reached.
#
# .. image:: /examples/images/Fig_adaptive_basis_isotropic.png
#     :width: 1300
#     :align: center


# define model
model = pygpc.testfunctions.Ishigami()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["a"] = 7.
parameters["b"] = 0.1

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["order_start"] = 10
options["order_end"] = 20
options["solver"] = "Moore-Penrose"
options["interaction_order"] = 2
options["order_max_norm"] = 1.0
options["n_cpu"] = 0
options["adaptive_sampling"] = False
options["eps"] = 0.05
options["fn_results"] = None
options["basis_increment_strategy"] = None
options["matrix_ratio"] = 4
options["grid"] = pygpc.Random
options["grid_options"] = {"seed": 1}

# define algorithm
algorithm = pygpc.RegAdaptive(problem=problem, options=options)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results = session.run()

#%%
# Anisotropic adaptive basis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# In addition to the adaptive selection from the previous method, the specification options["basis_increment_strategy"]
# = anisotropic can be used for the gpc object to use an algorithm by Gerstner and Griebel [1] for an anisotropic basis
# increment strategy. The motivation behind it lies in reducing the variance of the output data of the gpc.
#
# .. math::
#
#     Var(\mathbf{q}(\mathbf{\xi}) = \sum_{k=1}^{P}(\mathbf{u}_{\mathbf{\alpha}_k} || \mathbf{\Phi}_{\mathbf{\alpha}_k}
#     ||)^2
#
# where :math:`\mathbf{q}(\mathbf{\xi})` is the output data dependent on :math:`\mathbf{\xi}` which are the random
# variables, :math:`P` is the order of the gpc and :math:`\mathbf{u}_{\mathbf{\alpha}_k}` are the coefficients of the
# basis function :math:`\mathbf{\Phi}_{\mathbf{\alpha}_k}`. The variance depends directly on the coefficients
# :math:`\mathbf{u}_{\mathbf{\alpha}_k}` and can be reduced by using them as a optimization criterion in the mentioned
# algorithm. A normalized version :math:`\hat{\mathbf{u}}_{\mathbf{\alpha}_k}` directly corresponds to the variance and
# is given by
#
# .. math::
#
#     \hat{\mathbf{u}}_{\mathbf{\alpha}_k} = (\mathbf{u}_{\mathbf{\alpha}_k} || \mathbf{\Phi}_{\mathbf{\alpha}_k}
#     ||)^2
#
# In pygpc this quantity is calculated as the maximum L2-norm of the current coefficients and the relevant index
# :math:'k' is  extracted from the set of multi-indices. The anisotropic adaptive basis algorithm selects the
# multi-index :math:'k' with the
# highest norm as the starting point for a basis expansion. The goal during the expansion is to find suitable candidate
# indices that meet the following two criteria:
# (1) The index is not completely enclosed by other indices with higher basis components since this would mean that it
# is already included;
# (2) The index needs to have predecessors. This means that in all directions of decreasing order connecting
# multi-indices exist already and the new index is not 'floating'. In the figure below this is shown again for the
# three-dimensional case. First the outer multi-indices that have connected faces with other included multi-indices
# which 'have predecessors' are selected (marked green). Then
# the basis function coefficients are computed for these candidates and the multi-index with the highest coefficient
# is picked for the expansion (marked red). The index is then expanded in every dimension around it where the resulting
# index is not already included in the multi-index-set yet (marked orange).
#
# .. image:: /examples/images/Fig_adaptive_basis_anisotropic.png
#     :width: 1300
#     :align: center
#

# define model
model = pygpc.testfunctions.Ishigami()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["a"] = 7.
parameters["b"] = 0.1

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["order_start"] = 10
options["order_end"] = 20
options["solver"] = "Moore-Penrose"
options["interaction_order"] = 2
options["order_max_norm"] = 1.0
options["n_cpu"] = 0
options["adaptive_sampling"] = False
options["eps"] = 0.05
options["fn_results"] = None
options["basis_increment_strategy"] = "anisotropic"
options["matrix_ratio"] = 4
options["grid"] = pygpc.Random
options["grid_options"] = {"seed": 1}

# define algorithm
algorithm = pygpc.RegAdaptive(problem=problem, options=options)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results = session.run()

#
#
# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()