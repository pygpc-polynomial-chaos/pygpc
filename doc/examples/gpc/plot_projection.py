"""
Dimensionality reduction
========================

Introduction
^^^^^^^^^^^^
A large number of models show redundancies in the form of correlations between the parameters under investigation.
Especially in the case of high-dimensional problems, the correlative behavior of the target variable as a function
of the parameters is usually completely unknown.  Correlating parameters can be combined into surrogate parameters
by means of a principal component analysis, which enables a reduction of the effective parameter number.
This significantly reduces the computational effort compared to the original problem with only minor losses
in modeling accuracy.

In this case, the :math:`n_d` original random variables :math:`\\mathbf{\\xi}` are reduced to a new set of :math:`n_d'`
random variables :math:`\\mathbf{\\eta}`, where :math:`n_d'<n_d` and which are a linear combination of the original
random variables such that:

.. math::

    \\mathbf{\\eta} = [\\mathbf{W}]\\mathbf{\\xi}

The challenging part is to determine the projection matrix :math:`[\\mathbf{W}]`, which rotates the basis and
reformulates the original problem to a more efficient one without affecting the solution accuracy too much.
Determining the projection matrix results in a separate optimization problem besides of determining the
gPC coefficients. Several approaches where investigated in the literature from for example Tipireddy and Ghanem (2014)
as well as Tsilifis et al. (2019). They solved the coupled optimization problems alternately, i.e. keeping the solution
of the one fixed while solving the other until some convergence criterion is satisfied.

Method
^^^^^^
In pygpc, the projection matrix :math:`[\\mathbf{W}]` is determined from the singular value decomposition of
the gradients of the solution vector along the original parameter space. The gradients of the quantity of interest (QoI)
along :math:`\\mathbf{\\xi}` are stored in the matrix :math:`[\\mathbf{Y}_\\delta]`:

.. math::

    [\\mathbf{Y}_\\partial] =
    \\left[ \\begin{array}{ccc}
    \\left.\\frac{\\partial y}{\\partial\\xi_1}\\right|_{\\xi^{(1)}} & \\ldots & \\left.\\frac{\\partial y}{\\partial\\xi_d}\\right|_{\\xi^{(1)}}\\\\
    \\left.\\frac{\\partial y}{\\partial\\xi_1}\\right|_{\\xi^{(2)}} & \\ldots & \\left.\\frac{\\partial y}{\\partial\\xi_d}\\right|_{\\xi^{(2)}}\\\\
    \\vdots & \\ddots & \\vdots\\\\
    \\left.\\frac{\\partial y}{\\partial\\xi_1}\\right|_{\\xi^{(n_g)}} & \\ldots & \\left.\\frac{\\partial y}{\\partial\\xi_d}\\right|_{\\xi^{(n_g)}}\\\\
    \\end{array}\\right]

The matrix :math:`[\\mathbf{Y}_\\delta]` is of size :math:`n_g \\times n_d`, where :math:`n_g` is the number of sampling
points and :math:`n_d` is the number of random variables in the original parameter space. Its SVD is given by:

.. math::

    \\left[\\mathbf{Y}_\\delta\\right] = \\left[\\mathbf{U}\\right]\\left[\\mathbf{\Sigma}\\right]\\left[\\mathbf{V}^*\\right]

The matrix :math:`\\left[\\mathbf{\Sigma}\\right]` contains the :math:`n_d` singular values :math:`\\sigma_i` of
:math:`[\\mathbf{Y}_\\delta]`. The projection matrix :math:`[\\mathbf{W}]` is determined from the right singular
vectors :math:`[\\mathbf{V}^*]` by including principal axes up to a limit where the sum of the included singular values reaches
95% of the total sum of singular values:

.. math::

    \\sum_{i=1}^{n_d'} \\sigma_i \leq 0.95\\sum_{i=1}^{n_d} \\sigma_i

Hence, the projection matrix :math:`[\\mathbf{W}]` is given by the first :math:`n_d'` rows of :math:`[\\mathbf{V}^*]`.

.. math::

    [\\mathbf{V}^*] =
    \\left[ \\begin{array}{c}
    [\\mathbf{W}] \\\\
    [\\mathbf{W}']
    \\end{array}\\right]

The advantage of the SVD approach is that the rotation is optimal in the L2 sense because the new random variables
:math:`\\eta` are aligned with the principal axes of the solution. Moreover, the calculation of the SVD of
:math:`\\left[\\mathbf{Y}_\\delta\\right]` is fast. The disadvantage of the approach is however, that the gradient of the
solution vector is required to determine the projection matrix :math:`[\\mathbf{W}]`. Depending on the chosen
:ref:`gradient calculation approach <label_gradient_calculation_approach>` this may result in additional
function evaluations. Once the gradients have been calculated, however, the gPC coefficients can be computed with higher
accuracy and less additional sampling points as it is described in the
:ref:`gradient enhanced gPC <label_gradient_enhanced_gpc>`. Accordingly, the choice of which method to select is
(as usual) highly dependent on the underlying problem and its compression capabilities.

It is noted that the projection matrix :math:`[\\mathbf{W}]` has to be determined for
each QoI separately.

The projection approach is implemented in the following algorithms:

* :ref:`Algorithm: StaticProjection`
* :ref:`Algorithm: RegAdaptiveProjection`
* :ref:`Algorithm: MEStaticProjection`
* :ref:`Algorithm: MERegAdaptiveProjection`

.. image:: /examples/images/FD_fwd.png
    :width: 500
    :align: center

Example
^^^^^^^
Lets consider the following :math:`n_d` dimensional testfunction:

.. math::

    y = \\cos \\left( 2 \\pi u + a\\sum_{i=1}^{n_d}\\xi_i \\right)

with :math:`u=0.5` and :math:`a=5.0`. Without loss of generality, we assume the two-dimensional case,
i.e. :math:`n_d=2`, for now. This function can be expressed by only one random variable :math:`\\eta`, which is a
linear combination of the original random variables :math:`\\xi`:

.. math::

    \\eta = \\sum_{i=1}^{n_d}\\xi_i,

This function is implemented in the :mod:`testfunctions <pygpc.testfunctions.testfunctions>` module of pygpc in
:class:`GenzOscillatory <pygpc.testfunctions.testfunctions.GenzOscillatory>`. In the following,
we will set up a static gPC with fixed order using the previously described projection approach to reduce the
original dimensionality of the problem.
"""

#%%
# Setting up the problem
# ----------------------

import pygpc
from collections import OrderedDict

# Loading the model and defining the problem
# ------------------------------------------

# Define model
model = pygpc.testfunctions.GenzOscillatory()

# Define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
problem = pygpc.Problem(model, parameters)

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["interaction_order"] = 1
options["n_cpu"] = 0
options["error_type"] = "nrmsd"
options["n_samples_validation"] = 1e3
options["error_norm"] = "relative"
options["matrix_ratio"] = 2
options["qoi"] = 0
options["fn_results"] = 'tmp/staticprojection'
options["save_session_format"] = ".pkl"
options["grid"] = pygpc.Random
options["grid_options"] = {"seed": 1}
options["n_grid"] = 100

#%%
# Since we have to compute the gradients of the solution anyway for the projection approach, we will make use of them
# also when determining the gPC coefficients. Therefore, we enable the "gradient_enhanced" gPC. For more details
# please see :ref:`gradient enhanced gPC <label_gradient_enhanced_gpc>`.
options["gradient_enhanced"] = True

#%%
# In the following we choose the :ref:`method to determine the gradients <label_gradient_calculation_approach>`.
# We will use a classical first order finite difference forward approximation for now.
options["gradient_calculation"] = "FD_fwd"
options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}

#%%
# We will use a 10th order approximation. It is noted that the model will consist of only one random variable.
# Including the mean (0th order coefficient) there will be 11 gPC coefficients in total.
options["order"] = [10]
options["order_max"] = 10

#%%
# Now we are defining the :class:`StaticProjection <pygpc.Algorithm.StaticProjection>` algorithm to solve the given problem.
algorithm = pygpc.StaticProjection(problem=problem, options=options)

#%%
# Running the gpc
# ---------------

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# Run gPC algorithm
session, coeffs, results = session.run()

#%%
# Inspecting the gpc object
# -------------------------
# The SVD of the gradients of the solution vector resulted in the following projection matrix reducing the problem
# from the two dimensional case to the one dimensional case:
print(f"Projection matrix [W]: {session.gpc[0].p_matrix}")

#%%
# It is of size :math:`n_d' \times n_d`, i.e. :math:`[1 \times 2]` in our case. Because of the simple sum of the
# random variables it can be seen directly from the projection matrix that the principal axis is 45° between the
# original parameter axes, exactly as the SVD of the gradient of the solution predicts. As a result,
# the number of gPC coefficients for a 10th order gPC approximation with only one random variable is 11:
print(f"Number of gPC coefficients: {session.gpc[0].basis.n_basis}")

#%%
# Accordingly, the gPC matrix has 11 columns, one for each gPC coefficient:
print(f"Size of gPC matrix: {session.gpc[0].gpc_matrix.shape}")

#%%
# It was mentioned previously that the one can make use of the
# :ref:`gradient enhanced gPC <label_gradient_enhanced_gpc>` when using the projection approach.
# Internally, the gradients are also rotated and the gPC matrix is extended by the gPC matrix
# containing the derivatives:
print(f"Size of gPC matrix containing the derivatives: {session.gpc[0].gpc_matrix_gradient.shape}")

#%%
# The random variables of the original and the reduced problem can be reviewed in:
print(f"Original random variables: {session.gpc[0].problem_original.parameters_random}")
print(f"Reduced random variables: {session.gpc[0].problem.parameters_random}")

#%%
# Postprocessing
# --------------
# The post-processing works identical as in a standard gPC. The routines identify whether the problem is reduced
# and provide all sensitivity measures with respect to the original model parameters.

# Post-process gPC and save sensitivity coefficients in .hdf5 file
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=False,
                             algorithm="sampling",
                             n_samples=10000)

# Get summary of sensitivity coefficients
sobol, gsens = pygpc.get_sens_summary(fn_gpc=options["fn_results"],
                                      parameters_random=session.parameters_random,
                                      fn_out=None)
print(f"\nSobol indices:")
print(f"==============")
print(sobol)

print(f"\nGlobal average derivatives:")
print(f"===========================")
print(gsens)

#%%
# Validation
# ----------
# Validate gPC vs original model function (2D-surface)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The possibility of parameter reduction becomes best clear if one visualizes the function values in the parameter
# space. In this simple example there is almost no difference between the original model (left) and the reduced gPC
# (center).

pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=list(problem.parameters_random.keys()),
                        n_grid=[51, 51],
                        output_idx=[0],
                        fn_out=None,
                        folder=None,
                        n_cpu=session.n_cpu)

# %%
# References
# ^^^^^^^^^^
# .. [1] Tipireddy, R., & Ghanem, R. (2014). Basis adaptation in homogeneous chaos spaces.
#    Journal of Computational Physics, 259, 304-317.
#
# .. [2] Tsilifis, P., Huan, X., Safta, C., Sargsyan, K., Lacaze, G., Oefelein, J. C., Najm, H. N.,
#    & Ghanem, R. G. (2019). Compressive sensing adaptation for polynomial chaos expansions.
#    Journal of Computational Physics, 380, 29-47.
#
