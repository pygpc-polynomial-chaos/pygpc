"""
Example: Modelling of an electrode
==================================

About the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This tutorial shows the application of pygpc to an equivalent electrical circuit, modelling the impedance of
an open-ended coaxial electrode.
The model consists of a Randles circuit that was modified according to the coaxial geometry of the electrode.
The parameters model the different contributions of the physical phenomena as follows:

1. **Rs** models the contribution of the serial resistance of an electrolyte that the electrode is dipped into.
2. **Qdl** models the distributed double layer capacitance of the electrode.
3. **Rct** models the charge transfer resistance between the electrode and the electrolyte
4. **Qd** and **Rd** model the diffusion of charge carriers and other particles towards the electrode surface.

The elements **Qdl** and **Qd** can be described with:
:math:`\\frac{1}{Q(j\\omega)^\\alpha}`
The equation depends on the angular frequency :math:`\\omega` as a variable and :math:`Q` and :math:`\\alpha`
as parameters.

The impedance of the equivalent circuit is complex valued, has seven parameters :math:`Rs`,  :math:`Rct`,  :math:`Rd`,
:math:`Qd`, :math:`\\alpha d`, :math:`Qdl`, :math:`\\alpha dl` and one variable :math:`\\omega`.
"""

import matplotlib.pyplot as plt

_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/modified_Randles_circuit.png"))
_ = plt.axis('off')


#%%
# Loading the model and defining the problem
# ------------------------------------------

import pygpc
import numpy as np
from collections import OrderedDict

fn_results = 'tmp/electrode'   # filename of output
save_session_format = ".hdf5"  # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)

# define model
model = pygpc.testfunctions.ElectrodeModel()

# define problem
parameters = OrderedDict()
# Set parameters
mu_n_Qdl = 0.67
parameters["n_Qdl"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_n_Qdl*0.9, mu_n_Qdl*1.1])
mu_Qdl = 6e-7
parameters["Qdl"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_Qdl*0.9, mu_Qdl*1.1])
mu_n_Qd = 0.95
mu_n_Qd_end = 1.0
parameters["n_Qd"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_n_Qd*0.9, mu_n_Qd_end])
mu_Qd = 4e-10
parameters["Qd"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_Qd*0.9, mu_Qd*1.1])
Rs_begin = 0
Rs_end = 1000
parameters["Rs"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[Rs_begin, Rs_end])
mu_Rct = 10e3
parameters["Rct"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_Rct*0.9, mu_Rct*1.1])
mu_Rd = 120e3
parameters["Rd"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[mu_Rd*0.9, mu_Rd*1.1])
# parameters["w"] = np.logspace(0, 9, 1000)
parameters["w"] = 2*np.pi*np.logspace(0, 9, 1000)
problem = pygpc.Problem(model, parameters)

#%%
# Setting up the algorithm
# ------------------------

# Set gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [5] * problem.dim
options["order_max"] = 5
options["interaction_order"] = 3
options["matrix_ratio"] = 3
options["error_type"] = "nrmsd"
options["n_samples_validation"] = 1e3
options["n_cpu"] = 0
options["fn_results"] = fn_results
options["save_session_format"] = '.pkl'
options["gradient_enhanced"] = False
options["gradient_calculation"] = "FD_1st2nd"
options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
options["backend"] = "omp"
options["grid"] = pygpc.Random
options["grid_options"] = None

# Define grid
n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                       order_glob_max=options["order_max"],
                                       order_inter_max=options["interaction_order"],
                                       dim=problem.dim)

grid = pygpc.Random(parameters_random=problem.parameters_random,
                    n_grid=options["matrix_ratio"] * n_coeffs,
                    seed=1)
# Define algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

#%%
# Running the gpc
# ---------------

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC algorithm
session, coeffs, results = session.run()

#%%
# Postprocessing
# --------------

# read session
session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

# Post-process gPC and add results to .hdf5 file
pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             n_samples=1e4)

#%%
# Validation
# ----------
# Validate gPC vs original model function (2D-surface)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Validate gPC vs original model function
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["Rd", "n_Qd"],
                        n_grid=[51, 51],
                        output_idx=500,
                        fn_out=None,
                        n_cpu=session.n_cpu)

#%%
# Validate gPC vs original model function (Monte Carlo)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
nrmsd = pygpc.validate_gpc_mc(session=session,
                              coeffs=coeffs,
                              n_samples=int(1e4),
                              output_idx=500,
                              n_cpu=session.n_cpu,
                              fn_out=None)

print("> Maximum NRMSD (gpc vs original): {:.2}%".format(max(nrmsd)))

#%%
# Mean and std of the real part of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Result
_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/modified_Randles_circuit_GPC_re.png"))
_ = plt.axis('off')

#%%
# Mean and std of the imaginary part of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Result
_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/modified_Randles_circuit_GPC_im.png"))
_ = plt.axis('off')

#%%
# Sobol indices of the parameters of the real part of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Result
_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/modified_Randles_circuit_GPC_sobol_re.png"))
_ = plt.axis('off')

#%%
# Sobol indices of the parameters of the imaginary part of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  Result
_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/modified_Randles_circuit_GPC_sobol_im.png"))
_ = plt.axis('off')
