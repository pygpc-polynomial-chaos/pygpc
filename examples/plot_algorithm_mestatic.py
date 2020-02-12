"""
Algorithm: MEStatic
===================
"""
import pygpc
from collections import OrderedDict

fn_results = 'tmp/mestatic'       # filename of output
save_session_format = ".hdf5"     # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)

#%%
# Loading the model and defining the problem
# ------------------------------------------

# define model
model = pygpc.testfunctions.SurfaceCoverageSpecies()

# define problem
parameters = OrderedDict()
parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])
parameters["alpha"] = 1.
problem = pygpc.Problem(model, parameters)

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [9, 9]
options["order_max"] = 9
options["interaction_order"] = 2
options["matrix_ratio"] = 2
options["n_cpu"] = 0
options["gradient_enhanced"] = True
options["gradient_calculation"] = "FD_2nd"
options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
options["error_type"] = "loocv"
options["qoi"] = "all"
options["n_grid_gradient"] = 5
options["classifier"] = "learning"
options["classifier_options"] = {"clusterer": "KMeans",
                                 "n_clusters": 2,
                                 "classifier": "MLPClassifier",
                                 "classifier_solver": "lbfgs"}
options["fn_results"] = fn_results
options["save_session_format"] = save_session_format
options["grid"] = pygpc.Random
options["grid_options"] = None

# generate grid
grid = pygpc.Random(parameters_random=problem.parameters_random,
                    n_grid=1000,  # options["matrix_ratio"] * n_coeffs
                    seed=1)

# define algorithm
algorithm = pygpc.MEStatic(problem=problem, options=options, grid=grid)

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

# Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="sampling",
                             n_samples=1e3)

#%%
# Validation
# ----------
# Validate gPC vs original model function (2D-surface)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=list(problem.parameters_random.keys()),
                        n_grid=[51, 51],
                        output_idx=[0],
                        fn_out=None,
                        folder=None,
                        n_cpu=session.n_cpu)
#%%
# Validate gPC vs original model function (Monte Carlo)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
nrmsd = pygpc.validate_gpc_mc(session=session,
                              coeffs=coeffs,
                              n_samples=int(1e4),
                              output_idx=[0],
                              fn_out=None,
                              folder=None,
                              plot=True,
                              n_cpu=session.n_cpu)

print("> Maximum NRMSD (gpc vs original): {:.2}%".format(max(nrmsd)))