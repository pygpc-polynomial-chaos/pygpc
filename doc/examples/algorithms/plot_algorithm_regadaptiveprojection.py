"""
Algorithm: RegAdaptiveProjection
================================
:: _label_algorithm_regadaptive_projection:
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
from collections import OrderedDict

fn_results = 'tmp/regadaptiveprojection'   # filename of output
save_session_format = ".pkl"              # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)

#%%
# Loading the model and defining the problem
# ------------------------------------------

# define model
model = pygpc.testfunctions.GenzOscillatory()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
problem = pygpc.Problem(model, parameters)

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["order_start"] = 2
options["order_end"] = 15
options["interaction_order"] = 2
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["seed"] = 1
options["matrix_ratio"] = 2
options["n_cpu"] = 0
options["fn_results"] = fn_results
options["save_session_format"] = save_session_format
options["adaptive_sampling"] = False
options["gradient_enhanced"] = True
options["gradient_calculation"] = "FD_1st"
options["gradient_calculation_options"] = {"dx": 0.5, "distance_weight": -2}
options["n_grid_gradient"] = 5
options["qoi"] = 0
options["error_type"] = "loocv"
options["eps"] = 1e-3
options["grid"] = pygpc.Random
options["grid_options"] = {"seed": 1}

# define algorithm
algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)

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

# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
