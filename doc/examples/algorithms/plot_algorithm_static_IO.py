"""
Algorithm: Static_IO
==============================
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

from collections import OrderedDict

fn_results = 'tmp/static_IO'   # filename of output
save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
np.random.seed(1)

#%%
# Setup input and output data
# ----------------------------------------------------------------------------------------------------------------

# We artificially generate some coordinates for the input data the user has to provide where the model was sampled
n_grid = 100
x1 = np.random.rand(n_grid) * 0.8 + 1.2
x2 = 1.25
x3 = np.random.rand(n_grid) * 0.6

# define the properties of the random variables
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])

# generate a grid object from the input data
grid = pygpc.RandomGrid(parameters_random=parameters, coords=np.vstack((x1,x3)).T)

# get output data (here: Peaks function)
results = (3.0 * (1 - x1) ** 2. * np.exp(-(x1 ** 2) - (x3 + 1) ** 2)
           - 10.0 * (x1 / 5.0 - x1 ** 3 - x3 ** 5)
           * np.exp(-x1 ** 2 - x3 ** 2) - 1.0 / 3
           * np.exp(-(x1 + 1) ** 2 - x3 ** 2)) +  x2
results = results[:, np.newaxis]

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "LarsLasso"
options["settings"] = None
options["order"] = [9, 9]
options["order_max"] = 9
options["interaction_order"] = 2
options["error_type"] = "loocv"
options["n_samples_validation"] = None
options["fn_results"] = fn_results
options["save_session_format"] = save_session_format
options["backend"] = "omp"
options["verbose"] = True

# define algorithm
algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results)

#%%
# Running the gpc
# ---------------

# initialize gPC Session
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
                             algorithm="standard")

# plot gPC approximation and IO data
pygpc.plot_gpc(session=session,
               coeffs=coeffs,
               random_vars=["x1", "x3"],
               output_idx=0,
               n_grid=[100, 100],
               coords=grid.coords,
               results=results,
               fn_out=None)

# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
