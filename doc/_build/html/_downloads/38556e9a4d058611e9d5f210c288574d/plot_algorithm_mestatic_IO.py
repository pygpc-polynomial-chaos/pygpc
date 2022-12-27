"""
Algorithm: MEStatic_IO
==============================
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

fn_results = 'tmp/mestatic_IO'  # filename of output
save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
np.random.seed(1)

#%%
# Setup input and output data
# ----------------------------------------------------------------------------------------------------------------

# We artificially generate some coordinates for the input data the user has to provide where the model was sampled
n_grid = 400
rho_0 = np.random.rand(n_grid)
beta = np.random.rand(n_grid) * 20.
alpha = 1.

# define the properties of the random variables
parameters = OrderedDict()
parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])

# generate a grid object from the input data
grid = pygpc.RandomGrid(parameters_random=parameters, coords=np.vstack((rho_0,beta)).T)

# get output data (here: SurfaceCoverageSpecies function)
def deq(rho, t, alpha, beta, gamma):
    return alpha * (1. - rho) - gamma * rho - beta * (rho - 1) ** 2 * rho

# Constants
gamma = 0.01

# Simulation parameters
dt = 0.01
t_end = 1.
t = np.arange(0, t_end, dt)

# Solve
results = odeint(deq, rho_0, t, args=(alpha, beta, gamma))[-1][:, np.newaxis]

#%%
# Setting up the algorithm
# ------------------------

# gPC options
options = dict()
options["solver"] = "LarsLasso"
options["settings"] = None
options["order"] = [9, 9]
options["order_max"] = 9
options["interaction_order"] = 2
options["matrix_ratio"] = None
options["n_cpu"] = 0
options["error_type"] = "loocv"
options["qoi"] = "all"
options["classifier"] = "learning"
options["classifier_options"] = {"clusterer": "KMeans",
                                 "n_clusters": 2,
                                 "classifier": "MLPClassifier",
                                 "classifier_solver": "lbfgs"}
options["fn_results"] = fn_results
options["save_session_format"] = save_session_format
options["verbose"] = True

# define algorithm
algorithm = pygpc.MEStatic_IO(parameters=parameters, options=options, grid=grid, results=results)

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
                             algorithm="standard",
                             n_samples=int(1e4))

# plot gPC approximation and IO data
pygpc.plot_gpc(session=session,
               coeffs=coeffs,
               random_vars=["rho_0", "beta"],
               output_idx=0,
               n_grid=[100, 100],
               coords=grid.coords,
               results=results,
               fn_out=None,
               camera_pos=[45., 65])

# On Windows subprocesses will import (i.e. execute) the main module at start.
# You need to insert an if __name__ == '__main__': guard in the main module to avoid
# creating subprocesses recursively.
#
# if __name__ == '__main__':
#     main()
