"""
Example: Ishigami Testfunction
==============================

About the model
^^^^^^^^^^^^^^^
This easy tutorial shows the application of pygpc to the Ishigami function, which can be found in the testfunctions
section.
The model consists of three random variables that are considered as input parameters (*x1*, *x2*, *x3*). The shape
parameters of the function are chosen to be :code:`a=7` and :code:`b=0.1`.

The model returns an output array with a value *y* for every sampling point.
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import os
import h5py
import pooch
import pygpc
from collections import OrderedDict
from pygpc.testfunctions import MUAP_gpc

#%%
# Download the model data
# ^^^^^^^^^^^^^^^^^^^^^^^
# The MEP model needs measured data that is not shipped with pygpc. It is hosted on Zenodo and downloaded with
# `pooch <https://www.fatiando.org/pooch/>`_ on the first run. The files are cached locally (default: the OS cache
# folder, e.g. ~/.cache/pygpc on Linux; override with the environment variable PYGPC_DATA_DIR) and verified by
# their SHA256 hashes, so subsequent runs do not download them again.
ZENODO_DOI = "10.5281/zenodo.XXXXXXX"  # TODO: replace with the DOI of the Zenodo record (version DOI)

data_fetcher = pooch.create(
    path=pooch.os_cache("pygpc"),
    base_url=f"doi:{ZENODO_DOI}/",
    env="PYGPC_DATA_DIR",
    registry={
        "S1.h5": "sha256:f8e280c4b88a72e0765a502ebe95934e22296d1cd41a0c66df33a7ba47aa6ede",
        "AP.h5": "sha256:71ae734270495766b0bee3697f476b76760dec324399382e5d09728a4c0bc895",
        "mu_spiketimes_S1.h5": "sha256:8e690e1783b8e1350dfad95314fd21a60f90b0e7499704532eb9956531be7277",
    },
)

for fname in data_fetcher.registry:
    data_fetcher.fetch(fname, progressbar=True)

data_dir = str(data_fetcher.abspath)

fn_results = "pygpc/test"

if os.path.exists(fn_results + ".hdf5"):
    os.remove(fn_results + ".hdf5")

if os.path.exists(fn_results + "_val.hdf5"):
    os.remove(fn_results + "_val.hdf5")

if os.path.exists(fn_results + "_mc.hdf5"):
    os.remove(fn_results + "_mc.hdf5")

# define model
model = MUAP_gpc(data_dir=data_dir)

# define problem
parameters = OrderedDict()
# MUAP parameters
parameters["b"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[45, 425]) # MUAP amplitude parameter. pdf_limits - sampling range
parameters["lam"] = pygpc.Norm(pdf_shape=[3.619, 0.774]) # MUAP shape parameter. Normal distribution. pdf share mu, std

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["order"] = [4] * problem.dim
options["order_max"] = 15
options["order_start"] = 2
options["method"] = 'reg'
options["solver"] = "Moore-Penrose"
options["interaction_order"] = 2
options["order_max_norm"] = 1.0
options["n_cpu"] = 0
options["eps"] = 0.1
options["fn_results"] = fn_results
options["basis_increment_strategy"] = None
options["plot_basis"] = False
options["n_grid"] = 1300
options["save_session_format"] = ".hdf5"
options["matrix_ratio"] = 2
options["grid"] = pygpc.LHS
options["error_type"] = "nrmsd"
options["grid_options"] = {"seed": 1, 'criterion': 'ese'}

# define algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=None)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results = session.run()


# Load results
with h5py.File(fn_results + ".hdf5", "r") as f:
    coeffs = f["coeffs"][:]
    # filename of associated gPC .pkl files
    fn_session = os.path.join(os.path.split(fn_results)[0], f["misc/fn_session"][0].astype(str))
    fn_session_folder = f["misc/fn_session_folder"][0].astype(str)

# print("> Loading gpc session object: {}".format(fn_session))
session = pygpc.io.read_session(fname=fn_session, folder=fn_session_folder)

# Validation
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["b", "lam"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out=session.fn_results + '_val',
                        n_cpu=session.n_cpu)

nrmsd = pygpc.validate_gpc_mc(session=session,
                                coeffs=coeffs,
                                n_samples=int(1e4),
                                output_idx=0,
                                n_cpu=session.n_cpu,
                                fn_out=session.fn_results + '_mc')


# Sensitivity analysis
pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                                output_idx=None,
                                calc_sobol=True,
                                calc_global_sens=True,
                                calc_pdf=True,
                                n_samples=int(1e4))

sobol, gsens = pygpc.get_sens_summary(fn_results, parameters, fn_results + "_sens_summary.txt")
pygpc.plot_sens_summary(sobol=sobol, gsens=gsens, fn_plot=session.fn_results + "_sens_summary.pdf")

#
# import os
# import pygpc
# import numpy as np
# from collections import OrderedDict
# import matplotlib
# # matplotlib.use("Qt5Agg")
#
# fn_results = "tmp/example_ishigami"
#
# if os.path.exists(fn_results + ".hdf5"):
#     os.remove(fn_results + ".hdf5")
#
# if os.path.exists(fn_results + "_val.hdf5"):
#     os.remove(fn_results + "_val.hdf5")
#
# if os.path.exists(fn_results + "_mc.hdf5"):
#     os.remove(fn_results + "_mc.hdf5")
#
# # define model
# model = pygpc.testfunctions.Ishigami()
#
# # define problem
# parameters = OrderedDict()
# parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters["a"] = 7.
# parameters["b"] = 0.1
#
# parameters_random = OrderedDict()
# parameters_random["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters_random["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
# parameters_random["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
#
# problem = pygpc.Problem(model, parameters)
#
# # gPC options
# options = dict()
# options["order"] = [15] * problem.dim
# options["order_max"] = 15
# options["order_start"] = 15
# options["method"] = 'reg'
# options["solver"] = "Moore-Penrose"
# options["interaction_order"] = 2
# options["order_max_norm"] = 1.0
# options["n_cpu"] = 0
# options["eps"] = 0.01
# options["fn_results"] = fn_results
# options["basis_increment_strategy"] = None
# options["plot_basis"] = False
# options["n_grid"] = 1300
# options["save_session_format"] = ".pkl"
# options["matrix_ratio"] = 2
# options["grid"] = pygpc.Random
# options["grid_options"] = {"seed": 1}
#
# # define algorithm
# algorithm = pygpc.Static(problem=problem, options=options, grid=None)
#
# # Initialize gPC Session
# session = pygpc.Session(algorithm=algorithm)
#
# # run gPC session
# session, coeffs, results = session.run()
#
# #%%
# # Postprocessing
# # ^^^^^^^^^^^^^^
# # Postprocess gPC and add results to .hdf5 file
# pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
#                              output_idx=None,
#                              calc_sobol=True,
#                              calc_global_sens=True,
#                              calc_pdf=True,
#                              n_samples=int(1e4))
#
# #%%
# # Validation
# # ^^^^^^^^^^
# # Validate gPC vs original model function
# pygpc.validate_gpc_plot(session=session,
#                         coeffs=coeffs,
#                         random_vars=["x1", "x2"],
#                         n_grid=[51, 51],
#                         output_idx=0,
#                         fn_out=session.fn_results + '_val',
#                         n_cpu=session.n_cpu)
#
# #%%
# # Validate gPC vs original model function (Monte Carlo)
# nrmsd = pygpc.validate_gpc_mc(session=session,
#                               coeffs=coeffs,
#                               n_samples=int(1e4),
#                               output_idx=0,
#                               n_cpu=session.n_cpu,
#                               fn_out=session.fn_results + '_mc')
#
# #%%
# # Sensitivity analysis
# # ^^^^^^^^^^^^^^^^^^^^
# sobol, gsens = pygpc.get_sens_summary(fn_results, parameters_random, fn_results + "_sens_summary.txt")
# pygpc.plot_sens_summary(sobol=sobol, gsens=gsens)
#
# #
# # On Windows subprocesses will import (i.e. execute) the main module at start.
# # You need to insert an if __name__ == '__main__': guard in the main module to avoid
# # creating subprocesses recursively.
# #
# # if __name__ == '__main__':
# #     main()