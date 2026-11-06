"""
Spinal-peripheral MEP model
=====================================================

This example applies pygpc to a spinal-peripheral model that spans the
corticospinal input, the spinal circuitry and the peripheral hand muscles. The
model was designed to fit individual motor evoked potential (MEP) waveforms
across a range of intensities of transcranial magnetic stimulation (TMS).

MEP waveforms are obtained by summing the motor unit action potentials (MUAPs)
of all motor units (MUs) at the times their motor neurons fire. To keep the
example fast, the spinal circuitry is not re-simulated: the motor neuron spike
times were fitted beforehand for one subject and are loaded from disk. Each
pygpc sample therefore only regenerates the MUAPs and superposes them at the
stored spike times. The model output is the coefficient of determination
:math:`R^2` between the simulated and the measured MEPs.

MUAP model
----------

Each MUAP is modelled with a first-order Hermite-Rodriguez function, normalized so that its positive peak equals the amplitude :math:`A_i`:

.. math::

    H_i(t) = \frac{A_i}{h_{\max}} \, (\tau - t) \,
             \exp\left(-\left(\frac{\tau - t}{\lambda}\right)^2\right),
    \qquad
    h_{\max} = \max_t \left[ (\tau - t) \,
               \exp\left(-\left(\frac{\tau - t}{\lambda}\right)^2\right) \right]

where :math:`t` is the time after the motor neuron spike, :math:`\lambda` is
the shape parameter controlling the MUAP duration, and :math:`\tau` is the
MUAP zero-crossing time. The axonal delay is tied to the MUAP duration and is the same for
all MUs:

.. math::

    \tau = 2.35\,\lambda - 0.18 \ \text{ms}

The MUAP amplitudes follow an exponential distribution over the :math:`N = 100`
MUs, ordered by recruitment:

.. math::

    A_i = b^{\frac{i-1}{N-1}}, \qquad i = 1, \dots, N


Fixed settings
--------------

These settings belong to the spinal circuitry model and describe how the
stored spike times were generated:

* ``SUBJ``: subject whose MEPs are fitted
* ``WITHRC``: if 1, Renshaw cells are included in the spinal circuitry model
* ``AMPAWEIGHT``: fraction of AMPA-mediated synaptic currents; if ``None``,
  it was optimized during the fit
* ``SPIKE_FILE``: file containing the previously fitted motor neuron spike times

Random parameters
-----------------

The sensitivity analysis is performed on the two MUAP parameters:

* ``b``: MUAP amplitude parameter
* ``lam``: MUAP shape parameter
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import os
import h5py
import pygpc
from collections import OrderedDict
from pygpc.testfunctions import MUAP_gpc

fn_results = "tmp/example_MEP"

if os.path.exists(fn_results + ".hdf5"):
    os.remove(fn_results + ".hdf5")

if os.path.exists(fn_results + "_val.hdf5"):
    os.remove(fn_results + "_val.hdf5")

if os.path.exists(fn_results + "_mc.hdf5"):
    os.remove(fn_results + "_mc.hdf5")

# define model
model = MUAP_gpc()

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

session = pygpc.io.read_session(fname=fn_session, folder=fn_session_folder)

#%%
# Validation
# ^^^^^^^^^^
# Validate gPC vs original model function
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


#%%
# Postprocessing
# ^^^^^^^^^^^^^^
# Postprocess gPC and add results to .hdf5 file
pygpc.get_sensitivities_hdf5(fn_gpc=session.fn_results,
                                output_idx=None,
                                calc_sobol=True,
                                calc_global_sens=True,
                                calc_pdf=True,
                                n_samples=int(1e4))

#%%
# Sensitivity analysis
# ^^^^^^^^^^^^^^^^^^^^
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