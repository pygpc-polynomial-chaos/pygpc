import pygpc
import numpy as np
import pandas as pd
import time as t
from collections import OrderedDict
from examples.PyRates_Model import MyModel
from datetime import datetime
from pyrates.utility.grid_search import linearize_grid
from pyrates.utility import plot_connectivity
import matplotlib.pyplot as plt
import h5py
import os

# id_ = datetime.now().strftime("%d%m%y-%H%M%S")
# fn_results = f'/nobackup/spanien1/salomon/PyGPC/PyRates_GPC/{id_}/PyRates_GPC'
# os.makedirs(fn_results, exist_ok=True)
#
#
# num_ke = num_ki = 3
# params = {'k_e': np.linspace(25., 30., num_ke), 'k_i': np.linspace(10., 25., num_ki)}
# param_grid = linearize_grid(params, permute=True)
# coords = param_grid.values
# params = {'k_e': coords[:, 0],
#           'k_i': coords[:, 1]}
#
# # define model
# t0 = t.time()
# model = MyModel(p=params, context={})
# data, _ = model.simulate(process_id=0)
# print(data)
# print(f'Computation finished. Elapsed time: {t.time()-t0} seconds')
#
# with h5py.File(f'{fn_results}/temp.h5', 'w') as file:
#     file.create_dataset(name='Data', data=data)

# Plot results
with h5py.File(f'/nobackup/spanien1/salomon/PyGPC/PyRates_GPC/150319-104403/PyRates_GPC/temp.h5', 'r') as file:
    data = file['Data'][:]
    # num_ke = num_ki = 3
num_ke = num_ki = 3
results = data.reshape(num_ke, num_ki, order='F')
print(results)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 15), gridspec_kw={})
plot_connectivity(results, ax=ax)
plt.show()




# # define problem (the parameter names have to be the same as in the model)
# parameters = OrderedDict()
# parameters["k_i"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[10., 20.])
# parameters["k_e"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[20., 30.])
# problem = pygpc.Problem(model, parameters)
#
# # gPC options
# options = dict()
# options["order_start"] = 1
# options["order_end"] = 10
# options["interaction_order"] = 2
# options["solver"] = "Moore-Penrose"
# options["settings"] = None
# options["seed"] = 1
# options["matrix_ratio"] = 2
# options["eps"] = 1e-3
# options["n_cpu"] = 0    # n_cpu = 0; the model is capable of to compute multiple grid-points in parallel
# options["fn_results"] = fn_results
# options["print_func_time"] = True
#
# # define algorithm
# algorithm = pygpc.RegAdaptive(problem=problem, options=options)
#
# # run gPC algorithm
# gpc, coeffs, results = algorithm.run()
#
# # # plot 2D grid (only feasible for 2D problems)
# pygpc.plot_2d_grid(coords=gpc.grid.coords,
#                    fn_plot=fn_results + '_grid')
#
# # # Post-process gPC
# # pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
# #                              output_idx=None,
# #                              calc_sobol=True,
# #                              calc_global_sens=True,
# #                              calc_pdf=True)
# #
# # # # Validate gPC vs original model function (Monte Carlo)
# # nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
# #                               coeffs=coeffs,
# #                               n_samples=int(1e4),
# #                               output_idx=0,
# #                               fn_out=options["fn_results"] + '_validation_mc')
#
# # Validate gPC vs original model function (2D-slice)
# pygpc.validate_gpc_plot(gpc=gpc,
#                         coeffs=coeffs,
#                         random_vars=["k_i", "k_e"],
#                         n_grid=[21, 21],
#                         output_idx=[0],
#                         fn_out=options["fn_results"] + '_validation_2d',
#                         n_cpu=0)
#
# # print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))
#
# print("done!\n")
