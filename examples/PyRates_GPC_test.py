import pygpc
import numpy as np
from collections import OrderedDict
from examples.PyRates_Model import MyModel
from datetime import datetime
from pyrates.utility.grid_search import linearize_grid


# id_ = datetime.now().strftime("%d%m%y-%H%M%S")
# fn_results = f'/nobackup/spanien1/salomon/PyGPC/PyRates_GPC/{id_}/PyRates_GPC/'

parameters = OrderedDict()
params = {'k_e': np.linspace(20., 30., 21), 'k_i': np.linspace(10., 20., 21)}
param_grid = linearize_grid(params, permute=True)

parameters = OrderedDict()
parameters['k_e'] = param_grid.values[:, 0]
parameters['k_i'] = param_grid.values[:, 1]


# define model
# model = MyModel

# # define problem (the parameter names have to be the same as in the model)
# parameters = OrderedDict()
# parameters["k_i"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[15., 20.])
# parameters["k_e"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[15., 20.])
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
# # Post-process gPC
# pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
#                              output_idx=None,
#                              calc_sobol=True,
#                              calc_global_sens=True,
#                              calc_pdf=True)
#
# # # Validate gPC vs original model function (Monte Carlo)
# nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
#                               coeffs=coeffs,
#                               n_samples=int(1e4),
#                               output_idx=0,
#                               fn_out=options["fn_results"] + '_validation_mc')
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
