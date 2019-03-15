import os
import pygpc
import numpy as np
from collections import OrderedDict
from examples.PyRates_Model import MyModel
from datetime import datetime
from pyrates.utility.grid_search import linearize_grid
from pyrates.cluster_compute.cluster_compute import read_cgs_results


id_ = datetime.now().strftime("%d%m%y-%H%M%S")
fn_results = f'/nobackup/spanien1/salomon/PyGPC/PyRates_GPC/{id_}/PyRates_GPC/'
os.makedirs(fn_results, exist_ok=True)

# define model
model = MyModel

# define problem (the parameter names have to be the same as in the model)
parameters = OrderedDict()
parameters["k_i"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[10., 20.])
parameters["k_e"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[20., 30.])
problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [3, 3]
options["order_max"] = 7
options["interaction_order"] = 2
options["n_cpu"] = 0    # n_cpu = 0; the model is capable of to compute multiple grid-points in parallel
options["fn_results"] = fn_results
options["print_func_time"] = True

# Load Results
##############
n_grid_1D = 101
params = OrderedDict()
params = {'k_e': np.linspace(parameters["k_e"].pdf_limits[0], parameters["k_e"].pdf_limits[1], n_grid_1D),
          'k_e': np.linspace(parameters["k_i"].pdf_limits[0], parameters["k_i"].pdf_limits[1], n_grid_1D)}
param_grid = linearize_grid(params, permute=True)
fp = "/nobackup/spanien1/salomon/ClusterGridSearch/Montbrio_EIC/Computation_2/Results/DefaultGrid_0/CGS_result_DefaultGrid_0.h5"
data = read_cgs_results(fp, key='Num_Peaks')
results = np.zeros(len(param_grid.index))

# assign results to grid points
for i_grid, k in enumerate(param_grid.values):
    results[i_grid] = data[k[1]][k[0]].values.item()
results = results[:, np.newaxis]

# Initialize Grid object
########################
grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                        options={"n_grid": n_grid_1D**2, "seed": 1})

grid.coords = param_grid.values
grid.coords_norm = grid.get_normalized_coordinates(grid.coords)

pygpc.plot_2d_grid(coords=grid.coords, fn_plot='_grid')

# Initialize Reg gPC object
###########################
gpc = pygpc.Reg(problem=problem,
                order=options["order"],
                order_max=options["order_max"],
                interaction_order=options["interaction_order"],
                fn_results=options["fn_results"])
gpc.grid = grid
gpc.init_gpc_matrix()

# determine gPC coeffs
######################
coeffs = gpc.solve(sim_results=results,
                   solver=options["solver"],
                   verbose=True)

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

eps = gpc.loocv(sim_results=results,
                solver=gpc.solver,
                settings=gpc.settings,
                n_loocv=100)

# Validate gPC vs original model function (2D-slice)
pygpc.validate_gpc_plot(gpc=gpc,
                        coeffs=coeffs,
                        random_vars=["k_i", "k_e"],
                        coords=grid.coords,
                        output_idx=[0],
                        data_original=results,
                        fn_out=options["fn_results"] + '_validation_2d')

# print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))

print("done!\n")
