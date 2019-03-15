import pygpc
import numpy as np
from collections import OrderedDict
from examples.PyRates_Model import MyModel
from datetime import datetime


id_ = datetime.now().strftime("%d%m%y-%H%M%S")
fn_results = f'/nobackup/spanien1/salomon/PyGPC/PyRates_GPC/{id_}/PyRates_GPC/'

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
options["order_max"] = 3
options["interaction_order"] = 2
options["n_cpu"] = 0    # n_cpu = 0; the model is capable of to compute multiple grid-points in parallel
options["fn_results"] = fn_results
options["print_func_time"] = True

grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                        options={"n_grid": 101*101, "seed": 1})

grid.coords = ... # [101*101 x 2]
grid.coords_norm = grid.get_denormalized_coordinates(grid.coords)

pygpc.plot_2d_grid(coords=grid.coords, fn_plot='_grid')

# define algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

# run gPC algorithm
gpc, coeffs, results = algorithm.run()

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

# Validate gPC vs original model function (2D-slice)
pygpc.validate_gpc_plot(gpc=gpc,
                        coeffs=coeffs,
                        random_vars=["k_i", "k_e"],
                        n_grid=[21, 21],
                        output_idx=[0],
                        fn_out=options["fn_results"] + '_validation_2d',
                        n_cpu=0)

# print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))

print("done!\n")
