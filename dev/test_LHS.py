import os
from collections import OrderedDict

import pygpc

folder = "/tmp"
test_name = "LHS_1"
# define model
model = pygpc.testfunctions.Peaks()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
parameters["x2"] = pygpc.Norm(pdf_shape=[1.4, 0.6], p_perc=0.95)
parameters["x3"] = pygpc.Gamma(pdf_shape=[2, 2, 1.2], p_perc=0.95)
problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order"] = [9, 9, 9]
options["order_max"] = 9
options["interaction_order"] = 2
options["matrix_ratio"] = 2
options["error_type"] = "nrmsd"
options["n_samples_validation"] = 1e3
options["n_cpu"] = 0
options["fn_results"] = os.path.join(folder, test_name)
options["gradient_enhanced"] = True
options["GPU"] = False
options["grid"] = pygpc.LHS
options["grid_options"] = {"corr"}

# generate grid
n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                       order_glob_max=options["order_max"],
                                       order_inter_max=options["interaction_order"],
                                       dim=problem.dim)

grid = pygpc.Random(parameters_random=problem.parameters_random,
                    n_grid=options["matrix_ratio"] * n_coeffs,
                    seed=1,
                    options=options["grid_options"])

# define algorithm
algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC algorithm
session, coeffs, results = session.run()

session.gpc[0].grid.coords_norm
session.gpc[0].grid.coords

# Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="standard",
                             n_samples=1e3)


# Validate gPC vs original model function (2D-surface)
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["x1", "x2"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out=options["fn_results"] + "_val",
                        n_cpu=options["n_cpu"])

# Validate gPC vs original model function (Monte Carlo)
nrmsd = pygpc.validate_gpc_mc(session=session,
                              coeffs=coeffs,
                              n_samples=int(1e4),
                              output_idx=0,
                              fn_out=options["fn_results"] + "_pdf",
                              plot=True)