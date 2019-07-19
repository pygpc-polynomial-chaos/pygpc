import pygpc
import os
import numpy as np
from collections import OrderedDict
import matlab.engine
from templates.MyModel_matlab import MyModel_matlab

fn_results = "/home/kporzig/tmp"

# define model
model = MyModel_matlab(fun_path=os.path.join(pygpc.__path__[0], "testfunctions"))

# define problem (the parameter names have to be the same as in the model)
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
parameters["x3"] = 0.
parameters["a"] = 7.
parameters["b"] = 0.1

problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["order_start"] = 5
options["order_end"] = 20
options["solver"] = "LarsLasso"
options["interaction_order"] = 2
options["order_max_norm"] = 0.7
options["n_cpu"] = 0
options["adaptive_sampling"] = True
options["gradient_enhanced"] = True
options["fn_results"] = os.path.join(fn_results, "MyGPC_matlab")
options["eps"] = 0.0075
options["matlab_model"] = True

# define algorithm
algorithm = pygpc.RegAdaptive(problem=problem, options=options)

# run gPC algorithm
gpc, coeffs, results = algorithm.run()

# Post-process gPC and add results to .hdf5 file
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             n_samples=1e4)

# Validate gPC vs original model function
pygpc.validate_gpc_plot(gpc=gpc,
                        coeffs=coeffs,
                        random_vars=["x1", "x2"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out= gpc.options["fn_results"] + '_validation_plot',
                        n_cpu=options["n_cpu"])

# Validate gPC vs original model function (Monte Carlo)
nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                              coeffs=coeffs,
                              n_samples=int(1e4),
                              output_idx=0,
                              n_cpu=gpc.options["n_cpu"],
                              fn_out=gpc.options["fn_results"] + '_validation_mc')

print("done!\n")
