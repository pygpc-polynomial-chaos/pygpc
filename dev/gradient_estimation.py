import pygpc
from collections import OrderedDict

# define model
model = pygpc.testfunctions.Peaks()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
parameters["x2"] = 1.25
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
problem = pygpc.Problem(model, parameters)

# define grid
n_grid = 100
grid = pygpc.Random(parameters_random=problem.parameters_random,
                    n_grid=n_grid,
                    seed=1)

# create grid points for finite difference approximation
grid.create_gradient_grid(delta=1e-3)

# evaluate model function
com = pygpc.Computation(n_cpu=0, matlab_model=False)

res = com.run(model=model,
              problem=problem,
              coords=grid.coords,
              coords_norm=grid.coords_norm,
              i_iter=None,
              i_subiter=None,
              fn_results=None,
              print_func_time=False)

grad_res_3D = pygpc.get_gradient(model=model,
                                 problem=problem,
                                 grid=grid,
                                 results=res,
                                 com=com,
                                 method="standard_forward",
                                 gradient_results=None,
                                 i_iter=None,
                                 i_subiter=None,
                                 print_func_time=False)

