import pygpc
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

methods = ["FD_fwd", "FD_1st", "FD_2nd", "FD_1st2nd"]
dx = [1e-3, 0.02, 0.02, 0.02]

# define model
model = pygpc.testfunctions.Peaks()

# define problem
parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
parameters["x2"] = 1.
parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
problem = pygpc.Problem(model, parameters)

# define grid
n_grid = 1000
grid = pygpc.Random(parameters_random=problem.parameters_random,
                    n_grid=n_grid,
                    seed=1)

# create grid points for finite difference approximation
grid.create_gradient_grid(delta=1e-3)

# evaluate model function
com = pygpc.Computation(n_cpu=0, matlab_model=False)

# [n_grid x n_out]
res = com.run(model=model,
              problem=problem,
              coords=grid.coords,
              coords_norm=grid.coords_norm,
              i_iter=None,
              i_subiter=None,
              fn_results=None,
              print_func_time=False)

df = pd.DataFrame(columns=["method", "nrmsd", "coverage"])
grad_res = dict()

for i_m, m in enumerate(methods):
    # [n_grid x n_out x dim]
    grad_res[m] = pygpc.get_gradient(model=model,
                                     problem=problem,
                                     grid=grid,
                                     results=res,
                                     com=com,
                                     method=m,
                                     gradient_results=None,
                                     i_iter=None,
                                     i_subiter=None,
                                     print_func_time=False,
                                     dx=dx[i_m],
                                     distance_weight=-2)

    if m != "FD_fwd":
        df.loc[i_m, "method"] = m
        df.loc[i_m, "coverage"] = 1-(np.sum(np.isnan(grad_res[m][:, 0, 0]))/n_grid)
        df.loc[i_m, "nrmsd"] = pygpc.nrmsd(grad_res[m][:, 0, :], grad_res["FD_fwd"][:, 0, :])

for i in range(problem.dim):
    plt.figure()
    for i_m, m in enumerate(methods):
        plt.stem(grad_res[m][:, 0, i], markerfmt=f"C{i_m}o", linefmt=f"C{i_m}-")
    plt.legend(methods)
    plt.title(f"$df/dx_{i}$")
