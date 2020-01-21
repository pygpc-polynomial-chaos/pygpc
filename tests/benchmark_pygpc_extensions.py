import pygpc
import os
import numpy as np
import time
from collections import OrderedDict


folder = "./tmp"
test_name = "benchmark pygpc_extensions"

dimensions_lst = [8]
# order_lst = np.linspace(1, 10, 10)
# n_samples_validation_lst = np.logspace(2, 6, 5)
order_lst = [10]
n_samples_validation_lst = [1000000]
time_python_lst = []
time_cpu_lst = []
n_basis_lst = []

for i_dimensions in dimensions_lst:
    for i_order in order_lst:
        for i_n_samples_validation in n_samples_validation_lst:

            # define model
            model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

            # define problem
            parameters = OrderedDict()
            for i_local_parameters in range(i_dimensions):
                name = "x" + str(i_local_parameters)
                parameters[name] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
            problem = pygpc.Problem(model, parameters)

            # gPC options
            options = dict()
            options["gradient_enhanced"] = False
            options["matlab_model"] = False

            # define test grid
            grid = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                                    options={"n_grid": 100, "seed": 1})

            # run benchmark for python implementation
            options["backend"] = "python"
            # setup gPC
            gpc = pygpc.Reg(problem=problem,
                            order=[i_order]*i_dimensions,
                            order_max=i_order,
                            order_max_norm=0.8,
                            interaction_order=2,
                            interaction_order_current=2,
                            options=options)
            gpc.grid = grid

            start = time.time()

            # run function
            gpc.init_gpc_matrix()

            time_python = time.time() - start
            time_python_lst.append(time_python)

            # run benchmark for cpu implementation
            options["backend"] = "cpu"
            # setup gPC
            gpc = pygpc.Reg(problem=problem,
                            order=[i_order] * i_dimensions,
                            order_max=i_order,
                            order_max_norm=0.8,
                            interaction_order=2,
                            interaction_order_current=2,
                            options=options)
            gpc.grid = grid

            start = time.time()

            # run function
            gpc.init_gpc_matrix()

            time_cpu = time.time() - start
            time_cpu_lst.append(time_cpu)

time_python_array = np.array(time_python_lst)
time_cpu_array = np.array(time_cpu_lst)

print(time_python_array/time_cpu_array)
