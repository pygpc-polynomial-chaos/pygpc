import pygpc
import os
import numpy as np
import time
from collections import OrderedDict


folder = "tmp"
test_name = "benchmark pygpc_extensions"

dimensions_lst = [4]
order_lst = np.linspace(5, 10, 5)
n_samples_validation_lst = np.logspace(2, 6, 5)
n_gpc_coeffs_lst = [100000]

time_python_lst = []
time_omp_lst = []
n_basis_lst = []

for i_n_gpc_coeffs in n_gpc_coeffs_lst:
    for i_dimensions in dimensions_lst:
        for i_order in order_lst:
            for i_n_samples_validation in n_samples_validation_lst:

                print("----------")

                n_basis = pygpc.get_num_coeffs_sparse(i_order, i_order, i_order, i_dimensions)

                print("Number of basis functions: ", n_basis)
                print("Number of random samples", i_n_samples_validation)

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
                                order_max_norm=1,
                                interaction_order=i_order,
                                interaction_order_current=i_order,
                                options=options)
                gpc.grid = grid
                # generate gpc coeffs matrix
                coeffs = np.ones([len(gpc.basis.b), i_n_gpc_coeffs])

                start = time.time()

                # run function
                gpc.get_approximation(coeffs, gpc.grid.coords_norm)

                time_python = time.time() - start
                time_python_lst.append(time_python)

                print("Time Python: ", time_python)

                # run benchmark for omp implementation
                options["backend"] = "omp"
                # setup gPC
                gpc = pygpc.Reg(problem=problem,
                                order=[i_order] * i_dimensions,
                                order_max=i_order,
                                order_max_norm=1,
                                interaction_order=i_order,
                                interaction_order_current=i_order,
                                options=options)
                gpc.grid = grid
                # generate gpc coeffs matrix
                coeffs = np.ones([len(gpc.basis.b), i_n_gpc_coeffs])

                start = time.time()

                # run function
                gpc.get_approximation(coeffs, gpc.grid.coords_norm)

                time_omp = time.time() - start
                time_omp_lst.append(time_omp)

                print("Time OpenMP:", time_omp)
                print("Performance gain: ", time_python/time_omp)

time_python_array = np.array(time_python_lst)
time_omp_array = np.array(time_omp_lst)
