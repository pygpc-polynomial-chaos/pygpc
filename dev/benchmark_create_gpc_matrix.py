import pygpc
import os
import numpy as np
import time
from collections import OrderedDict

dimensions_lst = [4]

order_lst = np.array([1, 5, 10, 11])
n_samples_validation_lst = np.logspace(1, 6, 10)

# order_lst = [2]
# n_samples_validation_lst = [10]

time_python_lst = []
time_cpu_lst = []
time_omp_lst = []
time_cuda_lst = []
n_basis_lst = []
grid = []
gpc = []

# define model
model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

# Construct test grids
for dimensions in dimensions_lst:
    print("Initializing grids")
    print("==================")

    # define problem
    parameters = OrderedDict()
    for i_local_parameters in range(dimensions):
        name = "x" + str(i_local_parameters)
        parameters[name] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    problem = pygpc.Problem(model, parameters)

    for n_samples_validation in n_samples_validation_lst:

        start = time.time()

        grid.append(pygpc.Random(parameters_random=problem.parameters_random,
                                     n_grid=n_samples_validation,
                                     options={"n_grid": n_samples_validation, "seed": 1}))

        time_grid = time.time() - start
        print(f"Grid (N={n_samples_validation}): {time_grid}")


# Init gPCs
for dimensions in dimensions_lst:
    print("Initializing GPCs")
    print("=================")

    # define problem
    parameters = OrderedDict()
    for i_local_parameters in range(dimensions):
        name = "x" + str(i_local_parameters)
        parameters[name] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    problem = pygpc.Problem(model, parameters)

    for order in order_lst:
        n_basis = pygpc.get_num_coeffs_sparse([order] * dimensions, order, dimensions, dimensions, dimensions, 1)
        n_basis_lst.append(n_basis)

        # gPC options
        options = dict()
        options["gradient_enhanced"] = False
        options["matlab_model"] = False

        # run benchmark for python implementation
        options["backend"] = "python"
        # setup gPC
        start = time.time()
        gpc.append(pygpc.Reg(problem=problem,
                             order=[order] * dimensions,
                             order_max=order,
                             order_max_norm=1,
                             interaction_order=dimensions,
                             interaction_order_current=dimensions,
                             options=options))
        time_gpc = time.time() - start
        print(f"GPC (M={n_basis}): {time_gpc}")

# run benchmark
for dimensions in dimensions_lst:
    print(f"Dim: {dimensions}")
    for i_o, order in enumerate(order_lst):
        print("Order: ", order)
        print("==========")
        local_time_python_lst = []
        local_time_cpu_lst = []
        local_time_omp_lst = []
        local_time_cuda_lst = []
        for i_n_s, n_samples_validation in enumerate(n_samples_validation_lst):
            print("n_samples: ", n_samples_validation)
            n_basis = pygpc.get_num_coeffs_sparse([order]*dimensions, order, dimensions, dimensions, dimensions, 1)
            n_basis_lst.append(n_basis)
            print("n_basis: ", n_basis)

            gpc[i_o].backend = "python"
            gpc[i_o].grid = grid[i_n_s]

            start = time.time()

            # run function
            gpc[i_o].create_gpc_matrix(b=gpc[i_o].basis.b, x=gpc[i_o].grid.coords_norm)

            time_python = time.time() - start
            print("Python: ", time_python)
            local_time_python_lst.append(time_python)

            # run benchmark for cpu implementation
            gpc[i_o].backend = "cpu"
            gpc[i_o].grid = grid[i_n_s]

            start = time.time()

            # run function
            gpc[i_o].create_gpc_matrix(b=gpc[i_o].basis.b, x=gpc[i_o].grid.coords_norm)

            time_cpu = time.time() - start
            print("CPU: ", time_cpu)
            local_time_cpu_lst.append(time_cpu)

            # run benchmark for omp implementation
            gpc[i_o].backend = "omp"
            gpc[i_o].grid = grid[i_n_s]

            start = time.time()

            # run function
            gpc[i_o].create_gpc_matrix(b=gpc[i_o].basis.b, x=gpc[i_o].grid.coords_norm)

            time_omp = time.time() - start
            print("OMP: ", time_omp)
            local_time_omp_lst.append(time_omp)

            try:
                # run benchmark for cuda implementation
                gpc[i_o].backend = "cuda"
                gpc[i_o].grid = grid[i_n_s]

                start = time.time()

                # run function
                gpc[i_o].create_gpc_matrix(b=gpc[i_o].basis.b, x=gpc[i_o].grid.coords_norm)

                time_cuda = time.time() - start
                print("CUDA: ", time_cuda)
                local_time_cuda_lst.append(time_cuda)
            except NotImplementedError:
                pass

            print("-----------")

        # outer lists
        time_python_lst.append(local_time_python_lst)
        time_cpu_lst.append(local_time_cpu_lst)
        time_omp_lst.append(local_time_omp_lst)
        time_cuda_lst.append(local_time_cuda_lst)

time_python_array = np.array(time_python_lst)
time_cpu_array = np.array(time_cpu_lst)
time_omp_array = np.array(time_omp_lst)
time_cuda_array = np.array(time_cuda_lst)
n_samples_validation_array = np.array(n_samples_validation_lst)
n_basis_array = np.array(n_basis_lst)

np.save('time_python_array_create_gpc_matrix_dim_' + str(dimensions), time_python_array)
np.save('time_cpu_array_create_gpc_matrix_dim_' + str(dimensions), time_cpu_array)
np.save('time_omp_array_create_gpc_matrix_dim_' + str(dimensions), time_omp_array)
np.save('time_cuda_array_create_gpc_matrix_dim_' + str(dimensions), time_cuda_array)
np.save('n_samples_validation_array_create_gpc_matrix_dim_' + str(dimensions), n_samples_validation_array)
np.save('n_basis_array_create_gpc_matrix_dim_' + str(dimensions), n_basis_array)
