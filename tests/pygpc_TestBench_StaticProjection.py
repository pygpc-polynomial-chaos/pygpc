import pygpc
import copy
import os

folder = "/data/pt_01756/tmp/TestBench"
n_cpu = 16
repetitions = 3
order = [2, 3, 4, 5]  # 2, 4, 6, 8, 10, 12, 14
dims = [2, 3, 4, 5]


####################################
# StaticProjection (Moore-Penrose) #
####################################
algorithm = pygpc.StaticProjection
order_max_norm = 1.

for o in order:
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [o]
    options["order_max"] = o
    options["order_max_norm"] = order_max_norm
    options["interaction_order"] = 3
    options["matrix_ratio"] = 2
    options["n_cpu"] = 0
    options["error_norm"] = "relative"
    options["error_type"] = "nrmsd"
    options["projection_qoi"] = 0
    options["n_grid_gradient"] = 1000
    options["gradient_calculation"] = "standard_forward"

    options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
                                                    options=copy.deepcopy(options),
                                                    repetitions=repetitions,
                                                    n_cpu=n_cpu)
    TestBenchContinuous.run()

    options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
                                                        options=copy.deepcopy(options),
                                                        repetitions=repetitions,
                                                        dims=dims,
                                                        n_cpu=n_cpu)
    TestBenchContinuousND.run()

    if o < 7:
        options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}".format(
            algorithm.__name__, options["solver"], options["order_max_norm"]))
        TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
                                                            options=copy.deepcopy(options),
                                                            repetitions=repetitions,
                                                            n_cpu=n_cpu)
        TestBenchContinuousHD.run()

    options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
                                                          options=copy.deepcopy(options),
                                                          repetitions=repetitions,
                                                          n_cpu=n_cpu)
    TestBenchDiscontinuous.run()

    options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
                                                              options=copy.deepcopy(options),
                                                              repetitions=repetitions,
                                                              dims=dims,
                                                              n_cpu=n_cpu)
    TestBenchDiscontinuousND.run()


##########################
# StaticProjection (OMP) #
##########################
algorithm = pygpc.StaticProjection
order_max_norm = 1.

for o in order:
    options = dict()
    options["method"] = "reg"
    options["solver"] = "OMP"
    options["settings"] = {"sparsity": 0.5}
    options["order"] = [o]
    options["order_max"] = o
    options["order_max_norm"] = order_max_norm
    options["interaction_order"] = 3
    options["matrix_ratio"] = 1
    options["n_cpu"] = 0
    options["error_norm"] = "relative"
    options["error_type"] = "nrmsd"
    options["projection_qoi"] = 0
    options["n_grid_gradient"] = 1000
    options["gradient_calculation"] = "standard_forward"

    options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_OMP_q_{}".format(
        algorithm.__name__, options["order_max_norm"]))
    TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
                                                    options=copy.deepcopy(options),
                                                    repetitions=repetitions,
                                                    n_cpu=n_cpu)
    TestBenchContinuous.run()

    options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
                                                        options=copy.deepcopy(options),
                                                        repetitions=repetitions,
                                                        dims=dims,
                                                        n_cpu=n_cpu)
    TestBenchContinuousND.run()

    if o < 7:
        options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}".format(
            algorithm.__name__, options["solver"], options["order_max_norm"]))
        TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
                                                            options=copy.deepcopy(options),
                                                            repetitions=repetitions,
                                                            n_cpu=n_cpu)
        TestBenchContinuousHD.run()

    options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
                                                          options=copy.deepcopy(options),
                                                          repetitions=repetitions,
                                                          n_cpu=n_cpu)
    TestBenchDiscontinuous.run()

    options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousHD/{}_{}_q_{}".format(
        algorithm.__name__, options["solver"], options["order_max_norm"]))
    TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
                                                              options=copy.deepcopy(options),
                                                              repetitions=repetitions,
                                                              dims=dims,
                                                              n_cpu=n_cpu)
    TestBenchDiscontinuousND.run()
