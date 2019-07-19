import pygpc
import copy
import os

folder = "/data/pt_01756/tmp/TestBench"
n_cpu = 16
repetitions = 3
order = [2, 4, 6, 8, 10, 12, 14]  # 2, 4, 6, 8, 10, 12, 14
dims = [2, 3, 4, 5]


####################################
# StaticProjection (Moore-Penrose) #
####################################
algorithm = pygpc.StaticProjection
order_max_norm = [1.]
gradient_enhanced = [False, True]

for g_e in gradient_enhanced:
    for o in order_max_norm:
        for ord in order:
            options = dict()
            options["method"] = "reg"
            options["solver"] = "Moore-Penrose"
            options["settings"] = None
            options["order"] = [ord]
            options["order_max"] = ord
            options["order_max_norm"] = o
            options["interaction_order"] = 3
            options["matrix_ratio"] = 2
            options["n_cpu"] = 0
            options["error_norm"] = "relative"
            options["error_type"] = "nrmsd"
            options["projection_qoi"] = 0
            options["n_grid_gradient"] = 25
            options["gradient_calculation"] = "standard_forward"
            options["gradient_enhanced"] = g_e

            options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}_ge_{}".format(
                algorithm.__name__, options["solver"], options["order_max_norm"], int(g_e)))
            TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
                                                            options=copy.deepcopy(options),
                                                            repetitions=repetitions,
                                                            n_cpu=n_cpu)
            TestBenchContinuous.run()

            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(g_e)))
            # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     dims=dims,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousND.run()
            #
            # if o < 7:
            #     options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}_ge_{}".format(
            #         algorithm.__name__, options["solver"], options["order_max_norm"], int(g_e)))
            #     TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
            #                                                         options=copy.deepcopy(options),
            #                                                         repetitions=repetitions,
            #                                                         n_cpu=n_cpu)
            #     TestBenchContinuousHD.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(g_e)))
            # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
            #                                                       options=copy.deepcopy(options),
            #                                                       repetitions=repetitions,
            #                                                       n_cpu=n_cpu)
            # TestBenchDiscontinuous.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(g_e)))
            # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
            #                                                           options=copy.deepcopy(options),
            #                                                           repetitions=repetitions,
            #                                                           dims=dims,
            #                                                           n_cpu=n_cpu)
            # TestBenchDiscontinuousND.run()
