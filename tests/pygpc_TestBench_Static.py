import pygpc
import copy
import os

folder = "/data/pt_01756/studies/pygpc/"
n_cpu = 18
repetitions = 10
# order = [2, 4, 6, 8, 10, 12, 14]  # 2, 4, 6, 8, 10, 12, 14
# dims = [2, 3, 4, 5]

order = [2, 3, 4, 6, 8, 10, 12]  # 2, 3, 4, 6, 8, 10, 12
dims = [2, 3]

##########
# Static #
##########
algorithm = pygpc.Static
order_max_norm = [1.]
gradient_enhanced = [False]

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
            options["adaptive sampling"] = True
            options["interaction_order"] = 3
            options["matrix_ratio"] = 2
            options["n_cpu"] = 0
            options["error_norm"] = "relative"
            options["error_type"] = "nrmsd"
            options["gradient_enhanced"] = g_e
            options["gradient_calculation"] = "standard_forward"
            options["grid"] = pygpc.Random
            options["grid_options"] = None

            options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_{}_q_{}_ge_{}".format(
                algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
                                                            options=copy.deepcopy(options),
                                                            repetitions=repetitions,
                                                            n_cpu=n_cpu)
            TestBenchContinuous.run()

            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     dims=dims,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousND.run()

            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
            #                                                       options=copy.deepcopy(options),
            #                                                       repetitions=repetitions,
            #                                                       n_cpu=n_cpu)
            # TestBenchDiscontinuous.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
            #                                                           options=copy.deepcopy(options),
            #                                                           repetitions=repetitions,
            #                                                           dims=dims,
            #                                                           n_cpu=n_cpu)
            # TestBenchDiscontinuousND.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousHD.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchNoisy/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchNoisy = pygpc.TestBenchNoisy(algorithm=algorithm,
            #                                       options=copy.deepcopy(options),
            #                                       repetitions=repetitions,
            #                                       n_cpu=n_cpu)
            # TestBenchNoisy.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchNoisyND/{}_{}_{}_q_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["grid"].__name__, options["order_max_norm"], int(g_e)))
            # TestBenchNoisyND = pygpc.TestBenchNoisyND(algorithm=algorithm,
            #                                           options=copy.deepcopy(options),
            #                                           repetitions=repetitions,
            #                                           dims=dims,
            #                                           n_cpu=n_cpu)
            # TestBenchNoisyND.run()
