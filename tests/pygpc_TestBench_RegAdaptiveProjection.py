import pygpc
import copy
import os

folder = "/data/pt_01756/tmp/TestBench"
n_cpu = 32
repetitions = 3
dims = [2, 3, 4, 5]

# #########################################
# # RegAdaptiveProjection (Moore-Penrose) #
# #########################################
# algorithm = pygpc.RegAdaptiveProjection
# order_max_norm = [1.0]
# adaptive_sampling = [False]
# gradient_enhanced = [False, True]
#
# for g_e in gradient_enhanced:
#     for a_s in adaptive_sampling:
#         for o in order_max_norm:
#             options = dict()
#             options["order_start"] = 1
#             options["order_end"] = 15
#             options["interaction_order"] = 3
#             options["solver"] = "Moore-Penrose"
#             options["order_max_norm"] = o
#             options["settings"] = None
#             options["matrix_ratio"] = 2
#             options["n_cpu"] = 0
#             options["eps"] = 1e-3
#             options["error_norm"] = "relative"
#             options["error_type"] = "nrmsd"
#             options["projection_qoi"] = 0
#             options["n_grid_gradient"] = 5
#             options["gradient_calculation"] = "standard_forward"
#             options["adaptive_sampling"] = a_s
#             options["gradient_enhanced"] = g_e
#
#             options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}_as_{}_ge_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], int(a_s), int(g_e)))
#             TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
#                                                             options=options,
#                                                             repetitions=repetitions,
#                                                             n_cpu=n_cpu)
#             TestBenchContinuous.run()

            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(a_s), int(g_e)))
            # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     dims=dims,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousND.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(a_s), int(g_e)))
            # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
            #                                                       options=copy.deepcopy(options),
            #                                                       repetitions=repetitions,
            #                                                       n_cpu=n_cpu)
            # TestBenchDiscontinuous.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(a_s), int(g_e)))
            # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
            #                                                           options=copy.deepcopy(options),
            #                                                           repetitions=repetitions,
            #                                                           dims=dims,
            #                                                           n_cpu=n_cpu)
            # TestBenchDiscontinuousND.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], int(a_s), int(g_e)))
            # TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousHD.run()


# ###############################
# # RegAdaptiveProjection (OMP) #
# ###############################
# algorithm = pygpc.RegAdaptiveProjection
# order_max_norm = [0.5, 1.0]
# sparsity = [0.25, 0.5]
# adaptive_sampling = [False, True]
#
# for a_s in adaptive_sampling:
#     for o in order_max_norm:
#         for s in sparsity:
#             options = dict()
#             options["order_start"] = 1
#             options["order_end"] = 10
#             options["interaction_order"] = 3
#             options["solver"] = "OMP"
#             options["order_max_norm"] = o
#             options["settings"] = {"sparsity": s}
#             options["matrix_ratio"] = 0.5
#             options["n_cpu"] = 0
#             options["eps"] = 5e-3
#             options["error_norm"] = "relative"
#             options["error_type"] = "nrmsd"
#             options["projection_qoi"] = 0
#             options["n_grid_gradient"] = 25
#             options["gradient_calculation"] = "standard_forward"
#             options["adaptive_sampling"] = a_s
#
#             options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}_s_{}_as_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["sparsity"],
#                 int(a_s)))
#             TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
#                                                             options=options,
#                                                             repetitions=repetitions,
#                                                             n_cpu=n_cpu)
#             TestBenchContinuous.run()
#
#             options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}_s_{}_as_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["sparsity"],
#                 int(a_s)))
#             TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
#                                                                 options=copy.deepcopy(options),
#                                                                 repetitions=repetitions,
#                                                                 dims=dims,
#                                                                 n_cpu=n_cpu)
#             TestBenchContinuousND.run()
#
#             options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}_s_{}_as_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["sparsity"],
#                 int(a_s)))
#             TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
#                                                                   options=copy.deepcopy(options),
#                                                                   repetitions=repetitions,
#                                                                   n_cpu=n_cpu)
#             TestBenchDiscontinuous.run()
#
#             options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}_s_{}_as_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["sparsity"],
#                 int(a_s)))
#             TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
#                                                                       options=copy.deepcopy(options),
#                                                                       repetitions=repetitions,
#                                                                       dims=dims,
#                                                                       n_cpu=n_cpu)
#             TestBenchDiscontinuousND.run()
#
#             options["order_end"] = 5
#             options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}_s_{}_as_{}".format(
#                 algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["sparsity"],
#                 int(a_s)))
#             TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
#                                                                 options=copy.deepcopy(options),
#                                                                 repetitions=repetitions,
#                                                                 n_cpu=n_cpu)
#             TestBenchContinuousHD.run()


#####################################
# RegAdaptiveProjection (LarsLasso) #
#####################################
algorithm = pygpc.RegAdaptiveProjection
order_max_norm = [1.0]
adaptive_sampling = [False, True]
gradient_enhanced = [False, True]

for g_e in gradient_enhanced:
    for a_s in adaptive_sampling:
        for o in order_max_norm:
            options = dict()
            options["order_start"] = 1
            options["order_end"] = 6
            options["interaction_order"] = 3
            options["solver"] = "LarsLasso"
            options["order_max_norm"] = o
            options["settings"] = {"alpha": 1e-5}
            options["matrix_ratio"] = 1
            options["n_cpu"] = 0
            options["eps"] = 1e-3
            options["error_norm"] = "relative"
            options["error_type"] = "nrmsd"
            options["projection_qoi"] = 0
            options["n_grid_gradient"] = 15
            options["gradient_calculation"] = "standard_forward"
            options["adaptive_sampling"] = a_s
            options["gradient_enhanced"] = g_e

            # options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}_a_{}_as_{}_ge_{}_p_norm".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["alpha"],
            #     int(a_s), int(g_e)))
            # TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
            #                                                 options=options,
            #                                                 repetitions=repetitions,
            #                                                 n_cpu=n_cpu)
            # TestBenchContinuous.run()

            # options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}_a_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["alpha"],
            #     int(a_s), int(g_e)))
            # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
            #                                                     options=copy.deepcopy(options),
            #                                                     repetitions=repetitions,
            #                                                     dims=dims,
            #                                                     n_cpu=n_cpu)
            # TestBenchContinuousND.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}_a_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["alpha"],
            #     int(a_s), int(g_e)))
            # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
            #                                                       options=copy.deepcopy(options),
            #                                                       repetitions=repetitions,
            #                                                       n_cpu=n_cpu)
            # TestBenchDiscontinuous.run()
            #
            # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}_a_{}_as_{}_ge_{}".format(
            #     algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["alpha"],
            #     int(a_s), int(g_e)))
            # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
            #                                                           options=copy.deepcopy(options),
            #                                                           repetitions=repetitions,
            #                                                           dims=dims,
            #                                                           n_cpu=n_cpu)
            # TestBenchDiscontinuousND.run()
            #
            options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}_a_{}_as_{}_ge_{}_p_norm".format(
                algorithm.__name__, options["solver"], options["order_max_norm"], options["settings"]["alpha"],
                int(a_s), int(g_e)))
            TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
                                                                options=copy.deepcopy(options),
                                                                repetitions=repetitions,
                                                                n_cpu=n_cpu)
            TestBenchContinuousHD.run()
