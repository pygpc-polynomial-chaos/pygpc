import pygpc
import copy

# ##########
# # Static #
# ##########
# algorithm = pygpc.Static
# order = [2, 4, 6, 8, 10, 12, 14]  #
# order_max_norm = 1.
# n_cpu = 8
# repetitions = 3
#
# for o in order:
#     options = dict()
#     options["method"] = "reg"
#     options["solver"] = "Moore-Penrose"
#     options["settings"] = None
#     options["order"] = [o]
#     options["order_max"] = o
#     options["order_max_norm"] = order_max_norm
#     options["interaction_order"] = 3
#     options["matrix_ratio"] = 2
#     options["n_cpu"] = 0
#     options["error_norm"] = "relative"
#     options["error_type"] = "nrmsd"
#
#     options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuous/{}_MP_q_{}".format(
#             algorithm.__name__, options["order_max_norm"])
#     TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
#                                                     options=copy.deepcopy(options),
#                                                     repetitions=repetitions,
#                                                     n_cpu=n_cpu)
#     TestBenchContinuous.run()

    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuousND/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
    #                                                     options=copy.deepcopy(options),
    #                                                     repetitions=repetitions,
    #                                                     dims=dims,
    #                                                     n_cpu=n_cpu)
    # TestBenchContinuousND.run()
    #
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuous/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
    #                                                       options=copy.deepcopy(options),
    #                                                       repetitions=repetitions,
    #                                                       n_cpu=n_cpu)
    # TestBenchDiscontinuous.run()
    #
    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuousND/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
    #                                                           options=copy.deepcopy(options),
    #                                                           repetitions=repetitions,
    #                                                           dims=dims,
    #                                                           n_cpu=n_cpu)
    # TestBenchDiscontinuousND.run()

# ###############################
# # RegAdaptive (Moore-Penrose) #
# ###############################
# algorithm = pygpc.RegAdaptive
# order_max_norm = [0.5, 1.0]
# n_cpu = 8
# repetitions = 3
#
# for o in order_max_norm:
#     options = dict()
#     options["order_start"] = 1
#     options["order_end"] = 14
#     options["interaction_order"] = 3
#     options["solver"] = "Moore-Penrose"
#     options["order_max_norm"] = o
#     options["settings"] = None
#     options["matrix_ratio"] = 2
#     options["n_cpu"] = 0
#     options["eps"] = 1e-3
#     options["error_norm"] = "relative"
#     options["error_type"] = "nrmsd"
#
#     options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuous/{}_MP_q_{}".format(
#             algorithm.__name__, options["order_max_norm"])
#     TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
#                                                     options=options,
#                                                     repetitions=repetitions,
#                                                     n_cpu=n_cpu)
#     TestBenchContinuous.run()

    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuousND/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
    #                                                     options=copy.deepcopy(options),
    #                                                     repetitions=repetitions,
    #                                                     dims=dims,
    #                                                     n_cpu=n_cpu)
    # TestBenchContinuousND.run()
    #
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuous/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
    #                                                       options=copy.deepcopy(options),
    #                                                       repetitions=repetitions,
    #                                                       n_cpu=n_cpu)
    # TestBenchDiscontinuous.run()
    #
    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuousND/{}_MP_q_{}".format(
    #     algorithm.__name__, options["order_max_norm"])
    # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
    #                                                           options=copy.deepcopy(options),
    #                                                           repetitions=repetitions,
    #                                                           dims=dims,
    #                                                           n_cpu=n_cpu)
    # TestBenchDiscontinuousND.run()


####################
# RegAdaptive (OMP #
####################
algorithm = pygpc.RegAdaptive
order_max_norm = [0.5, 1.0]
n_cpu = 8
repetitions = 5

for o in order_max_norm:
    options = dict()
    options["order_start"] = 1
    options["order_end"] = 14
    options["interaction_order"] = 3
    options["solver"] = "OMP"
    options["order_max_norm"] = o
    options["settings"] = {"sparsity": 0.25}
    options["matrix_ratio"] = 0.5
    options["n_cpu"] = 0
    options["eps"] = 1e-3
    options["error_norm"] = "relative"
    options["error_type"] = "nrmsd"

    options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuous/{}_OMP_q_{}_s_{}".format(
            algorithm.__name__, options["order_max_norm"], options["settings"]["sparsity"])
    TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
                                                    options=options,
                                                    repetitions=repetitions,
                                                    n_cpu=n_cpu)
    TestBenchContinuous.run()

    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchContinuousND/{}_OMP_q_{}_s_{}".format(
    #     algorithm.__name__, options["order_max_norm"], options["settings"]["sparsity"])
    # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
    #                                                     options=copy.deepcopy(options),
    #                                                     repetitions=repetitions,
    #                                                     dims=dims,
    #                                                     n_cpu=n_cpu)
    # TestBenchContinuousND.run()
    #
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuous/{}_OMP_q_{}_s_{}".format(
    #     algorithm.__name__, options["order_max_norm"], options["settings"]["sparsity"])
    # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
    #                                                       options=copy.deepcopy(options),
    #                                                       repetitions=repetitions,
    #                                                       n_cpu=n_cpu)
    # TestBenchDiscontinuous.run()
    #
    # dims = [2, 3]
    # options["fn_results"] = "/NOBACKUP2/tmp/TestBench/TestBenchDiscontinuousND/{}_OMP_q_{}_s_{}".format(
    #     algorithm.__name__, options["order_max_norm"], options["settings"]["sparsity"])
    # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
    #                                                           options=copy.deepcopy(options),
    #                                                           repetitions=repetitions,
    #                                                           dims=dims,
    #                                                           n_cpu=n_cpu)
    # TestBenchDiscontinuousND.run()
