import pygpc
import copy
import os

folder = "/data/pt_01756/tmp/TestBench"
n_cpu = 32
repetitions = 3
order = [2, 3, 4]  # 2, 4, 6, 8, 10, 12, 14
dims = [2, 3, 4, 5]

##########
# Static #
##########
algorithm = pygpc.Static
order_max_norm = [1.]

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

        # options["fn_results"] = os.path.join(folder, "TestBenchContinuous/{}_{}_q_{}".format(
        #     algorithm.__name__, options["solver"], options["order_max_norm"]))
        # TestBenchContinuous = pygpc.TestBenchContinuous(algorithm=algorithm,
        #                                                 options=copy.deepcopy(options),
        #                                                 repetitions=repetitions,
        #                                                 n_cpu=n_cpu)
        # TestBenchContinuous.run()

        # options["fn_results"] = os.path.join(folder, "TestBenchContinuousND/{}_{}_q_{}".format(
        #     algorithm.__name__, options["solver"], options["order_max_norm"]))
        # TestBenchContinuousND = pygpc.TestBenchContinuousND(algorithm=algorithm,
        #                                                     options=copy.deepcopy(options),
        #                                                     repetitions=repetitions,
        #                                                     dims=dims,
        #                                                     n_cpu=n_cpu)
        # TestBenchContinuousND.run()
        #
        #
        # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuous/{}_{}_q_{}".format(
        #     algorithm.__name__, options["solver"], options["order_max_norm"]))
        # TestBenchDiscontinuous = pygpc.TestBenchDiscontinuous(algorithm=algorithm,
        #                                                       options=copy.deepcopy(options),
        #                                                       repetitions=repetitions,
        #                                                       n_cpu=n_cpu)
        # TestBenchDiscontinuous.run()
        #
        # options["fn_results"] = os.path.join(folder, "TestBenchDiscontinuousND/{}_{}_q_{}".format(
        #     algorithm.__name__, options["solver"], options["order_max_norm"]))
        # TestBenchDiscontinuousND = pygpc.TestBenchDiscontinuousND(algorithm=algorithm,
        #                                                           options=copy.deepcopy(options),
        #                                                           repetitions=repetitions,
        #                                                           dims=dims,
        #                                                           n_cpu=n_cpu)
        # TestBenchDiscontinuousND.run()

        options["fn_results"] = os.path.join(folder, "TestBenchContinuousHD/{}_{}_q_{}".format(
            algorithm.__name__, options["solver"], options["order_max_norm"]))
        TestBenchContinuousHD = pygpc.TestBenchContinuousHD(algorithm=algorithm,
                                                            options=copy.deepcopy(options),
                                                            repetitions=repetitions,
                                                            n_cpu=n_cpu)
        TestBenchContinuousHD.run()
