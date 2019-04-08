from .Test import *
from .Grid import *
from .misc import *
from .Algorithm import *
from .validation import *
from .postprocessing import *
import os
import copy
from collections import OrderedDict
import multiprocessing
import multiprocessing.pool
from _functools import partial


def run_test(algorithm):
    print("Running: Algorithm: {}   -    Problem: {}".format(type(algorithm).__name__,
                                                             os.path.split(algorithm.options["fn_results"])[1]))
    gpc, coeffs, results = algorithm.run()

    # Post-process gPC
    get_sensitivities_hdf5(fn_gpc=algorithm.options["fn_results"],
                           output_idx=None,
                           calc_sobol=True,
                           calc_global_sens=True,
                           calc_pdf=True)

    # Validate gPC vs original model function (Monte Carlo)
    nrmsd = validate_gpc_mc(gpc=gpc,
                            coeffs=coeffs,
                            n_samples=int(1e4),
                            output_idx=0,
                            n_cpu=algorithm.options["n_cpu"],
                            fn_out=algorithm.options["fn_results"] + '_validation_mc')
    print("\t -> NRMSD = {}".format(nrmsd))


class TestBench(object):
    """
    TestBench for gPC algorithms
    """

    def __init__(self, algorithm, problem, options, n_cpu):
        """
        Initializes the TestBench class object instance

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        problem : Dict() or OrderedDict() of pygpc.Problem instances
            Problem instances to test
        options : Dict or OrderedDict()
            Algorithm options
        n_cpu : int
            Number of threads to run tests in parallel
        """
        self.algorithm = OrderedDict()
        self.fn_results = copy.deepcopy(options["fn_results"])

        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)
        self.pool = multiprocessing.Pool(n_cpu)
        self.run_test_partial = partial(run_test)

        for key in problem:
            if algorithm == Static:
                options["order"] = [options["order"][0] for _ in range(problem[key].dim)]
                n_coeffs = get_num_coeffs_sparse(order_dim_max=options["order"],
                                                 order_glob_max=options["order_max"],
                                                 order_inter_max=options["interaction_order"],
                                                 order_glob_max_norm=options["order_max_norm"],
                                                 dim=problem[key].dim)

                grid = RandomGrid(parameters_random=problem[key].parameters_random,
                                  options={"n_grid": options["matrix_ratio"] * n_coeffs, "seed": options["seed"]})

                options["fn_results"] = os.path.join(self.fn_results, key)

                self.algorithm[key] = algorithm(problem=problem[key],
                                                options=copy.deepcopy(options),
                                                grid=copy.deepcopy(grid))

            else:
                self.algorithm[key] = algorithm(problem=problem[key],
                                                options=copy.deepcopy(options))

    def run(self):
        """
        Run algorithms with test problems and save results
        """

        algorithm_list = [self.algorithm[key] for key in self.algorithm.keys()]

        self.pool.map(self.run_test_partial, algorithm_list)
        self.pool.close()
        self.pool.join()


class TestBenchContinuous(TestBench):
    """
    TestBenchContinuous
    """
    def __init__(self, algorithm, options, n_cpu=1):
        """
        Initializes TestBenchContinuous class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        # set up test problems
        problem = OrderedDict()
        problem["Peaks"] = Peaks().problem
        problem["Franke"] = Franke().problem
        problem["Lim2002"] = Lim2002().problem
        problem["Ishigami2D"] = Ishigami(dim=2).problem
        problem["Ishigami3D"] = Ishigami(dim=3).problem

        super(TestBenchContinuous, self).__init__(algorithm, problem, options, n_cpu)


class TestBenchContinuousND(TestBench):
    """
    TestBenchContinuousND
    """
    def __init__(self, algorithm, options, dims, n_cpu=1):
        """
        Initializes TestBenchContinuousND class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        dims : list of int
            Number of dimensions
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        # set up test problems
        problem = OrderedDict()

        for d in dims:
            problem["ManufactureDecay{}D".format(d)] = ManufactureDecay(dim=d).problem
            problem["GenzContinuous{}D".format(d)] = GenzContinuous(dim=d).problem
            problem["GenzCornerPeak{}D".format(d)] = GenzCornerPeak(dim=d).problem
            problem["GenzGaussianPeak{}D".format(d)] = GenzGaussianPeak(dim=d).problem
            problem["GenzOscillatory{}D".format(d)] = GenzOscillatory(dim=d).problem
            problem["GenzProductPeak{}D".format(d)] = GenzProductPeak(dim=d).problem
            problem["Ridge{}D".format(d)] = Ridge(dim=d).problem
            problem["SphereFun{}D".format(d)] = SphereFun(dim=d).problem
            problem["GFunction{}D".format(d)] = GFunction(dim=d).problem

        problem["OakleyOhagan2004"] = OakleyOhagan2004().problem
        problem["Welch1992"] = Welch1992().problem
        # problem["WingWeight"] = WingWeight().problem

        super(TestBenchContinuousND, self).__init__(algorithm, problem, options, n_cpu)


class TestBenchDiscontinuous(TestBench):
    """
    TestBenchDiscontinuous
    """
    def __init__(self, algorithm, options, n_cpu=1):
        """
        Initializes TestBenchDiscontinuous class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        # set up test problems
        problem = OrderedDict()
        problem["HyperbolicTangent"] = HyperbolicTangent().problem
        problem["MovingParticleFrictionForce"] = MovingParticleFrictionForce().problem
        problem["SurfaceCoverageSpecies2D"] = SurfaceCoverageSpecies(dim=2).problem
        problem["SurfaceCoverageSpecies3D"] = SurfaceCoverageSpecies(dim=3).problem

        super(TestBenchDiscontinuous, self).__init__(algorithm, problem, options, n_cpu)


class TestBenchDiscontinuousND(TestBench):
    """
    TestBenchDiscontinuousND
    """
    def __init__(self, algorithm, options, dims, n_cpu=1):
        """
        Initializes TestBenchDiscontinuousND class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        dims : list of int
            Number of dimensions
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        # set up test problems
        problem = OrderedDict()
        for d in dims:
            problem["GenzDiscontinuous{}D".format(d)] = GenzDiscontinuous().problem

        super(TestBenchDiscontinuousND, self).__init__(algorithm, problem, options, n_cpu)
