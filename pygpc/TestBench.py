# -*- coding: utf-8 -*-
import copy
import glob
import h5py
import multiprocessing
import multiprocessing.pool
import os
import pickle
from _functools import partial
from collections import OrderedDict

from .Algorithm import *
from .Test import *
from .misc import *
from .postprocessing import *
from .validation import *
from .Session import *


def run_test(session):
    print("Running: Algorithm: {}   -    Problem: {}".format(type(session).__name__,
                                                             os.path.split(session.fn_results)[1]))
    session, coeffs, results = session.run()

    # Post-process gPC
    # get_sensitivities_hdf5(fn_gpc=algorithm.options["fn_results"],
    #                        output_idx=None,
    #                        calc_sobol=True,
    #                        calc_global_sens=True,
    #                        calc_pdf=True)


class TestBench(object):
    """
    TestBench for gPC algorithms
    """

    def __init__(self, algorithm, problem, options, repetitions=1, n_cpu=1):
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
        repetitions : int (default=1)
            Number of repeated runs
        n_cpu : int (default=1)
            Number of threads to run tests in parallel
        """
        self.session = OrderedDict()
        self.session_keys = []
        self.algorithm = OrderedDict()
        self.algorithm_type = algorithm
        self.problem = problem
        self.fn_results = copy.deepcopy(options["fn_results"])
        self.repetitions = repetitions
        self.problem_keys = problem.keys()

        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)
        self.pool = multiprocessing.Pool(n_cpu)
        self.run_test_partial = partial(run_test)

        for key in self.problem_keys:
            for rep in range(repetitions):

                self.session_keys.append(key + "_" + str(rep).zfill(4))

                if algorithm == Static:
                    options["fn_results"] = os.path.join(self.fn_results,
                                                         key + "_p_{}_".format(options["order"][0]) + str(rep).zfill(4))
                    options["order"] = [options["order"][0] for _ in range(problem[key].dim)]
                    n_coeffs = get_num_coeffs_sparse(order_dim_max=options["order"],
                                                     order_glob_max=options["order_max"],
                                                     order_inter_max=options["interaction_order"],
                                                     order_glob_max_norm=options["order_max_norm"],
                                                     dim=problem[key].dim)

                    if "seed" in options.keys():
                        grid = RandomGrid(parameters_random=problem[key].parameters_random,
                                          options={"n_grid": options["matrix_ratio"] * n_coeffs, "seed": options["seed"]})
                    else:
                        grid = RandomGrid(parameters_random=problem[key].parameters_random,
                                          options={"n_grid": options["matrix_ratio"] * n_coeffs})

                    self.algorithm[self.session_keys[-1]] = algorithm(problem=problem[key],
                                                                      options=copy.deepcopy(options),
                                                                      grid=copy.deepcopy(grid))

                elif algorithm == StaticProjection:
                    options["fn_results"] = os.path.join(self.fn_results,
                                                         key + "_p_{}_".format(options["order"][0]) + str(rep).zfill(4))

                    self.algorithm[self.session_keys[-1]] = algorithm(problem=problem[key],
                                                                      options=copy.deepcopy(options))

                else:
                    options["fn_results"] = os.path.join(self.fn_results, key + "_" + str(rep).zfill(4))
                    self.algorithm[self.session_keys[-1]] = algorithm(problem=problem[key],
                                                                      options=copy.deepcopy(options))

                self.session[self.session_keys[-1]] = Session(algorithm=self.algorithm[self.session_keys[-1]])

    def run(self):
        """
        Run algorithms with test problems and save results
        """

        session_list = [self.session[key] for key in self.session.keys()]

        run_test(session_list[0])
        # self.pool.map(self.run_test_partial, session_list)
        # self.pool.close()
        # self.pool.join()

        print("Merging .hdf5 files ...")
        # merge .hdf5 files of repetitions
        for key in self.problem_keys:

            # merge gpc files
            print(key)
            if isinstance(self.session[key + "_0000"].algorithm, Static) or \
                    isinstance(self.session[key + "_0000"].algorithm, StaticProjection):
                fn_hdf5 = os.path.join(self.fn_results, key + "_p_{}".format(
                    self.session[key + "_0000"].gpc[0].options["order"][0])) + ".hdf5"
            else:
                fn_hdf5 = os.path.join(self.fn_results, key) + ".hdf5"

            with h5py.File(fn_hdf5, 'w') as f:

                for rep in range(self.repetitions):
                    f.create_group(str(rep).zfill(4))

                    with h5py.File(self.session[key + "_" + str(rep).zfill(4)].fn_results + ".hdf5", 'r') as g:
                        for gkey in g.keys():
                            g.copy(gkey, f[str(rep).zfill(4)])

                    # delete individual .hdf5 files
                    os.remove(self.session[key + "_" + str(rep).zfill(4)].fn_results + ".hdf5")

            # # merge validation files
            # with h5py.File(os.path.join(self.fn_results, key) + "_validation_mc.hdf5", 'w') as f:
            #     for rep in range(self.repetitions):
            #         f.create_group(str(rep).zfill(4))
            #
            #         with h5py.File(os.path.splitext(self.algorithm[key + "_" + str(rep).zfill(4)].options["fn_results"])[0] + "_validation_mc.hdf5", 'r') as g:
            #             for gkey in g.keys():
            #                 g.copy(gkey, f[str(rep).zfill(4)])
            #
            #         # delete individual .hdf5 files
            #         os.remove(os.path.splitext(self.algorithm[key + "_" + str(rep).zfill(4)].options["fn_results"])[0] + "_validation_mc.hdf5")

        # delete .pdf files
        for f in glob.glob(os.path.join(self.fn_results, "*.pdf")):
            os.remove(f)

        del self.pool

        # save TestBench object
        print("Saving testbench.pkl object ...")
        with open(os.path.join(self.fn_results, "testbench.pkl"), 'wb') as output:
            pickle.dump(self, output, -1)


class TestBenchContinuous(TestBench):
    """
    TestBenchContinuous
    """
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchContinuous class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        repetitions : int
            Number of repeated runs
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["Peaks"] = Peaks().problem
        problem["Franke"] = Franke().problem
        problem["Lim2002"] = Lim2002().problem
        problem["Ishigami_2D"] = Ishigami(dim=2).problem
        problem["Ishigami_3D"] = Ishigami(dim=3).problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=1e4, n_cpu=options["n_cpu"])
            self.validation = gpc.validation

        super(TestBenchContinuous, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchContinuousND(TestBench):
    """
    TestBenchContinuousND
    """
    def __init__(self, algorithm, options, dims, repetitions, n_cpu=1):
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
        repetitions : int
            Number of repeated runs
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        self.dims = dims
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()

        for d in dims:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            problem["ManufactureDecay_{}D".format(d)] = ManufactureDecay(dim=d).problem
            problem["GenzContinuous_{}D".format(d)] = GenzContinuous(dim=d).problem
            problem["GenzCornerPeak_{}D".format(d)] = GenzCornerPeak(dim=d).problem
            problem["GenzGaussianPeak_{}D".format(d)] = GenzGaussianPeak(dim=d).problem
            problem["GenzOscillatory_{}D".format(d)] = GenzOscillatory(dim=d).problem
            problem["GenzProductPeak_{}D".format(d)] = GenzProductPeak(dim=d).problem
            problem["Ridge_{}D".format(d)] = Ridge(dim=d).problem
            problem["SphereFun_{}D".format(d)] = SphereFun(dim=d).problem
            problem["GFunction_{}D".format(d)] = GFunction(dim=d).problem

        # create validation sets
        for p in problem:
            problem[p].create_validation_set(n_samples=1e4, n_cpu=options["n_cpu"])

        super(TestBenchContinuousND, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchContinuousHD(TestBench):
    """
    TestBenchContinuousHD
    """
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchContinuousND class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        repetitions : int
            Number of repeated runs
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["OakleyOhagan2004"] = OakleyOhagan2004().problem
        problem["Welch1992"] = Welch1992().problem
        # problem["WingWeight"] = WingWeight().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            problem[p].create_validation_set(n_samples=1e4, n_cpu=options["n_cpu"])

        super(TestBenchContinuousHD, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchDiscontinuous(TestBench):
    """
    TestBenchDiscontinuous
    """
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchDiscontinuous class. Setting up pygpc.Problem instances.

        Parameters
        ----------
        algorithm : pygpc.Algorithm Object
            Algorithm to benchmark
        options : Dict or OrderedDict()
            Algorithm options
        repetitions : int
            Number of repeated runs
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["HyperbolicTangent"] = HyperbolicTangent().problem
        problem["MovingParticleFrictionForce"] = MovingParticleFrictionForce().problem
        problem["SurfaceCoverageSpecies_2D"] = SurfaceCoverageSpecies(dim=2).problem
        problem["SurfaceCoverageSpecies_3D"] = SurfaceCoverageSpecies(dim=3).problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            problem[p].create_validation_set(n_samples=1e4, n_cpu=options["n_cpu"])

        super(TestBenchDiscontinuous, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchDiscontinuousND(TestBench):
    """
    TestBenchDiscontinuousND
    """
    def __init__(self, algorithm, options, dims, repetitions, n_cpu=1):
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
        repetitions : int
            Number of repeated runs
        n_cpu : int
            Number of threads to run pygpc.Problems in parallel
        """
        self.dims = dims
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        for d in dims:
            problem["GenzDiscontinuous_{}D".format(d)] = GenzDiscontinuous().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            problem[p].create_validation_set(n_samples=1e4, n_cpu=options["n_cpu"])

        super(TestBenchDiscontinuousND, self).__init__(algorithm, problem, options, repetitions, n_cpu)
