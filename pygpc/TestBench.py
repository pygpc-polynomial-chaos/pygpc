import copy
import glob
import h5py
import multiprocessing
import multiprocessing.pool
import os
import pickle
from _functools import partial
from collections import OrderedDict
from .io import write_session_pkl
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
    get_sensitivities_hdf5(fn_gpc=session.fn_results,
                           output_idx=None,
                           calc_sobol=True,
                           calc_global_sens=True,
                           calc_pdf=True)

    # Validate gPC vs original model function (2D-surface)
    if len(list(session.parameters_random.keys())) == 1:
        random_vars = list(session.parameters_random.keys())
        n_grid = [101]
    else:
        random_vars = list(session.parameters_random.keys())[0:2]
        n_grid = [51, 51]

    validate_gpc_plot(session=session,
                      coeffs=coeffs,
                      random_vars=random_vars,
                      n_grid=n_grid,
                      output_idx=0,
                      fn_out=session.fn_results + "_val",
                      n_cpu=session.n_cpu)

    return session


class TestBench(object):
    """
    TestBench for gPC algorithms

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

    def __init__(self, algorithm, problem, options, repetitions=1, n_cpu=1):
        """
        Initializes the TestBench class object instance
        """
        self.session = OrderedDict()
        self.session_keys = []
        self.algorithm = OrderedDict()
        self.algorithm_type = algorithm
        self.problem = problem
        self.fn_results = copy.deepcopy(options["fn_results"])
        self.repetitions = repetitions
        self.problem_keys = list(problem.keys())

        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)
        self.pool = multiprocessing.Pool(n_cpu)
        self.run_test_partial = partial(run_test)

        if "seed" not in list(options.keys()):
            options["seed"] = None

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

                    grid = options["grid"](parameters_random=problem[key].parameters_random,
                                           n_grid=options["matrix_ratio"] * n_coeffs,
                                           seed=options["seed"],
                                           options=options["grid_options"])

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

        session_list = [self.session[key] for key in list(self.session.keys())]

        # for session in session_list:
        #     run_test(session)
        session_list = self.pool.map(self.run_test_partial, session_list)
        self.pool.close()
        self.pool.join()

        # transform session list back to dict
        for i, key in enumerate(self.session_keys):
            self.session[key] = session_list[i]

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
                        for gkey in list(g.keys()):
                            g.copy(gkey, f[str(rep).zfill(4)])

                    # delete individual .hdf5 files
                    os.remove(self.session[key + "_" + str(rep).zfill(4)].fn_results + ".hdf5")

            # merge validation files
            with h5py.File(os.path.join(self.fn_results, key) + "_val.hdf5", 'w') as f:
                for rep in range(self.repetitions):
                    f.create_group(str(rep).zfill(4))

                    with h5py.File(os.path.splitext(self.algorithm[key + "_" +
                                   str(rep).zfill(4)].options["fn_results"])[0] + "_val.hdf5", 'r') as g:
                        for gkey in list(g.keys()):
                            g.copy(gkey, f[str(rep).zfill(4)])

                    # delete individual .hdf5 files
                    os.remove(os.path.splitext(self.algorithm[key +
                              "_" + str(rep).zfill(4)].options["fn_results"])[0] + "_val.hdf5")

        # delete .pdf files
        for f in glob.glob(os.path.join(self.fn_results, "*.pdf")):
            os.remove(f)

        del self.pool

        # save TestBench object
        print("Saving testbench.pkl object ...")
        write_session_pkl(self, os.path.join(self.fn_results, "testbench.pkl"))


class TestBenchContinuous(TestBench):
    """
    TestBenchContinuous

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
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchContinuous class. Setting up pygpc.Problem instances.
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["BohachevskyFunction1"] = BohachevskyFunction1().problem
        # problem["BoothFunction"] = BoothFunction().problem
        # problem["BukinFunctionNumber6"] = BukinFunctionNumber6().problem
        # problem["Franke"] = Franke().problem
        # problem["Ishigami_2D"] = Ishigami(dim=2).problem
        # problem["Ishigami_3D"] = Ishigami(dim=3).problem
        # problem["Lim2002"] = Lim2002().problem
        # problem["MatyasFunction"] = MatyasFunction().problem
        problem["McCormickFunction"] = McCormickFunction().problem
        # problem["Peaks"] = Peaks().problem
        # problem["SixHumpCamelFunction"] = SixHumpCamelFunction().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchContinuous, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchContinuousND(TestBench):
    """
    TestBenchContinuousND

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
    def __init__(self, algorithm, options, dims, repetitions, n_cpu=1):
        """
        Initializes TestBenchContinuousND class. Setting up pygpc.Problem instances.
        """
        self.dims = dims
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()

        for d in dims:
            problem["DixonPriceFunction{}D".format(d)] = DixonPriceFunction(dim=d).problem
            problem["GenzContinuous_{}D".format(d)] = GenzContinuous(dim=d).problem
            problem["GenzCornerPeak_{}D".format(d)] = GenzCornerPeak(dim=d).problem
            problem["GenzGaussianPeak_{}D".format(d)] = GenzGaussianPeak(dim=d).problem
            problem["GenzOscillatory_{}D".format(d)] = GenzOscillatory(dim=d).problem
            problem["GenzProductPeak_{}D".format(d)] = GenzProductPeak(dim=d).problem
            problem["GFunction_{}D".format(d)] = GFunction(dim=d).problem
            problem["ManufactureDecay_{}D".format(d)] = ManufactureDecay(dim=d).problem
            problem["PermFunction{}D".format(d)] = PermFunction(dim=d).problem
            problem["Ridge_{}D".format(d)] = Ridge(dim=d).problem
            problem["RosenbrockFunction{}D".format(d)] = RosenbrockFunction(dim=d).problem
            problem["RotatedHyperEllipsoid{}D".format(d)] = RotatedHyperEllipsoid(dim=d).problem
            problem["SphereFunction{}D".format(d)] = SphereFunction(dim=d).problem
            problem["SumOfDifferentPowersFunction{}D".format(d)] = SumOfDifferentPowersFunction(dim=d).problem
            problem["ZakharovFunction{}D".format(d)] = ZakharovFunction(dim=d).problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchContinuousND, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchContinuousHD(TestBench):
    """
    TestBenchContinuousHD

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
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchContinuousND class. Setting up pygpc.Problem instances.
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
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchContinuousHD, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchDiscontinuous(TestBench):
    """
    TestBenchDiscontinuous

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
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchDiscontinuous class. Setting up pygpc.Problem instances.
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["Cluster3Simple"] = Cluster3Simple().problem
        problem["DeJongFunctionFive"] = DeJongFunctionFive().problem
        problem["HyperbolicTangent"] = HyperbolicTangent().problem
        problem["MovingParticleFrictionForce"] = MovingParticleFrictionForce().problem
        problem["SurfaceCoverageSpecies_2D"] = SurfaceCoverageSpecies(dim=2).problem
        problem["SurfaceCoverageSpecies_3D"] = SurfaceCoverageSpecies(dim=3).problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchDiscontinuous, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchDiscontinuousND(TestBench):
    """
    TestBenchDiscontinuousND

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

    def __init__(self, algorithm, options, dims, repetitions, n_cpu=1):
        """
        Initializes TestBenchDiscontinuousND class. Setting up pygpc.Problem instances.
        """
        self.dims = dims
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        for d in dims:
            problem["GenzDiscontinuous_{}D".format(d)] = GenzDiscontinuous().problem
            problem["MichalewiczFunction{}D".format(d)] = MichalewiczFunction().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchDiscontinuousND, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchNoisy(TestBench):
    """
    TestBenchNoisy

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
    def __init__(self, algorithm, options, repetitions, n_cpu=1):
        """
        Initializes TestBenchNoisy class. Setting up pygpc.Problem instances.
        """
        self.dims = []
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        problem["CrossinTrayFunction"] = CrossinTrayFunction().problem
        problem["DropWaveFunction"] = DropWaveFunction().problem
        problem["GramacyLeeFunction"] = GramacyLeeFunction().problem
        problem["SchafferFunction4"] = SchafferFunction4().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchNoisy, self).__init__(algorithm, problem, options, repetitions, n_cpu)


class TestBenchNoisyND(TestBench):
    """
    TestBenchNoisyND

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
    def __init__(self, algorithm, options, dims, repetitions, n_cpu=1):
        """
        Initializes TestBenchNoisyND class. Setting up pygpc.Problem instances.
        """
        self.dims = dims
        self.validation = OrderedDict()

        # set up test problems
        problem = OrderedDict()
        for d in dims:
            problem["Ackley{}D".format(d)] = Ackley().problem

        # create validation sets
        for p in problem:
            gpc = GPC(problem=problem[p], options=None, validation=None)
            gpc.create_validation_set(n_samples=int(1e4), n_cpu=options["n_cpu"])
            self.validation[p] = gpc.validation

        super(TestBenchNoisyND, self).__init__(algorithm, problem, options, repetitions, n_cpu)
