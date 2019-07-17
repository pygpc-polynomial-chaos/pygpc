# -*- coding: utf-8 -*-
import multiprocessing
import multiprocessing.pool
import subprocess
from pygpc import Worker
import time
import copy
import numpy as np
import dispy
import os
import re
from collections import OrderedDict
from .RandomParameter import *
from .io import iprint


def Computation(n_cpu, matlab_model=False):
    """
    Helper function to initialize the Computation class.
    n_cpu = 0 : use this if the model is capable of to evaluate several parameterizations in parallel
    n_cpu = 1 : the model is called in serial for every paramerization.
    n_cpu > 1 : A multiprocessing.Pool will be opened and n_cpu parameterizations are calculated in parallel

    Parameters
    ----------
    n_cpu : int
        Number of CPU cores to use
    matlab_model : boolean, optional, default: False
        Use a Matlab model

    Returns
    -------
    obj : object instance of Computation class
        Object instance of Computation class
    """
    if n_cpu == 0:
        return ComputationFuncPar(n_cpu, matlab_model=matlab_model)
    else:
        return ComputationPoolMap(n_cpu, matlab_model=matlab_model)


class ComputationPoolMap:
    """
    Computation sub-class to run the model using a processing pool for parallelization
    """

    def __init__(self, n_cpu, matlab_model=False):
        """
        Constructor; Initializes ComputationPoolMap class
        """
        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)

        self.i_grid = 0

        # Use a process queue to assign persistent, unique IDs to the processes in the pool
        self.process_manager = multiprocessing.Manager()
        self.process_queue = self.process_manager.Queue()
        self.process_pool = multiprocessing.Pool(self.n_cpu, Worker.init, (self.process_queue,))

        # Global counter used by all threads to keep track of the progress
        self.global_task_counter = self.process_manager.Value('i', 0)

        for i in range(0, n_cpu):
            self.process_queue.put(i)

        # Necessary to synchronize read/write access to serialized results
        self.global_lock = self.process_manager.RLock()

        self.matlab_engine = None

        # start matlab engine
        if matlab_model:
            import matlab.engine
            iprint("Starting Matlab engine ...", tab=0, verbose=False)
            self.matlab_engine = matlab.engine.start_matlab()

    def run(self, model, problem, coords, coords_norm=None, i_iter=None, i_subiter=None, fn_results=None,
            print_func_time=False):
        """
        Runs model evaluations for parameter combinations specified in coords array

        Parameters
        ----------
        model: Model object
            Model object instance of model to investigate (derived from AbstractModel class, implemented by user)
        problem: Problem class instance
            GPC Problem under investigation, includes the parameters of the model (constant and random)
        coords: ndarray of float [n_sims, n_dim]
            Set of n_sims parameter combinations to run the model with (only the random parameters!).
        coords_norm: ndarray of float [n_sims, n_dim]
            Set of n_sims parameter combinations to run the model with (normalized coordinates [-1, 1].
        i_iter: int
            Index of main-iteration
        i_subiter: int
            Index of sub-iteration
        fn_results : string, optional, default=None
            If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
        print_func_time : bool
            Print time of single function evaluation

        Returns
        -------
        res: ndarray of float [n_sims x n_out]
            n_sims simulation results of the n_out output quantities of the model under investigation.
        """
        if i_iter is None:
            i_iter = "N/A"

        if i_subiter is None:
            i_subiter = "N/A"

        # read new grid points and convert to list for multiprocessing
        grid_new = coords.tolist()

        n_grid_new = len(grid_new)

        # create worker objects that will evaluate the function
        worker_objs = []
        self.global_task_counter.value = 0  # since we re-use the  global counter, we need to reset it first
        seq_num = 0

        # assign the instances of the random_vars to the respective
        # replace random vars of the Problem with single instances
        # determined by the PyGPC framework:
        # assign the instances of the random_vars to the respective
        # entries of the dictionary
        # -> As a result we have the same keys in the dictionary but
        #    no RandomParameters anymore but a sample from the defined PDF.

        for j, random_var_instances in enumerate(grid_new):

            if coords_norm is None:
                c_norm = None
            else:
                c_norm = coords_norm[j, :][np.newaxis, :]

            # setup context (let the process know which iteration, interaction order etc.)
            context = {
                'global_task_counter': self.global_task_counter,
                'lock': self.global_lock,
                'seq_number': seq_num,
                'i_grid': self.i_grid,
                'max_grid': n_grid_new,
                'i_iter': i_iter,
                'i_subiter': i_subiter,
                'fn_results': fn_results,
                'coords': np.array(random_var_instances)[np.newaxis, :],
                'coords_norm': c_norm,
                'print_func_time': print_func_time
            }

            parameters = OrderedDict()
            for key in problem.parameters:
                parameters[key] = problem.parameters[key]

            # replace RandomParameters with grid points
            for i in range(0, len(random_var_instances)):
                if type(random_var_instances[i]) is not np.array:
                    random_var_instances[i] = np.array([random_var_instances[i]])
                parameters[list(problem.parameters_random.keys())[i]] = random_var_instances[i]

            # append new worker which will evaluate the model with particular parameters from grid
            worker_objs.append(model.set_parameters(p=parameters, context=context))

            self.i_grid += 1
            seq_num += 1

        # start model evaluations
        if self.n_cpu == 1:
            res_new_list = []

            for i in range(len(worker_objs)):
                res_new_list.append(Worker.run(obj=worker_objs[i], matlab_engine=self.matlab_engine))

        else:
            # The map-function deals with chunking the data
            res_new_list = self.process_pool.map(Worker.run, worker_objs, self.matlab_engine)

        # Initialize the result array with the correct size and set the elements according to their order
        # (the first element in 'res' might not necessarily be the result of the first Process/i_grid)
        res = [None] * n_grid_new
        for result in res_new_list:
            res[result[0]] = result[1]

        res = np.vstack(res)

        return res

    def close(self):
        """ Closes the pool """
        self.process_pool.close()
        self.process_pool.join()


class ComputationFuncPar:
    """
    Computation sub-class to run the model using a the models internal parallelization
    """

    def __init__(self, n_cpu, matlab_model):
        """
        Constructor; Initializes ComputationPoolMap class
        """
        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)

        self.i_grid = 0

        # Global counter used by all threads to keep track of the progress
        self.global_task_counter = 0

        self.matlab_engine = None

        # start matlab engine
        if matlab_model:
            import matlab.engine
            iprint("Starting Matlab engine ...", tab=0, verbose=True)
            self.matlab_engine = matlab.engine.start_matlab()

    def run(self, model, problem, coords, coords_norm=None, i_iter=None, i_subiter=None, fn_results=None,
            print_func_time=False):
        """
        Runs model evaluations for parameter combinations specified in coords array

        Parameters
        ----------
        model: Model object
            Model object instance of model to investigate (derived from AbstractModel class, implemented by user)
        problem: Problem class instance
            GPC Problem under investigation, includes the parameters of the model (constant and random)
        coords: ndarray of float [n_sims, n_dim]
            Set of n_sims parameter combinations to run the model with (only the random parameters!).
        coords_norm: ndarray of float [n_sims, n_dim]
            Set of n_sims parameter combinations to run the model with (normalized coordinates [-1, 1].
        i_iter: int
            Index of main-iteration
        i_subiter: int
            Index of sub-iteration
        fn_results : string, optional, default=None
            If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
        print_func_time : bool
            Print time of single function evaluation

        Returns
        -------
        res: ndarray of float [n_sims x n_out]
            n_sims simulation results of the n_out output quantities of the model under investigation.
        """
        if i_iter is None:
            i_iter = "N/A"

        if i_subiter is None:
            i_subiter = "N/A"

        n_grid = coords.shape[0]

        # i_grid indices is now a range [min_idx, max_idx]
        self.i_grid = [np.max(self.i_grid), np.max(self.i_grid) + n_grid]

        # assign the instances of the random_vars to the respective
        # replace random vars of the Problem with single instances
        # determined by the PyGPC framework:
        # assign the instances of the random_vars to the respective
        # entries of the dictionary
        # -> As a result we have the same keys in the dictionary but
        #    no RandomParameters anymore but a sample from the defined PDF.

        if coords_norm is None:
            c_norm = None
        else:
            c_norm = coords_norm

        # setup context (let the process know which iteration, interaction order etc.)
        context = {
            'global_task_counter': self.global_task_counter,
            'lock': None,
            'seq_number': None,
            'i_grid': self.i_grid,
            'max_grid': n_grid,
            'i_iter': i_iter,
            'i_subiter': i_subiter,
            'fn_results': fn_results,
            'coords': coords,
            'coords_norm': c_norm,
            'print_func_time': print_func_time
        }

        parameters = OrderedDict()
        i_random_parameter = 0

        for key in problem.parameters:

            if isinstance(problem.parameters[key], RandomParameter):
                # replace RandomParameters with grid points
                parameters[key] = coords[:, i_random_parameter]
                i_random_parameter += 1

            else:
                # copy constant parameters n_grid times
                if type(problem.parameters[key]) == float or problem.parameters[key].size == 1:
                    parameters[key] = problem.parameters[key] * np.ones(n_grid)
                else:
                    if str(type(problem.parameters[key])) == "<class 'matlab.engine.matlabengine.MatlabEngine'>":
                        parameters[key] = problem.parameters[key]
                    else:
                        parameters[key] = np.tile(problem.parameters[key], (n_grid, 1))

        # generate worker, which will evaluate the model (here only one for all grid points in coords)
        worker_objs = model.set_parameters(p=parameters, context=context)

        # start model evaluations
        res = Worker.run(obj=worker_objs, matlab_engine=self.matlab_engine)

        res = np.array(res[1])

        return res

    def close(self):
        """ Closes the pool """
        pass


def compute_cluster(algorithms, nodes, start_scheduler=True):
    """
    Computes Algorithm instances on compute cluster composed of nodes. The first node is also the dispy-scheduler.
    Afterwards, the dispy-nodes are started on every node. On every node, screen sessions are started with the names
    "scheduler" and "node", where the scheduler and the nodes are residing, respectively.
    They can be accessed by "screen -rD scheduler" or "screen -rD node" when connected via ssh to the machines.

    Parameters
    ----------
    algorithms : list of Algorithm instances
        Algorithm instances initialized with different gPC problems and/or models
    nodes : str or list of str
        Node names
    start_scheduler : bool
        Starts a scheduler on the first machine in the nodes list or not. Set this to False if a scheduler is already
        running somewhere on the cluster.
    """

    def _algorithm_run(f):
        f.run

    dispy.MsgTimeout = 90

    for n in nodes:
        # screen/dispy output will be send to devnull, to keep the terminal window clean
        with open(os.devnull, 'w') as f:

            # get PIDs for old scheduler and node screens and kill them
            regexp_pid = "\t(\d*)."  # after \tab, get digits until '.'

            for name in ["scheduler", "node"]:
                # get screen -list output for correct screen, which also has the pid
                stdout, stderr = subprocess.Popen(['ssh', n, 'screen -list | grep {}'.format(name)],
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE).communicate()
                subprocess.Popen(['ssh', n, 'screen', "-wipe"]).communicate()
                try:
                    pid = re.search(regexp_pid, stdout).group(0)[:-1]  # remove last char (.)
                    subprocess.Popen(['ssh', n, 'kill', pid]).communicate()

                except AttributeError:
                    # no 'scheduler' or 'node' screen session found on host
                    pass

            # start scheduler on first node
            if start_scheduler:
                print("Starting dispy scheduler on " + n)

                # subprocess.Popen("ssh -tt " + n + " screen -R scheduler -d -m python "
                #                  + os.path.join(dispy.__path__[0], "dispyscheduler.py &"), shell=True)

                # ssh -tt: pseudo terminal allocation
                #
                # screen
                #        -R scheduler: Reconnect or create session with name scheduler
                #        -d detach (is it needed?)
                #        -m "ignore $STY variable, do create a new screen session" ??
                #
                # subprocess
                #        -shell: False. If True, opens new shell and does not return
                #                If true, do not use [] argument passing style.
                #        -stdout: devnull. Pipe leads to flooded terminal.
                #
                # "export", "TERM=screen", "&&",
                #
                subprocess.Popen(["ssh", "-tt", n,
                                  "screen", "-dmS", "scheduler",
                                  "python " + os.path.join(dispy.__path__[0], "dispyscheduler.py")],
                                  shell=False, stdout=f)
                time.sleep(5)

            print("Starting dispy node on " + n)
            subprocess.Popen(["ssh", "-tt", n,
                              "screen", "-dmS", "node",
                              "python " + os.path.join(dispy.__path__[0], "dispynode.py --clean")],
                              shell=False, stdout=f)
            time.sleep(5)

    cluster = dispy.SharedJobCluster(_algorithm_run, scheduler_node=nodes[0], reentrant=True, port=0)

    time.sleep(5)

    # build job list and start computations
    jobs = []
    for a in algorithms:
        job = cluster.submit(a)
        job.id = a
        jobs.append(job)

    # wait until cluster finished the computations
    cluster.wait()
