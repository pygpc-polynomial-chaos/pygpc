from _functools import partial
import multiprocessing
import multiprocessing.pool
import Worker
import copy
import numpy as np


class Computation:
    """
    Computation class to run the model
    """
    def __init__(self, n_cpu):
        """
        Constructor; Initializes Computation class
        """
        # Setting up parallelization (setup thread pool)
        n_cpu_available = multiprocessing.cpu_count()
        self.n_cpu = min(n_cpu, n_cpu_available)

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

        self.i_grid = 0

    def run(self, gpc, coords):
        """
        Runs model evaluations for parameter combinations specified in coords array

        Parameters
        ----------
        gpc: GPC object instance
            Initialized gPC object containing the Problem and Model
        coords: ndarray of float [n_sims, n_dim]
            Set of n_sims parameter combinations to run the model with.

        Returns
        -------
        res: ndarray of float [n_sims x n_out]
            n_sims simulation results of the n_out output quantities of the model under investigation.
        """
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
        for random_var_instances in grid_new:
            parameters = copy.deepcopy(gpc.problem.parameters)
            # setup context (let the process know which iteration, interaction order etc.)
            context = {
                'global_task_ctr': self.global_task_counter,
                'lock': self.global_lock,
                'seq_number': seq_num,
                'order': gpc.order_max,
                'i_grid': self.i_grid,
                'max_grid': n_grid_new,
                'interaction_order_current': gpc.interaction_order_current,
                'fn_results': gpc.fn_results
            }

            # replace RandomParameters with grid points
            for i in range(0, len(random_var_instances)):
                parameters[gpc.problem.random_vars[i]] = random_var_instances[i]

            # append new worker which will evaluate the model with particular parameters from grid
            worker_objs.append(gpc.problem.model(parameters, context))

            self.i_grid += 1
            seq_num += 1

        # The map-function deals with chunking the data
        res_new_list = self.process_pool.map(Worker.run, worker_objs)

        # Initialize the result array with the correct size and set the elements according to their order
        # (the first element in 'res' might not necessarily be the result of the first Process/i_grid)
        res = [None] * n_grid_new
        for result in res_new_list:
            res[result[0]] = result[1]

        res = np.array(res)

        return res

    def close(self):
        """ Closes the pool """
        self.process_pool.close()
        self.process_pool.join()
