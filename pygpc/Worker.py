# -*- coding: utf-8 -*-
import time
import numpy as np
from .misc import list2dict


def init(queue):
    """
    This is a wrapper script to be called by the 'multiprocessing.map' function
    to calculate the model functions in parallel.

    This function will be called upon initialization of the process.
    It sets a global variable denoting the ID of this process that can
    be read by any function of this process

    Parameters
    ----------
    queue : multiprocessing.Queue
             the queue object that manages the unique IDs of the process pool
    """
    global process_id
    process_id = queue.get()


def run(obj, matlab_engine=None):
    """
    This is the main worker function of the process.
    Methods of the provided object will be called here.

    Parameters
    ----------
    obj : any callable object
           The object that
                a) handles the simulation work
                b) reading previous results
                c) writing the calculated result fields
                d) printing global process
    matlab_engine : Matlab engine object, optional, default: None
        Matlab engine object to run Matlab functions
    """
    global process_id

    if 'process_id' not in globals():
        process_id = 0

    if process_id is None:
        process_id = 0

    res = obj.read_previous_results(obj.coords)

    start_time = 0
    end_time = 0
    skip_sim = True

    # skip if there was no data row for that i_grid or if it was prematurely inserted (= all zero)
    if res is None or not np.any(res):
        start_time = time.time()
        out = obj.simulate(process_id, matlab_engine)

        # dictionary containing the results, the coords and (optionally) the additional data
        data_dict = dict()
        data_dict["grid/coords"] = obj.coords
        data_dict["grid/coords_norm"] = obj.coords_norm
        n_sim = obj.coords.shape[0]

        if type(out) is tuple:
            # results (nparray)
            res = out[0]

            # additional data (dict)
            if len(out) == 2:
                # in case of function parallelization transform list of dict to dict containing the lists
                if type(out[1]) is list:
                    additional_data = list2dict(out[1])
                else:
                    additional_data = out[1]

                for o in additional_data:
                    # make entries of additional data to list of list [n_grid][n_data[o]]

                    # make single entries to list
                    if type(additional_data[o]) is not list:
                        additional_data[o] = [[additional_data[o]]]

                    if n_sim == 1:
                        if type(additional_data[o][0]) is not list:
                            additional_data[o] = [additional_data[o]]
                    else:
                        if type(additional_data[o][0]) is not list:
                            additional_data[o] = [[k] for k in additional_data[o]]

                    data_dict[o] = np.array(additional_data[o])

                    if n_sim == 1 and data_dict[o].shape[0] != 1:
                        data_dict[o] = data_dict[o].transpose()
        else:
            # results (nparray), no additional data
            res = out

        # make res to a 2D ndarray [n_sim x n_out]
        if n_sim == 1 and res.ndim == 1:
            res = res[np.newaxis, :]
        elif n_sim == 1 and res.shape[0] > 1 and res.ndim == 2:
            res = res.transpose()

        # add results to data_dict
        data_dict["model_evaluations/results"] = res

        end_time = time.time()

        obj.write_results(data_dict=data_dict)
        skip_sim = False

    obj.increment_ctr()

    # determine function time
    if obj.print_func_time:
        func_time = end_time - start_time
    else:
        func_time = None

    obj.print_progress(func_time=func_time, read_from_file=skip_sim, )

    return obj.get_seq_number(), res
