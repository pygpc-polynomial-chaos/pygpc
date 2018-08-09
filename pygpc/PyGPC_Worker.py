#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 This is a wrapper script to be calles by the 'multiprocecsing.map' function in 'run_reg_adaptive2_parallel'

@author: Benjamin Kalloch
"""
import time

def init( _queue ):
    """
    This function will be called upon inititalization of the process.
    It sets a global variable denoting the ID of this process that can
    be read by any function of this process

    Parameters
    ----------
    _queue : multiprocessing.Queue
             the queue object that manages the unique IDs of the process pool
    """
    global process_id
    process_id = _queue.get()

def run( _obj ):
    """
    This is the main worker function of the process.
    Methods of the provided object will be called here.

    Parameters
    ----------
    _obj : any callable object
           The object that
                a) handles the simulation work
                b) reading previous results
                c) writing the calculated result fields
                d) printing global process
    """
    global process_id

    if process_id is None:
        process_id = 0

    # do stuff with _obj here, e.g.
    res = _obj.read_previous_results()

    start_time = 0
    end_time   = 0
    skip_sim   = True

    if res is None:
        start_time = time.time()
        res = _obj.simulate( process_id )
        end_time = time.time()
        _obj.write_new_results(res)
        skip_sim = False

    _obj.increment_ctr()
    _obj.print_progress(_func_time=end_time - start_time, _read_from_file=skip_sim);

    return res
