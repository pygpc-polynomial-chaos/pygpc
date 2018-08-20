#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 This wrapper script encapsulates the OpenFOAM simulation for usage with the PyGPC framework.

 *Inputs*
   - 1st parameter MUST be the list of conductivities this simulation should be conducted with
   - additional parameters are optional

 *Outputs*
   - The simulation resutls for every mesh element
     (order of the results does not matter, but must be consistent across consecutive calls)

@author: Benjamin Kalloch
"""

import numpy as np
import os
import h5py
import time

from abc import ABCMeta, abstractmethod
from .misc import fancy_bar


class AbstractModel:
    """
    Abstract base class for the SimulationWrapper.
    This base class provides basic functions for serialization/deserialization and printing progress.
    It cannot be used directly, but a derived class implementing the "simulate" method must be created.
    """
    __metaclass__ = ABCMeta

    def __init__(self, context):
        """
        Constructor; initialized the SimulationWrapper class with the provided context

        Parameters
        ----------
        context : dictionary
            dictionary that contains information about this worker's context
                - save_res_fn : location of the hdf5 file to serialize the results to
                - i_grid      : current iteration in the subgrid that is processed
                - max_grid    : size of the current subgrid that is processed
                - i_iter      : current main iteration
                - interaction_oder_current : current interaction order
                - lock        : reference to the Lock object that all processes share for synchronization
                - global_ctr  : reference to the Value object that is shared among all processes to keep track
                                of the overall progress
                - seq_number  : sequence number of the task this object represents; necessary to maintain the correct
                                sequence of results
        """
        self.save_res_fn = context['save_res_fn']
        self.i_grid = context['i_grid']
        self.i_iter = context['i_iter']
        self.lock = context['lock']
        self.max_grid = context['max_grid']
        self.interaction_oder_current = context['interaction_order_current']
        self.global_task_counter = context['global_task_ctr']
        self.seq_number = context['seq_number']

    def read_previous_results(self):
        """
        This functions reads previous serialized results from the hard disk (if present).
        When reading from the array containing the serialized results, the current
        grind-index (i_grid) is considered to maintain the order of the results when the
        SimulationModels are executed in parallel.

        :return:
            None :
                if no serialized results could be found
            list :
                containing the read data
        """
        if self.save_res_fn:
            self.lock.acquire()
            try:
                if os.path.exists(self.save_res_fn):
                    try:
                        with h5py.File(self.save_res_fn, 'r') as f:
                            ds = f['res']
                            # read data from file at current grid_position
                            res  = ds[self.i_grid, :]
                    
                            return res

                    except (KeyError, ValueError):
                        return None
            finally:
                self.lock.release()

        return None

    def write_results(self, data):
        """
        This function writes the data to a file on harddisk.
        When writing the data the current grid-index (i_grid) is considered.
        The data are written to the row corresponding i_grid in order to
        maintain the order of the results when the SimulationModels are
        executed in parallel.

        Parameters
        ----------
        data : ndarray, m x n
            The data to write, the results at n mesh points of m simulations
        """
        if self.save_res_fn:
            self.lock.acquire()
            try:
                require_size = self.i_grid + 1
                with h5py.File(self.save_res_fn, 'a') as f:
                    try:
                        ds = f['res']
                        if ds.shape[0] < require_size:   # check if resize is necessary
                            ds.resize(require_size, axis=0)
                        ds[self.i_grid, :] = data[np.newaxis, :]
                    except (KeyError,ValueError):
                        ds = f.create_dataset('res', (require_size,len(data)), maxshape=(None, len(data)))
                        ds[self.i_grid, :] = data[np.newaxis, :]
            finally:
                self.lock.release()

    def increment_ctr(self):
        """
        This functions increments the global counter by 1.
        """
        self.lock.acquire()
        try:
            self.global_task_counter.value += 1
        finally:
            self.lock.release()

    def print_progress(self, func_time=None, read_from_file=False):
        """
        This functions print the progress according to the current context and global_counter.
        """
        self.lock.acquire()
        try:
            if func_time:
                more_text = "Function evaluation took: " + repr(func_time) + "s"
            elif read_from_file:
                more_text = "Read data row from " + self.save_res_fn
            else:
                more_text = None

            fancy_bar("It/Subit: {}/{} Performing simulation".format(self.i_iter,
                                                                     self.interaction_oder_current),
                      self.global_task_counter.value,
                      self.max_grid,
                      more_text)
        finally:
            self.lock.release()

    def get_seq_number(self):
        return self.seq_number

    @abstractmethod
    def simulate(self, process_id):
        """
        This abstract method must be implemented by the subclass.
        It should perform the simulation task depding on the input_values
        provided to the object on instantiation.

        Parameters
        ----------
        _process_id : int
            A unique identifier; no two processes of the pool will run concurrently with the same identifier
        """
        pass
