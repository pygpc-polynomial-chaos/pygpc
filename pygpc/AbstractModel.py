#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
 This wrapper script encapsulates the OpenFOAM simulation for usage with the PyGPC framework.

 *Inputs*
   - 1st parameter MUST be the list of conductivities this simulation should be conducted with
   - additional parameters are optional

 *Outputs*
   - The simulation results for every mesh element
     (order of the results does not matter, but must be consistent across consecutive calls)

@author: Benjamin Kalloch
"""

import numpy as np
import os
import h5py

from abc import ABCMeta, abstractmethod
from .misc import display_fancy_bar


class AbstractModel:
    """
    Abstract base class for the SimulationWrapper.
    This base class provides basic functions for serialization/deserialization and printing progress.
    It cannot be used directly, but a derived class implementing the "simulate" method must be created.
    """
    __metaclass__ = ABCMeta

    def __init__(self, p, context):
        """
        Constructor; initialized the SimulationWrapper class with the provided context

        Parameters
        ----------
        p : dictionary
            Dictionary containing the model parameters
        context : dictionary
            dictionary that contains information about this worker's context
                - fn_results  : location of the hdf5 file to serialize the results to
                - i_grid      : current iteration in the sub-grid that is processed
                - max_grid    : size of the current sub-grid that is processed
                - i_iter      : current main-iteration
                - i_subiter   : current sub-iteration
                - lock        : reference to the Lock object that all processes share for synchronization
                - global_ctr  : reference to the Value object that is shared among all processes to keep track
                                of the overall progress
                - seq_number  : sequence number of the task this object represents; necessary to maintain the correct
                                sequence of results
        """

        self.p = p

        if context is not None:
            self.i_subiter = context['i_subiter']
            self.lock = context['lock']
            self.max_grid = context['max_grid']
            self.global_task_counter = context['global_task_ctr']
            self.seq_number = context['seq_number']
            self.fn_results = context['fn_results']
            self.i_grid = context['i_grid']
            self.i_iter = context['i_iter']
            self.coords = context['coords']
            self.coords_norm = context['coords_norm']

    def read_previous_results(self, coords):
        """
        This functions reads previous serialized results from the hard disk (if present).
        When reading from the array containing the serialized results, the current
        grid-index (i_grid) is considered to maintain the order of the results when the
        SimulationModels are executed in parallel.

        Parameters
        ----------
        coords : ndarray [n_sims x dim]
            Grid coordinates the simulations are conducted with

        Returns
        -------
            None :
                if no serialized results could be found or does not fit to grid
            list :
                data at coords
        """
        if self.fn_results:
            self.lock.acquire()
            try:
                if os.path.exists(self.fn_results):

                    # read results and coords
                    try:
                        with h5py.File(self.fn_results, 'r') as f:
                            ds = f['results']
                            res = ds[self.i_grid, :]

                            ds = f['coords']
                            coords_read = ds[self.i_grid, :]

                            if np.isclose(coords_read, coords).all():
                                return res.tolist()
                            else:
                                return None

                    except (KeyError, ValueError):
                        return None
            finally:
                self.lock.release()

        return None

    def write_results(self, data, coords, coords_norm=None):
        """
        This function writes the data to a file on hard disk.
        When writing the data the current grid-index (i_grid) is considered.
        The data are written to the row corresponding i_grid in order to
        maintain the order of the results when the SimulationModels are
        executed in parallel.

        Parameters
        ----------
        data : ndarray [n_sims x n_out]
            The data to write, the results of n outputs of m simulations
        coords : ndarray [n_sims x dim]
            Grid coordinates the simulations are conducted with
        coords_norm : ndarray [n_sims x dim], optional
            Normalized grid coordinates [-1, 1] the simulations are conducted with
        """
        if type(data) is list:
            data = np.array(data)

        if type(data) is float or type(data) is int \
                or type(data) is np.float64 or type(data) is np.int:
            data = np.array([[data]])

        if data.ndim == 1:
            data = data[np.newaxis, :]

        if type(coords) is list:
            coords = np.array(coords)

        if type(coords) is float or type(coords) is int \
                or type(coords) is np.float64 or type(coords) is np.int:
            coords = np.array([[coords]])

        if coords.ndim == 1:
            coords = coords[np.newaxis, :]

        if coords_norm is not None:
            if type(coords_norm) is list:
                coords_norm = np.array(coords_norm)

            if type(coords_norm) is float or type(coords_norm) is int or \
                    type(coords_norm) is np.float64 or type(coords_norm) is np.int:
                coords_norm = np.array([[coords_norm]])

            if coords_norm.ndim == 1:
                coords_norm = coords_norm[np.newaxis, :]

        if self.fn_results:
            self.lock.acquire()
            try:
                require_size = self.i_grid + 1
                with h5py.File(self.fn_results, 'a') as f:

                    # save results
                    try:
                        ds = f['results']
                        if ds.shape[0] < require_size:   # check if resize is necessary
                            ds.resize(require_size, axis=0)
                        ds[self.i_grid, :] = data
                    except (KeyError, ValueError):
                        ds = f.create_dataset('results', (require_size, data.shape[1]),
                                              maxshape=(None, data.shape[1]),
                                              dtype='float64')
                        ds[self.i_grid, :] = data

                    # save coords
                    try:
                        ds = f['coords']
                        if ds.shape[0] < require_size:   # check if resize is necessary
                            ds.resize(require_size, axis=0)
                        ds[self.i_grid, :] = coords
                    except (KeyError, ValueError):
                        ds = f.create_dataset('coords', (require_size, coords.shape[1]),
                                              maxshape=(None, coords.shape[1]),
                                              dtype='float64')
                        ds[self.i_grid, :] = coords

                    # save coords_norm
                    if coords_norm is not None:
                        try:
                            ds = f['coords_norm']
                            if ds.shape[0] < require_size:   # check if resize is necessary
                                ds.resize(require_size, axis=0)
                            ds[self.i_grid, :] = coords_norm
                        except (KeyError, ValueError):
                            ds = f.create_dataset('coords_norm', (require_size, coords_norm.shape[1]),
                                                  maxshape=(None, coords_norm.shape[1]),
                                                  dtype='float64')
                            ds[self.i_grid, :] = coords_norm

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
        This function prints the progress according to the current context and global_counter.
        """
        self.lock.acquire()
        try:
            if func_time:
                more_text = "Function evaluation took: " + repr(func_time) + "s"
            elif read_from_file:
                more_text = "Read data row from " + self.fn_results
            else:
                more_text = None

            display_fancy_bar("It/Sub-it: {}/{} Performing simulation".format(self.i_iter,
                                                                              self.i_subiter),
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
        It should perform the simulation task depending on the input_values provided to the object on instantiation.

        Parameters
        ----------
        process_id : int
            A unique identifier; no two processes of the pool will run concurrently with the same identifier
        """
        pass

    @abstractmethod
    def validate(self):
        """
        This abstract method must be implemented by the subclass.
        It should perform the validation task depending on the parameters defined in the problem.
        In cases, the model may not run correctly for some parameter combinations, this function changes the definition
        of the random parameters and the constants.
        """
        pass
