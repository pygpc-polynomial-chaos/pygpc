# -*- coding: utf-8 -*-
import numpy as np
import os
import h5py
import copy

from abc import ABCMeta, abstractmethod
from .misc import display_fancy_bar


class AbstractModel:
    """
    Abstract base class for the SimulationWrapper.
    This base class provides basic functions for serialization/deserialization and printing progress.
    It cannot be used directly, but a derived class implementing the "simulate" method must be created.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Constructor; initialized the SimulationWrapper class
        The model is initialized once. The parameters are set with the set_parameters class.
        Depending on the model, the user may call here some functions to initialize the model like
        starting a matlab engine etc...
        """
        pass

    def set_parameters(self, p, context=None):
        """
        Set model parameters and context of simulations.

        Parameters
        ----------
        p : dictionary
            Dictionary containing the model parameters
        context : dictionary
            dictionary that contains information about this worker's context
            - lock        : reference to the Lock object that all processes share for synchronization
            - max_grid    : size of the current sub-grid that is processed
            - global_task_counter  : reference to the Value object that is shared among all processes to keep track
                                     of the overall progress
            - seq_number  : sequence number of the task this object represents; necessary to maintain the correct
                            sequence of results
            - fn_results  : location of the hdf5 file to serialize the results to
            - i_grid      : current iteration in the sub-grid that is processed
            - i_iter      : current main-iteration
            - i_subiter   : current sub-iteration
            - coords      : parameters of particular simulation in original parameter space
            - coords_norm : parameters of particular simulation in normalized parameter space
        """

        self.p = p

        if context is not None:
            for key in context.keys():
                setattr(self, key, context[key])

        return copy.deepcopy(self)


    def read_previous_results(self, coords):
        """
        This functions reads previous results from the hard disk (if present).
        When reading from the array containing the results, the current
        grid-index (i_grid) is considered to maintain the order of the results when the
        SimulationModels are executed in parallel. If the function evaluated the results in parallel
        internally, i_grid is a range [i_grid_min, i_grid_max].

        Parameters
        ----------
        coords : ndarray of float [n_sims x dim]
            Grid coordinates the simulations are conducted with

        Returns
        -------
            None :
                if no serialized results could be found or does not fit to grid
            list :
                data at coords
        """
        if self.fn_results:
            if self.lock:
                self.lock.acquire()
            try:
                if os.path.exists(self.fn_results + ".hdf5"):

                    # read results and coords
                    try:
                        with h5py.File(self.fn_results + ".hdf5", 'r') as f:

                            if type(self.i_grid) is list:
                                res = f['model_evaluations/results'][self.i_grid[0]:self.i_grid[1], :]
                                coords_read = f['grid/coords'][self.i_grid[0]:self.i_grid[1], :]
                            else:
                                res = f['model_evaluations/results'][self.i_grid, :]
                                coords_read = f['grid/coords'][self.i_grid, :]

                            if np.isclose(coords_read, coords).all():
                                return res  #.tolist()
                            else:
                                return None

                    except (KeyError, ValueError):
                        return None
            finally:
                if self.lock:
                    self.lock.release()

        return None

    def write_results(self, data_dict):
        """
        This function writes the data to a file on hard disk.
        When writing the data the current grid-index (i_grid) is considered.
        The data are written to the row corresponding i_grid in order to
        maintain the order of the results when the SimulationModels are
        executed in parallel.

        Parameters
        ----------
        data_dict : dict of ndarray
            Dictionary, containing the data to write in an .hdf5 file. The keys are the dataset names.
        """

        if self.fn_results:     # full filename
            if self.lock:
                self.lock.acquire()
            try:
                # get new size of array
                if type(self.i_grid) is list:
                    require_size = np.max(self.i_grid)
                else:
                    require_size = self.i_grid + 1

                with h5py.File(self.fn_results + ".hdf5", 'a') as f:
                    for d in data_dict:
                        # # change list or single str to np.array
                        # if type(data_dict[d]) is list or type(data_dict[d]) is str:
                        #     data_dict[d] = np.array(data_dict[d]).flatten()

                        # change single numbers to np.array
                        # if type(data_dict[d]) is float or type(data_dict[d]) is int \
                        #         or type(data_dict[d]) is np.float64 or type(data_dict[d]) is np.int:
                        #     data_dict[d] = np.array([[data_dict[d]]]).flatten()

                        # # always flatten data because it has to be saved for every grid point
                        # if data_dict[d].ndim > 1:
                        #     data_dict[d] = data_dict[d].flatten()

                        # add axes such that it can be added to previous array
                        # if data_dict[d].ndim == 1:
                        #     data_dict[d] = data_dict[d][np.newaxis, :]

                        # check datatype
                        if type(data_dict[d][0][0]) is np.float64 or type(data_dict[d][0]) is float:
                            dtype='float64'
                        elif type(data_dict[d][0][0]) is np.int64:
                            dtype = 'int'
                        elif type(data_dict[d][0][0]) is np.string_ or type(data_dict[d][0][0]) is np.str_:
                            dtype = 'str'
                        else:
                            dtype='float64'

                        try:
                            ds = f[d]
                            # append
                            # for strings, the whole array has to be rewritten
                            if dtype is "str":
                                # ds = f[d][:]
                                ds = f[d]
                                del f[d]
                                ds = np.vstack((ds, data_dict[d]))
                                f.create_dataset(d, data=ds.astype("|S"))
                            else:
                                # change size of array and write data in it
                                if ds.shape[0] < require_size:  # check if resize is necessary
                                    ds.resize(require_size, axis=0)
                                if type(self.i_grid) is list:
                                    ds[self.i_grid[0]:self.i_grid[1], :] = data_dict[d]
                                else:
                                    ds[self.i_grid, :] = data_dict[d]

                        except (KeyError, ValueError, TypeError):
                            # create
                            try:
                                del f[d]
                            except KeyError:
                                pass

                            if dtype is "str":
                                f.create_dataset(d, data=data_dict[d].astype("|S"))
                            else:
                                ds = f.create_dataset(d, (require_size, data_dict[d].shape[1]),
                                                      maxshape=(None, data_dict[d].shape[1]),
                                                      dtype=dtype)

                                if type(self.i_grid) is list:
                                    ds[self.i_grid[0]:self.i_grid[1], :] = data_dict[d]
                                else:
                                    ds[self.i_grid, :] = data_dict[d]
            finally:
                if self.lock:
                    self.lock.release()

    def increment_ctr(self):
        """
        This functions increments the global counter by 1.
        """
        if self.lock:
            self.lock.acquire()
        try:
            if self.lock:
                self.global_task_counter.value += 1
            else:
                self.global_task_counter += 1
        finally:
            if self.lock:
                self.lock.release()

    def print_progress(self, func_time=None, read_from_file=False):
        """
        This function prints the progress according to the current context and global_counter.
        """
        if self.lock:
            self.lock.acquire()
        try:
            if func_time:
                more_text = "Function evaluation took: " + repr(func_time) + "s"
            elif read_from_file:
                more_text = "Read data from " + self.fn_results + ".hdf5"
            else:
                more_text = None

            if self.lock:
                global_task_counter = self.global_task_counter.value
            else:
                global_task_counter = self.global_task_counter

            display_fancy_bar("It/Sub-it: {}/{} Performing simulation".format(self.i_iter,
                                                                              self.i_subiter),
                              global_task_counter,
                              self.max_grid,
                              more_text)
        finally:
            if self.lock:
                self.lock.release()

    def get_seq_number(self):
        return self.seq_number

    @abstractmethod
    def simulate(self, process_id=None, matlab_engine=None):
        """
        This abstract method must be implemented by the subclass.
        It should perform the simulation task depending on the input_values provided to the object on instantiation.

        Parameters
        ----------
        process_id : int
            A unique identifier; no two processes of the pool will run concurrently with the same identifier
        matlab_engine : Matlab engine object
            Matlab engine to run Matlab models
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
