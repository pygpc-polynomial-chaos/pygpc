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
import re
import os
import h5py

import time

from subprocess import check_output
from shutil import copytree
from shutil import rmtree
from abc import ABCMeta, abstractmethod
from .misc import fancy_bar


class AbstractSimulationWrapper:
    """
    Abstract base class for the SimulationWrapper.
    This base class provides basic functions for serialization/deserialization and printing progress.
    It cannot be used directly, but a derived class implementing the "simulate" method must be created.
    """
    __metaclass__ = ABCMeta

    def __init__(_self, _context):
        """
        Constructor; initialized the SimulationWrapper class with the provided context

        Parameters
        ----------
        _context : dictionary
            dictionary that contains information about this worker's context
                - save_res_fn : location of the hdf5 file to serialize the results to
                - i_grid      : current iteration in the subgrid that is processed
                - max_grid    : size of the current subgrid that is processed
                - i_iter      : current main iteration
                - interaction_oder_current : current interaction order
                - lock        : reference to the Lock object that all processes share for synchronization
                - global_ctr  : reference to the Value object that is shared among all processes to keep track
                                of the overall progress
                - task_number : number of the task this object represents; necessary to maintain the correct
                                sequence of results
        """
        _self.save_res_fn = _context['save_res_fn']
        _self.i_grid = _context['i_grid']
        _self.i_iter = _context['i_iter']
        _self.lock = _context['lock']
        _self.max_grid = _context['max_grid']
        _self.interaction_oder_current = _context['interaction_order_current']
        _self.global_task_counter = _context['global_task_ctr']
        _self.task_number = _context['task_number']

    def read_previous_results( _self ):
        """
        This functions reads serialized previous results from the hard disk (if present)

        :return:
            None :
                if no serialized results could be found
            list :
                containing the read data
        """
        _self.lock.acquire()
        try:
            if os.path.exists( _self.save_res_fn ):
                try:
                    with h5py.File( _self.save_res_fn, 'r') as f:
                        # get datset
                        ds = f['res']
                        # ... then read res from file
                        res = ds[_self.i_grid, :]

                        return res

                except (KeyError, ValueError):
                    return None
        finally:
            _self.lock.release()

        return None

    def write_new_results( _self, _data):
        """
        This function writes the data to a file on harddisk.

        Parameters
        ----------
        _data : ndarray, m x n
            The data to write, the results at n mesh points of m simulations
        """
        _self.lock.acquire()
        try:
            with h5py.File(_self.save_res_fn, 'a') as f:
                try:
                    ds = f['res']
                    ds.resize(ds.shape[0] + 1, axis=0)
                    ds[ds.shape[0] - 1, :] = _data[np.newaxis, :]
                except (KeyError,ValueError):
                    f.create_dataset('res', data=_data[np.newaxis, :], maxshape=(None, len(_data)))
        finally:
            _self.lock.release()

    def increment_ctr( _self ):
        """
        This functions increments the global counter by 1.
        """
        _self.lock.acquire()
        try:
            _self.global_task_counter.value += 1
        finally:
            _self.lock.release()

    def print_progress( _self, _func_time=None, _read_from_file=False ):
        """
        This functions print the progress according to the current context and global_counter.
        """
        _self.lock.acquire()
        try:
            if _func_time:
                more_text = "Function evaluation took: " + repr(_func_time ) + "s"
            elif _read_from_file:
                more_text = "Read data row from " + _self.save_res_fn
            else:
                more_text = None

            fancy_bar("It/Subit: {}/{} Performing simulation".format(_self.i_iter,
                                                                     _self.interaction_oder_current),
                                                                     _self.global_task_counter.value,
                                                                     _self.max_grid,
                                                                     more_text)
        finally:
            _self.lock.release()

    def wait(_self):
        while True:
            if _self.global_task_counter.value == _self.task_number:
                return
            time.sleep(0.1)


    @staticmethod
    @abstractmethod
    def factory( _input_values, _context, _args ):
        """
        This abstract method must be implemented by the subclass.
        This factory method should return an object of the subclass.

        Parameters
        ----------
        _input_values : list
            a list of input values passed on to the simulation

        _context : dictionary
            contains information about this worker's context
            (see documentation of the constructor for further information)

        _args : list
            optional argument list that will be unrolled when instantiating the subclass

        :return
            Object of the derived type of AbstractSimulationWrappe
        """
        pass

    @abstractmethod
    def simulate( _self, _process_id ):
        """
        This abstract method must be implemented by the subclass.
        It should perform the simulation task depding on the input_values
        provided to the object on instantiation.

        Parameters
        ----------
        _process_id : int
            A unique identifier; no processes of the pool will run at once with the same identifier
        """
        pass

class SimulationWrapper(AbstractSimulationWrapper):
    # OF_CASE_DIR = "/home/kalloch/OpenFOAM/kalloch-3.0.1/run/PyGPC_head"
    OF_CASE_DIR_BASE = "/home/kalloch/OpenFOAM/kalloch-3.0.1/run/PyGPC_capacitor"

    def __init__(_self, _conductivities, _context, _n_cpu):
        super(SimulationWrapper, _self).__init__( _context )

        _self.conductivities = _conductivities
        _self.n_cpu          = _n_cpu

    def simulate( _self, _process_id ):
        _self.lock.acquire()
        try:
            print "Global Ctr = " + repr(_self.global_task_counter.value)
            print "task_num = " + repr(_self.task_number)
        finally:
            _self.lock.release()

        OF_CASE_DIR = SimulationWrapper.OF_CASE_DIR_BASE + "_" + repr( _process_id )

        scalp_cond = repr( _self.conductivities[0] )
        skull_cond = repr( _self.conductivities[1] )
        gm_cond = repr( _self.conductivities[2] )
        wm_cond = repr( _self.conductivities[3] )

        # Step 1: setup the case directory by replacing the conductivities with the ones provided by the PyGPC framework
        rmtree(OF_CASE_DIR + "/0", ignore_errors=True)
        copytree(OF_CASE_DIR + "/0_clean", OF_CASE_DIR + "/0")

        with open(OF_CASE_DIR + "/0/sigma") as f:
            sigma = f.read()

        sigma = sigma.replace("%SKIN_VAL%", scalp_cond)
        sigma = sigma.replace("%SKULL_VAL%", skull_cond)
        sigma = sigma.replace("%CSF_VAL%", "1.65")
        sigma = sigma.replace("%GM_VAL%", gm_cond)
        sigma = sigma.replace("%WM_VAL%", wm_cond)
        sigma = sigma.replace("%ELECTRODE_VAL%", "1.4")

        with open(OF_CASE_DIR + "/0/sigma", "w") as f:
            f.write(sigma)

        # Step 2: run the simulation
        # The script is located in the case-dir and we need to provide the case-dir as an argument to the solver application
        stdout_string = check_output(["bash " + OF_CASE_DIR + "/runSim.sh " + OF_CASE_DIR], shell=True)

        # We must check the number of iterations, because the simualtion results will be stored in a
        # directory with that number
        #print "*************************************** Solver output - start *************************************"
        #print stdout_string
        #print "*************************************** Solver output - end *************************************"

        regex_result = re.search("SIMPLE solution converged in ([0-9]+) iterations", stdout_string);
        num_iterations = int(regex_result.group(1))

        # Step 3: Query the results
        internal_field_flag = False
        num_lines_to_read = -1

        with open(OF_CASE_DIR + "/" + repr(num_iterations + 1) + "/ElPot") as f:
            for line in f:
                if "internalField" in line:
                    num_lines_to_read = int(next(f))
                    next(f)
                    break

            ElPot = np.zeros(num_lines_to_read, dtype='float64')

            for i in range(0, num_lines_to_read):
                ElPot[i] = float(next(f))

            # check and wait for other processes that should be
            # finished before this one
            # -> We must maintain the sequence of the result matrix
            _self.wait()

            return ElPot

        return np.array([])

    @staticmethod
    def factory( _input_values, _context, args ):
        return SimulationWrapper( _input_values, _context, *(args) )

    @staticmethod
    def write_result_field( _fieldName, _data):
        out_dir = SimulationWrapper.OF_CASE_DIR_BASE + "/999/"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(SimulationWrapper.OF_CASE_DIR_BASE + "/Field_template") as f:
            template = f.read()

        data_str = np.char.mod('%f', _data)
        data_str = "\n".join(data_str)

        template = template.replace("%FIELDNAME%", _fieldName)
        template = template.replace("%NUM_VALUES%", repr(len(_data)))
        template = template.replace("%DATA%", data_str)

        with open(out_dir + "/" + _fieldName, "w") as f:
            f.write(template)
