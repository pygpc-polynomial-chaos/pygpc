# -*- coding: utf-8 -*-
import h5py
import os
from .misc import ten2mat
from .misc import mat2ten
from .Grid import Grid


class ValidationSet(object):
    """
    ValidationSet object
    """

    def __init__(self, grid=None, results=None, gradient_results=None):
        """
        Initializes ValidationSet

        Parameters
        ----------
        grid : Grid object
            Grid object containing the validation points (grid.coords, grid.coords_norm)
        results: ndarray [n_grid x n_out]
            Results of the model evaluation
        gradient_results: ndarray [n_grid x n_out x dim], optional, default=None
            Gradient of results of the model evaluations
        """
        self.grid = grid
        self.results = results
        self.gradient_results = gradient_results

    def write(self, fname):
        """ Save Validation set in .hdf5 format

        Parameters
        ----------
        fname : str
            Filename of ValidationSet containing the grid points and the results data

        Returns
        -------
        <file> : .hdf5 file
            File containing the grid points in grid/coords and grid/coords_norm
            and the corresponding results in model_evaluations/results
        """

        with h5py.File(os.path.splitext(fname)[0] + ".hdf5", 'w') as f:
            f["grid/coords"] = self.grid.coords
            f["grid/coords_norm"] = self.grid.coords_norm
            f["model_evaluations/results"] = self.results

            if self.gradient_results is not None:
                f["model_evaluations/gradient_results"] = ten2mat(self.gradient_results)

    def read(self, fname, coords_key=None, coords_norm_key=None, results_key=None, gradient_results_key=None):
        """ Load Validation set from .hdf5 format

        Parameters
        ----------
        fname : str
            Filename of ValidationSet containing the grid points and the results data
        coords_key : str, optional, default: "grid/coords"
            Path of coords in .hdf5 file
        coords_norm_key : str, optional, default: "grid/coords_norm"
            Path of coords_norm in .hdf5 file
        results_key : str, optional, default: "model_evaluations/results"
            Path of results in .hdf5 file
        gradient_results_key : str, optional, default: "model_evaluations/gradient_results"
            Path of gradient_results in .hdf5 file

        Returns
        -------
        val : ValidationSet Object
            ValidationSet object containing the grid points and the results data
        """

        if coords_key is None:
            coords_key = "grid/coords"

        if coords_norm_key is None:
            coords_norm_key = "grid/coords_norm"

        if results_key is None:
            results_key = "model_evaluations/results"

        if gradient_results_key is None:
            gradient_results_key = "model_evaluations/gradient_results"

        del self.results
        del self.gradient_results

        with h5py.File(os.path.splitext(fname)[0] + ".hdf5", 'r') as f:
            coords = f[coords_key][:]
            coords_norm = f[coords_norm_key][:]
            self.results = f[results_key][:]

            try:
                self.gradient_results = mat2ten(f[gradient_results_key][:])
            except KeyError:
                pass

        self.grid = Grid(parameters_random=[None]*coords.shape[1], coords=coords, coords_norm=coords_norm)

        return self
