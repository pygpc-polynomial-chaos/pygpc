import h5py
import os
from .misc import ten2mat
from .misc import mat2ten
from .Grid import Grid


class ValidationSet(object):
    """
    ValidationSet object

    Parameters
    ----------
    grid : Grid object
        Grid object containing the validation points (grid.coords, grid.coords_norm)
    results : ndarray [n_grid x n_out]
        Results of the model evaluation
    gradient_results : ndarray [n_grid x n_out x dim], optional, default=None
        Gradient of results of the model evaluations
    gradient_idx : ndarray of int [n_grid]
        Indices of grid points where the gradient was evaluated
    """

    def __init__(self, grid=None, results=None, gradient_results=None, gradient_idx=None):
        """
        Initializes ValidationSet
        """
        self.grid = grid
        self.results = results
        self.gradient_results = gradient_results
        self.gradient_idx = gradient_idx

    def write(self, fname, folder):
        """ Save ValidationSet in .hdf5 format

        Parameters
        ----------
        fname : str
            Filename of ValidationSet containing the grid points and the results data
        folder : str
            Path in .hdf5 file containing the validation set

        Returns
        -------
        <file> : .hdf5 file
            File containing the grid points in grid/coords and grid/coords_norm
            and the corresponding results in model_evaluations/results
        """

        with h5py.File(fname, 'a') as f:
            f[folder + "/grid/coords"] = self.grid.coords
            f[folder + "/grid/coords_norm"] = self.grid.coords_norm
            f[folder + "/model_evaluations/results"] = self.results

            if self.gradient_results is not None:
                f[folder + "/model_evaluations/gradient_results"] = ten2mat(self.gradient_results)
                f[folder + "/model_evaluations/gradient_results_idx"] = self.gradient_idx

    def read(self, fname, folder, coords_key=None, coords_norm_key=None, results_key=None, gradient_results_key=None,
             gradient_idx_key=None):
        """ Load Validation set from .hdf5 format

        Parameters
        ----------
        fname : str
            Filename of ValidationSet containing the grid points and the results data
        folder : str
            Path in .hdf5 file containing the validation set
        coords_key : str, optional, default: "grid/coords"
            Path of coords in .hdf5 file
        coords_norm_key : str, optional, default: "grid/coords_norm"
            Path of coords_norm in .hdf5 file
        results_key : str, optional, default: "model_evaluations/results"
            Path of results in .hdf5 file
        gradient_results_key : str, optional, default: "model_evaluations/gradient_results"
            Path of gradient_results in .hdf5 file
        gradient_idx_key : str, optional, default: "model_evaluations/gradient_results_idx"
            Path of gradient_results in .hdf5 file

        Returns
        -------
        val : ValidationSet Object
            ValidationSet object containing the grid points and the results data
        """

        if coords_key is None:
            coords_key = folder + "/grid/coords"

        if coords_norm_key is None:
            coords_norm_key = folder + "/grid/coords_norm"

        if results_key is None:
            results_key = folder + "/model_evaluations/results"

        if gradient_results_key is None:
            gradient_results_key = folder + "/model_evaluations/gradient_results"

        if gradient_idx_key is None:
            gradient_idx_key = folder + "/model_evaluations/gradient_results_idx"

        del self.results
        del self.gradient_results
        del self.gradient_idx

        with h5py.File(fname, 'r') as f:
            coords = f[coords_key][:]
            coords_norm = f[coords_norm_key][:]
            self.results = f[results_key][:]

            try:
                self.gradient_results = mat2ten(f[gradient_results_key][:])
                self.gradient_idx = f[gradient_idx_key]
            except KeyError:
                pass

        self.grid = Grid(parameters_random=[None]*coords.shape[1], coords=coords, coords_norm=coords_norm)

        return self
