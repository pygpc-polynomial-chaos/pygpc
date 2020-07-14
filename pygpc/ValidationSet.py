import h5py
import os
from .misc import ten2mat
from .misc import mat2ten
from .Grid import Grid
from .Grid import Random
from .Computation import *


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
    problem : Problem instance, optional, default: None
        GPC problem (needed to create a Validation set without GPC instance)
    """

    def __init__(self, grid=None, results=None, gradient_results=None, gradient_idx=None, problem=None):
        """
        Initializes ValidationSet
        """
        self.grid = grid
        self.results = results
        self.gradient_results = gradient_results
        self.gradient_idx = gradient_idx
        self.problem = problem

    def create(self, grid=None, n_samples=None, n_cpu=1):
        """
        Creates a Validation set; Calls model and evaluates results; Provide either grid or number of samples.

        Parameters
        ----------
        grid : Grid instance, optional, default: None
            Grid instance the Validation set is computed with
        n_samples : int, optional, default: None
            Number of samples; if grid is provided, the validation set is created using the grid
        n_cpu : int, optional, default: 1
            Number of CPU cores to use to create validation set.
        """
        if grid is not None:
            self.grid = grid

        if self.grid is None and n_samples is not None:
            self.grid = Random(parameters_random=self.problem.parameters_random,
                               n_grid=n_samples)
        else:
            raise ValueError("Provide grid or n_samples to create a validation set.")

        # Evaluate original model at grid points
        com = Computation(n_cpu=n_cpu, matlab_model=self.problem.model.matlab_model)
        self.results = com.run(model=self.problem.model, problem=self.problem, coords=self.grid.coords)

        if self.results.ndim == 1:
            self.results = self.results[:, np.newaxis]

    def write(self, fname, folder=None, overwrite=False):
        """
        Save ValidationSet in .hdf5 format

        Parameters
        ----------
        fname : str
            Filename of ValidationSet containing the grid points and the results data
        folder : str, optional, default: None
            Path in .hdf5 file containing the validation set
        overwrite : bool, optional, default: False
            Overwrite existing validation set

        Returns
        -------
        <file> : .hdf5 file
            File containing the grid points in grid/coords and grid/coords_norm
            and the corresponding results in model_evaluations/results
        """
        if folder is None:
            folder = ""

        with h5py.File(fname, 'a') as f:

            try:
                f.create_dataset(folder + "/grid/coords", data=self.grid.coords)
                f.create_dataset(folder + "/grid/coords_norm", data=self.grid.coords_norm)
                f.create_dataset(folder + "/model_evaluations/results", data=self.results)

                if self.gradient_results is not None:
                    f.create_dataset(folder + "/model_evaluations/gradient_results",
                                     data=ten2mat(self.gradient_results))
                    f.create_dataset(folder + "/model_evaluations/gradient_results_idx",
                                     data=self.gradient_idx)

            except RuntimeError:
                if not overwrite:
                    pass
                else:
                    if folder == "":
                        folder_read = "/"
                    for key in f[folder_read].keys():
                        del f[folder + key]

                    f.create_dataset(folder + "/grid/coords", data=self.grid.coords)
                    f.create_dataset(folder + "/grid/coords_norm", data=self.grid.coords_norm)
                    f.create_dataset(folder + "/model_evaluations/results", data=self.results)

                    if self.gradient_results is not None:
                        f.create_dataset(folder + "/model_evaluations/gradient_results",
                                         data=ten2mat(self.gradient_results))
                        f.create_dataset(folder + "/model_evaluations/gradient_results_idx",
                                         data=self.gradient_idx)

    def read(self, fname, folder=None, coords_key=None, coords_norm_key=None, results_key=None, gradient_results_key=None,
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
        if folder is None:
            folder = ""

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
