import numpy as np
from .misc import ten2mat
from .misc import mat2ten


def get_gradient(model, problem, grid, results, com,  method="standard_forward",
                 gradient_results=None, i_iter=None, i_subiter=None, print_func_time=False):
    """
    Determines the gradient of the model function in the grid points (self.grid.coords).
    The method to determine the gradient can be specified in self.options["gradient_calculation"] to be either:

    - "standard_forward" ... Forward approximation of the gradient using n_grid x dim additional sampling points
      stored in self.grid.coords_gradient and self.grid.coords_gradient_norm [n_grid x dim x dim].
    - "???" ... ???

    Parameters
    ----------
    grid : Grid object
        Grid object
    results : ndarray of float [n_grid x n_out]
        Results of model function in grid points
    com : Computation class instance
        Computation class instance to run the computations
    gradient_results : ndarray of float [n_grid_old x n_out x dim], optional, default: None
        Gradient of model function in grid points, already determined in previous calculations.
    i_iter : int (optional, default: None)
        Current iteration
    i_subiter : int (optional, default: None)
        Current sub-iteration

    Returns
    -------
    gradient_results : ndarray of float [n_grid x n_out x dim]
        Gradient of model function in grid points
    """
    if gradient_results is not None:
        n_gradient_results = gradient_results.shape[0]
    else:
        n_gradient_results = 0

    if n_gradient_results < results.shape[0]:
        #########################################
        # Standard forward gradient calculation #
        #########################################
        if method == "standard_forward":
            # add new grid points for gradient calculation in grid.coords_gradient and grid.coords_gradient_norm
            grid.create_gradient_grid(delta=1e-3)

            results_gradient_tmp = com.run(model=model,
                                           problem=problem,
                                           coords=ten2mat(grid.coords_gradient[n_gradient_results:, :, :]),
                                           coords_norm=ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]),
                                           i_iter=i_iter,
                                           i_subiter=i_subiter,
                                           fn_results=None,
                                           print_func_time=print_func_time,
                                           increment_grid=False)

            delta = np.repeat(np.linalg.norm(
                ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]) - \
                ten2mat(np.repeat(grid.coords_norm[n_gradient_results:, :, np.newaxis], problem.dim, axis=2)),
                axis=1)[:, np.newaxis], results_gradient_tmp.shape[1], axis=1)

            gradient_results_new = (ten2mat(np.repeat(results[n_gradient_results:, :, np.newaxis],
                                                      problem.dim, axis=2)) - results_gradient_tmp) / delta

            gradient_results_new = mat2ten(mat=gradient_results_new, incr=problem.dim)

            if gradient_results is not None:
                gradient_results = np.vstack((gradient_results, gradient_results_new))
            else:
                gradient_results = gradient_results_new

        ##############################
        # Gradient approximation ... #
        ##############################
        # TODO: implement gradient approximation schemes using neighbouring grid points

    return gradient_results