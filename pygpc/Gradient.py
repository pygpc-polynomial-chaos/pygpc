import numpy as np
from .misc import ten2mat
from .misc import mat2ten
from .misc import get_all_combinations


def get_gradient(model, problem, grid, results, com,  method="FD_fwd",
                 gradient_results=None, i_iter=None, i_subiter=None, print_func_time=False, dx=1e-3, distance_weight=-1):
    """
    Determines the gradient of the model function in the grid points (self.grid.coords).
    The method to determine the gradient can be specified in self.options["gradient_calculation"] to be either:

    Parameters
    ----------
    model: Model object
        Model object instance of model to investigate (derived from AbstractModel class, implemented by user)
    problem: Problem class instance
        GPC Problem under investigation
    grid : Grid object
        Grid object
    results : ndarray of float [n_grid x n_out]
        Results of model function in grid points
    com : Computation class instance
        Computation class instance to run the computations
    method : str, optional, default: "FD_fwd"
        Gradient calculation method:
        - "FD_fwd": Finite difference forward approximation of the gradient using n_grid x dim additional sampling
        points stored in self.grid.coords_gradient and self.grid.coords_gradient_norm [n_grid x dim x dim].
        - "FD_1st": Finite difference approximation of 1st order accuracy using only the available samples
        - "FD_2nd": Finite difference approximation of 2nd order accuracy using only the available samples
    gradient_results : ndarray of float [n_grid_old x n_out x dim], optional, default: None
        Gradient of model function in grid points, already determined in previous calculations.
    i_iter : int (optional, default: None)
        Current iteration
    i_subiter : int (optional, default: None)
        Current sub-iteration
    dx : float, optional, default: 1e-3
        Distance parameter, depending on applied method:
        - "FW_fwd": Distance of new grid-points in each dim from orig. grid points to compute forward approximation
        - "FW_1st": Radius around grid points to include adjacent grid-points in 1st order gradient approximation
    distance_weight : float, optional, default: 1
        Distance weight factor (exponent) for methods "FD_1st" and "FD_2nd".
        Defines the importance of the adjacent grid points to estimate the gradient by their distance.

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
        if method == "FD_fwd":
            # add new grid points for gradient calculation in grid.coords_gradient and grid.coords_gradient_norm
            grid.create_gradient_grid(delta=dx)

            # determine model solutions at new grid points
            results_gradient_tmp = com.run(model=model,
                                           problem=problem,
                                           coords=ten2mat(grid.coords_gradient[n_gradient_results:, :, :]),
                                           coords_norm=ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]),
                                           i_iter=i_iter,
                                           i_subiter=i_subiter,
                                           fn_results=None,
                                           print_func_time=print_func_time,
                                           increment_grid=False)

            # distance tensor
            delta = np.repeat(np.linalg.norm(
                ten2mat(grid.coords_gradient_norm[n_gradient_results:, :, :]) - \
                ten2mat(np.repeat(grid.coords_norm[n_gradient_results:, :, np.newaxis], problem.dim, axis=2)),
                axis=1)[:, np.newaxis], results_gradient_tmp.shape[1], axis=1)

            # determine gradient
            gradient_results_new = (ten2mat(np.repeat(results[n_gradient_results:, :, np.newaxis],
                                                      problem.dim, axis=2)) - results_gradient_tmp) / delta

            gradient_results_new = mat2ten(mat=gradient_results_new, incr=problem.dim)

            if gradient_results is not None:
                gradient_results = np.vstack((gradient_results, gradient_results_new))
            else:
                gradient_results = gradient_results_new

        #########################################################
        # Finite difference approximation (1st order accuracy)  #
        #########################################################
        elif method == "FD_1st":
            gradient_results = np.zeros((grid.coords.shape[0], results.shape[1], problem.dim))*np.nan
            delta = dx

            for i, x0 in enumerate(grid.coords_norm):
                mask = np.linalg.norm(grid.coords_norm-x0, axis=1) < np.sqrt(delta)

                if np.sum(mask) > 1:
                    coords_norm_selected = grid.coords_norm[mask, ]

                    # distance matrix (1st order)
                    D = coords_norm_selected-x0

                    # rhs
                    df = results[mask, ]-results[i, ]

                    # weight matrix (distance**distance_weight)
                    dist_w = np.linalg.norm(D, axis=1)
                    dist_w[dist_w != 0] = dist_w[dist_w != 0]**distance_weight
                    W = np.diag(dist_w)

                    # apply weight matrix
                    D = np.matmul(W, D)
                    df = np.matmul(W, df)

                    # gradient [n_grid x n_out x dim]
                    gradient_results[i, :, :] = np.matmul(np.linalg.pinv(D), df).transpose()

                else:
                    continue

        #########################################################
        # Finite difference approximation (2nd order accuracy)  #
        #########################################################
        elif method == "FD_2nd":
            gradient_results = np.zeros((grid.coords.shape[0], results.shape[1], problem.dim))*np.nan
            delta = dx

            for i, x0 in enumerate(grid.coords_norm):
                mask = np.linalg.norm(grid.coords_norm-x0, axis=1) < np.sqrt(delta)

                if np.sum(mask) > 1:
                    coords_norm_selected = grid.coords_norm[mask, ]

                    # distance matrix (1st order)
                    D = coords_norm_selected-x0

                    # distance matrix (2nd order)
                    M = np.zeros((coords_norm_selected.shape[0], np.sum(np.arange(problem.dim+1))))

                    # quadratic terms
                    for i_dim in range(problem.dim):
                        M[:, i_dim] = 0.5 * (coords_norm_selected[:, i_dim]-x0[i_dim])**2

                    # mixed linear terms
                    idx = get_all_combinations(np.arange(problem.dim), 2)

                    for j, idx_row in enumerate(idx):
                        M[:, j+problem.dim] = (coords_norm_selected[:, idx_row[0]] - x0[idx_row[0]]) * \
                                              (coords_norm_selected[:, idx_row[1]] - x0[idx_row[1]])

                    # rhs
                    df = results[mask, ]-results[i, ]

                    # weight matrix (distance**distance_weight)
                    dist_w = np.linalg.norm(D, axis=1)
                    dist_w[dist_w != 0] = dist_w[dist_w != 0]**distance_weight
                    W = np.diag(dist_w)

                    # apply weight matrix
                    D = np.matmul(W, D)
                    M = np.matmul(W, M)
                    df = np.matmul(W, df)

                    # derive orthogonal reduction of M
                    Q, T = np.linalg.qr(M, mode="complete")

                    # gradient [n_grid x n_out x dim]
                    QtD_inv = np.linalg.pinv(np.matmul(Q.transpose(), D)[3:, ])

                    rhs = np.matmul(Q.transpose(), df)[3:, ]

                    gradient_results[i, :, :] = np.matmul(QtD_inv, rhs).transpose()

                else:
                    continue
        else:
            raise NotImplementedError("Please provide a valid gradient estimation method!")

    return gradient_results
