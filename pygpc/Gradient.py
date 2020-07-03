import numpy as np
from .misc import ten2mat
from .misc import mat2ten
from .misc import get_all_combinations


def get_gradient(model, problem, grid, results, com,  method="FD_fwd",
                 gradient_results_present=None, gradient_idx_skip=None,
                 i_iter=None, i_subiter=None, print_func_time=False,
                 dx=1e-3, distance_weight=-1, verbose=False):
    """
    Determines the gradient of the model function in the grid points (self.grid.coords).
    The method to determine the gradient can be specified in self.options["gradient_calculation"].
    The new gradients and their indices are appended to the old results given in gradient_results_present and
    gradient_idx_skip.

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
        - "FD_1st": Finite difference approximation of 1st order accuracy using only the available samples [1]
        - "FD_2nd": Finite difference approximation of 2nd order accuracy using only the available samples [1]
        - "FD_1st2nd": Finite difference approximation of 1st and (where possible) 2nd order accuracy
        using only the available samples [1]
    gradient_results_present : ndarray of float [n_grid_old x n_out x dim], optional, default: None
        Gradient of model function in grid points, already determined in previous calculations.
        Those values will not be updated!
    gradient_idx_skip: ndarray of int [n_gradient_results.shape[0]]
        Indices of grid points where the gradient was already computed and is provided in gradient_results.
        Those grid points will be skipped.
    i_iter : int (optional, default: None)
        Current iteration
    i_subiter : int (optional, default: None)
        Current sub-iteration
    dx : float, optional, default: 1e-3
        Distance parameter, depending on applied method:
        - "FW_fwd": Distance of new grid-points in each dim from orig. grid points to compute forward approximation
        - "FW_1st": Radius around grid points to include adjacent grid-points in 1st order gradient approximation
        - "FW_2nd": Radius around grid points to include adjacent grid-points in 1st order gradient approximation
    distance_weight : float, optional, default: 1
        Distance weight factor (exponent) for methods "FD_1st" and "FD_2nd".
        Defines the importance of the adjacent grid points to estimate the gradient by their distance.
    verbose : bool, optional, default: False
        Print progress

    Returns
    -------
    gradient_results : ndarray of float [n_grid x n_out x dim]
        Gradient of model function in grid points
    gradient_results_idx : ndarray of int [n_grid]
        Indices of grid points where the gradient was evaluated

    Notes
    -----
    .. [1] Belward J, Turner IW, Ilic M, On derivative estimation and the solution of least squares problems,
       Journal of Computational and Applied Mathematics 2008, vol. 222, pp. 511-523.
    """
    gradient_results_new = None

    if gradient_results_present is not None:
        n_gradient_results = gradient_results_present.shape[0]
    else:
        n_gradient_results = 0

    if gradient_idx_skip is None:
        gradient_idx_skip = np.array([])

    gradient_idx_compute = np.arange(grid.coords.shape[0])

    if gradient_idx_skip is not None:
        gradient_idx_compute = np.delete(gradient_idx_compute, gradient_idx_skip.astype(int))

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
                                           coords=ten2mat(grid.coords_gradient[gradient_idx_compute, :, :]),
                                           coords_norm=ten2mat(grid.coords_gradient_norm[gradient_idx_compute, :, :]),
                                           i_iter=i_iter,
                                           i_subiter=i_subiter,
                                           fn_results=None,
                                           print_func_time=print_func_time,
                                           increment_grid=False,
                                           verbose=verbose)

            # distance tensor
            delta = np.repeat(np.linalg.norm(
                ten2mat(grid.coords_gradient_norm[gradient_idx_compute, :, :]) - \
                ten2mat(np.repeat(grid.coords_norm[gradient_idx_compute, :, np.newaxis], problem.dim, axis=2)),
                axis=1)[:, np.newaxis], results_gradient_tmp.shape[1], axis=1)

            # determine gradient
            gradient_results_new = (ten2mat(np.repeat(results[gradient_idx_compute, :, np.newaxis],
                                                      problem.dim, axis=2)) - results_gradient_tmp) / delta

            gradient_results_new = mat2ten(mat=gradient_results_new, incr=problem.dim)

            gradient_results_idx = np.hstack((gradient_idx_skip, gradient_idx_compute)).astype(int)

        #########################################################
        # Finite difference approximation (1st order accuracy)  #
        #########################################################
        elif method == "FD_1st":
            delta = dx

            # number of sampling points to sacrifice for 2nd order accuracy
            gradient_results_idx_can_compute = []

            for i, x0 in enumerate(grid.coords_norm):

                # determine neighbors within radius delta
                mask = np.linalg.norm(grid.coords_norm-x0, axis=1) < delta
                mask[i] = False

                # only determine gradient if we have enough neighboring sampling points
                # and if it was not computed previously
                if np.sum(mask) >= problem.dim and i in gradient_idx_compute:
                    gradient_results_idx_can_compute.append(i)

            # determine gradient not in skipped points
            gradient_results_idx_can_compute = np.array(gradient_results_idx_can_compute)

            if gradient_results_idx_can_compute.any():
                gradient_results_new = FD_1st(coords_norm=grid.coords_norm,
                                              coord_idx=gradient_results_idx_can_compute,
                                              results=results,
                                              dx=dx,
                                              distance_weight=distance_weight)

            gradient_results_idx = np.hstack((gradient_idx_skip, gradient_results_idx_can_compute)).astype(int)

        #########################################################
        # Finite difference approximation (2nd order accuracy)  #
        #########################################################
        elif method == "FD_2nd":
            delta = dx

            # number of sampling points to sacrifice for 2nd order accuracy
            n_2nd_order = np.sum(np.arange(problem.dim+1))
            gradient_results_idx_can_compute = []

            for i, x0 in enumerate(grid.coords_norm):

                # determine neighbors within radius delta
                mask = np.linalg.norm(grid.coords_norm-x0, axis=1) < delta
                mask[i] = False

                # only determine gradient if we have enough neighboring sampling points
                # and if it was not computed previously
                if np.sum(mask) >= (n_2nd_order + problem.dim) and i in gradient_idx_compute:
                    gradient_results_idx_can_compute.append(i)

            # determine gradient not in skipped points
            gradient_results_idx_can_compute = np.array(gradient_results_idx_can_compute)

            if gradient_results_idx_can_compute.any():
                gradient_results_new = FD_2nd(coords_norm=grid.coords_norm,
                                              coord_idx=gradient_results_idx_can_compute,
                                              results=results,
                                              dx=dx,
                                              distance_weight=distance_weight)

            gradient_results_idx = np.hstack((gradient_idx_skip, gradient_results_idx_can_compute)).astype(int)

        #################################################################
        # Finite difference approximation (1st and 2nd order accuracy)  #
        #################################################################
        elif method == "FD_1st2nd":
            delta = dx

            # number of sampling points to sacrifice for 2nd order accuracy
            n_2nd_order = np.sum(np.arange(problem.dim+1))
            coord_idx_1st = []
            coord_idx_2nd = []

            for i, x0 in enumerate(grid.coords_norm):

                # determine neighbors within radius delta
                mask = np.linalg.norm(grid.coords_norm-x0, axis=1) < delta
                mask[i] = False

                # choose method depending on number of neighboring sampling points
                # and if it was not computed previously
                if np.sum(mask) >= (n_2nd_order + problem.dim) and i in gradient_idx_compute:
                    coord_idx_2nd.append(i)
                elif np.sum(mask) >= problem.dim and i in gradient_idx_compute:
                    coord_idx_1st.append(i)

            coord_idx_1st = np.array(coord_idx_1st)
            coord_idx_2nd = np.array(coord_idx_2nd)

            # estimate gradients with 1st order accuracy
            gradient_results_1st = None
            if coord_idx_1st.any():
                gradient_results_1st = FD_1st(coords_norm=grid.coords_norm,
                                              coord_idx=coord_idx_1st,
                                              results=results,
                                              dx=dx,
                                              distance_weight=distance_weight)

            # estimate gradients with 2nd order accuracy
            gradient_results_2nd = None
            if coord_idx_2nd.any():
                gradient_results_2nd = FD_2nd(coords_norm=grid.coords_norm,
                                              coord_idx=coord_idx_2nd,
                                              results=results,
                                              dx=dx,
                                              distance_weight=distance_weight)

            # concatenate results
            if gradient_results_1st is not None and gradient_results_2nd is not None:
                gradient_results_new = np.vstack((gradient_results_1st, gradient_results_2nd))
            elif gradient_results_1st is not None and gradient_results_2nd is None:
                gradient_results_new = gradient_results_1st
            elif gradient_results_1st is not None and gradient_results_2nd is None:
                gradient_results_new = gradient_results_2nd

            gradient_results_idx = np.hstack((gradient_idx_skip, coord_idx_1st, coord_idx_2nd)).astype(int)

        else:
            raise NotImplementedError("Please provide a valid gradient estimation method!")

    # concatenate old with new results
    if gradient_results_present is not None and gradient_results_new is not None:
        gradient_results = np.vstack((gradient_results_present, gradient_results_new))
    elif gradient_results_present is None and gradient_results_new is not None:
        gradient_results = gradient_results_new
    elif gradient_results_present is not None and gradient_results_new is None:
        gradient_results = gradient_results_present
    else:
        gradient_results = None

    if gradient_results_idx.size == 0:
        gradient_results_idx = None

    return gradient_results, gradient_results_idx


def FD_1st(coords_norm, coord_idx, results, dx, distance_weight):
    """
    Determines the gradients of "results" in coords_norm[coords_idx, :] using a finite difference
    regression approach of first order accuracy.

    Parameters
    ----------
    coords_norm : ndarray of float [n_grid x dim]
        Normalized coordinates xi
    coord_idx : ndarray of int [n_coords_idx]
        Indices of coordinates (row idx in coords_norm) where the gradient has to be computed
    results : ndarray of float [n_grid x n_qoi]
        QOI results of the model at grid points
    dx : float, optional, default: 1e-3
        Radius around grid points to include adjacent grid-points in gradient approximation
    distance_weight : float, optional, default: 1
        Distance weight factor (exponent) adjacent grid points

    Returns
    -------
    gradient_results : ndarray of float [n_coords_idx x n_qoi x dim]
        Gradient of model function in grid points
    """

    n_dim = coords_norm.shape[1]
    gradient_results = np.zeros((len(coord_idx), results.shape[1], n_dim))*np.nan

    for i, i_c in enumerate(coord_idx):
        x0 = coords_norm[i_c, :]

        # determine neighbors within radius dx
        mask = np.linalg.norm(coords_norm-x0, axis=1) < dx
        mask[i_c] = False

        coords_norm_selected = coords_norm[mask, ]

        # distance matrix (1st order)
        D = coords_norm_selected-x0

        # rhs
        df = results[mask, ]-results[i_c, ]

        # weight matrix (distance**distance_weight)
        W = np.diag(np.linalg.norm(D, axis=1)**distance_weight)

        # apply weight matrix
        D = np.matmul(W, D)
        df = np.matmul(W, df)

        # gradient [n_grid x n_out x dim]
        gradient_results[i, :, :] = np.matmul(np.linalg.pinv(D), df).transpose()

    return gradient_results


def FD_2nd(coords_norm, coord_idx, results, dx, distance_weight):
    """
    Determines the gradients of "results" in coords_norm[coords_idx, :] using a finite difference
    regression approach of second order accuracy.

    Parameters
    ----------
    coords_norm : ndarray of float [n_grid x dim]
        Normalized coordinates xi
    coord_idx : ndarray of int [n_coords_idx]
        Indices of coordinates (row idx in coords_norm) where the gradient has to be computed
    results : ndarray of float [n_grid x n_qoi]
        QOI results of the model at grid points
    dx : float, optional, default: 1e-3
        Radius around grid points to include adjacent grid-points in gradient approximation
    distance_weight : float, optional, default: 1
        Distance weight factor (exponent) adjacent grid points.

    Returns
    -------
    gradient_results : ndarray of float [n_coords_idx x n_qoi x dim]
        Gradient of model function in grid points
    """
    n_dim = coords_norm.shape[1]
    gradient_results = np.zeros((len(coord_idx), results.shape[1], n_dim))*np.nan

    # number of sampling points to sacrifice for 2nd order accuracy
    n_2nd_order = np.sum(np.arange(coords_norm.shape[1]+1))

    for i, i_c in enumerate(coord_idx):
        x0 = coords_norm[i_c, :]

        # determine neighbors within radius dx
        mask = np.linalg.norm(coords_norm-x0, axis=1) < dx
        mask[i_c] = False

        coords_norm_selected = coords_norm[mask, ]

        # distance matrix (1st order)
        D = coords_norm_selected-x0

        # distance matrix (2nd order)
        M = np.zeros((coords_norm_selected.shape[0], n_2nd_order))

        # quadratic terms
        for i_dim in range(n_dim):
            M[:, i_dim] = 0.5 * (coords_norm_selected[:, i_dim]-x0[i_dim])**2

        # mixed linear terms
        idx = get_all_combinations(np.arange(n_dim), 2)

        for j, idx_row in enumerate(idx):
            M[:, j+n_dim] = (coords_norm_selected[:, idx_row[0]] - x0[idx_row[0]]) * \
                            (coords_norm_selected[:, idx_row[1]] - x0[idx_row[1]])

        # rhs
        df = results[mask, ]-results[i_c, ]

        # weight matrix (distance**distance_weight)
        W = np.diag(np.linalg.norm(D, axis=1)**distance_weight)

        # apply weight matrix
        D = np.matmul(W, D)
        M = np.matmul(W, M)
        df = np.matmul(W, df)

        # derive orthogonal reduction of M
        Q, T = np.linalg.qr(M, mode="complete")

        # gradient [n_grid x n_out x dim]
        QtD_inv = np.linalg.pinv(np.matmul(Q.transpose(), D)[n_2nd_order:, ])

        rhs = np.matmul(Q.transpose(), df)[n_2nd_order:, ]

        gradient_results[i, :, :] = np.matmul(QtD_inv, rhs).transpose()

    return gradient_results
