from __future__ import print_function
import time
import numpy as np
import scipy.special
import scipy.stats
import dill           # module for saving and loading object instances
import pickle         # module for saving and loading object instances
import h5py
from sklearn import linear_model, metrics, model_selection

from .grid import randomgrid
from .grid import norm
from .ni import calc_Nc, reg

from .misc import unique_rows
from .misc import allVL1leq


def error_validation_set(gpc, coeffs, validation_set_data, validation_set_coords):
    ''' Calculater the error of the expansion gpc over a validation set '''
    rel_err = np.zeros(len(validation_set_data))
    abs_err = np.zeros(len(validation_set_data))
    i = 0
    for i, c, d in zip(range(len(validation_set_data)), validation_set_coords, validation_set_data):
        print(i+1, len(validation_set_data), end='\r')
        norm_coords = norm(
            c[None, :], gpc.pdftype, gpc.pdfshape, gpc.limits)
        d_approx = gpc.evaluate(coeffs, norm_coords).squeeze()
        abs_err[i] = np.linalg.norm(d - d_approx)
        rel_err[i] = abs_err[i] / np.linalg.norm(d)

    error_dictionary = {}
    error_dictionary['mean_rel_error'] = np.mean(rel_err)
    error_dictionary['max_rel_error'] = np.max(rel_err)

    error_dictionary['mean_abs_error'] = np.mean(abs_err)
    error_dictionary['max_abs_error'] = np.max(abs_err)
    return error_dictionary


def run_reg_adaptive_validation(pdftype, pdfshape, limits, train_set_data,
                                train_set_coords,
                                validation_set_data=None,
                                validation_set_coords=None,
                                order_start=0, order_end=10, eps=1E-3,
                                data_poly_ratio=1.2,
                                alphas=np.logspace(-5, 3, 9),
                                record_loocv=False,
                                print_out=False):
    """ 
    Adaptive regression approach based on leave one out cross validation error
    estimation
    
    Parameters
    ----------

    pdftype : list
              Type of probability density functions of input parameters,
              i.e. ["beta", "norm",...]
    pdfshape : list of lists
               Shape parameters of probability density functions
               s1=[...] "beta": p, "norm": mean
               s2=[...] "beta": q, "norm": std
               pdfshape = [s1,s2]
    limits : list of lists
             Upper and lower bounds of random variables (only "beta")
             a=[...] "beta": lower bound, "norm": n/a define 0
             b=[...] "beta": upper bound, "norm": n/a define 0
             limits = [a,b]
    func : callable func(x,*args)
           The objective function to be minimized.
    args : tuple, optional
           Extra arguments passed to func, i.e. f(x,*args).         
    order_start : int, optional
                  Initial gpc expansion order (maximum order)
    order_end : int, optional
                Maximum gpc expansion order to expand to
    eps : float, optional
          Relative mean error of leave one out cross validation
    print_out : boolean, optional
          Print output of iterations and subiterations (True/False)      

    Returns
    -------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]        
    """
    
    # initialize iterators
    eps_gpc = eps+1
    i_grid = 0
    i_iter = 0
    interaction_order_count = 0
    interaction_order_max = -1
    DIM = len(pdftype)
    order = order_start

    error_dictionary = {}

    if validation_set_data is not None:
        error_dictionary['mean_rel_error'] = []
        error_dictionary['max_rel_error'] = []

        error_dictionary['mean_abs_error'] = []
        error_dictionary['max_abs_error'] = []
    error_dictionary['CV'] = []
    if record_loocv:
        error_dictionary['LOOCV'] = []
    error_dictionary['N_simus'] = []
    N_train_set = train_set_coords.shape[0]
    train_set_to_choose = np.arange(N_train_set, dtype=int)

    def choose_from_train_set(N):
        chosen = np.random.choice(train_set_to_choose, int(N), replace=False)
        train_set = train_set_to_choose[~np.in1d(train_set_to_choose, chosen)]
        chosen_bool = np.zeros(N_train_set, dtype=bool)
        chosen_bool[chosen] = True
        return chosen_bool, train_set

    while (eps_gpc > eps):
        
        # reset sub iteration if all polynomials of present order were added to gpc expansion
        if interaction_order_count > interaction_order_max:
            interaction_order_count = 0
            i_iter = i_iter + 1
            if print_out:    
                print("Iteration #{}".format(i_iter))
                print("=============\n")
           
        if i_iter == 1:
            # initialize gPC object in first iteration
            grid_init = randomgrid(pdftype, pdfshape, limits, np.ceil(data_poly_ratio*calc_Nc(order,DIM)))
            samples, train_set_to_choose = choose_from_train_set(np.ceil(data_poly_ratio*calc_Nc(order, DIM)))
            grid_init.coords = train_set_coords[samples, :]
            grid_init.coords_norm = norm(train_set_coords[samples, :], pdftype, pdfshape, limits)
            regobj = regularized(pdftype, pdfshape, limits, order*np.ones(DIM),
                                 order_max=order, interaction_order=DIM, grid=grid_init)
        else:
                       
            # generate new multi-indices if maximum gpc order increased
            if interaction_order_count == 0:
                order = order + 1
                if (order > order_end):
                    return regobj
                poly_idx_all_new = allVL1leq(DIM, order)
                poly_idx_all_new = poly_idx_all_new[np.sum(poly_idx_all_new,axis=1) == order]
                interaction_order_list = np.sum(poly_idx_all_new > 0,axis=1)
                interaction_order_max = np.max(interaction_order_list)
                interaction_order_count = 1

            if print_out:
                print("   Subiteration #{}".format(interaction_order_count))
                print("   =============\n")
            # filter out polynomials of interaction_order = interaction_order_count
            poly_idx_added = poly_idx_all_new[interaction_order_list==interaction_order_count,:]
            if data_poly_ratio * len(poly_idx_added) > N_train_set - regobj.grid.coords.shape[0]:
                break

            # add polynomials to gpc expansion
            regobj.enrich_polynomial_basis(poly_idx_added)

            # generate new grid-points
            regobj.enrich_gpc_matrix_samples(data_poly_ratio)
            samples, train_set_to_choose = choose_from_train_set(regobj.grid.coords.shape[0] - i_grid)
            regobj.grid.coords[i_grid:, :] = train_set_coords[samples, :]
            regobj.grid.coords_norm[i_grid:, :] =\
                norm(train_set_coords[samples, :], pdftype, pdfshape, limits)

            interaction_order_count = interaction_order_count + 1

        # run repeated simulations
        for i_grid, s in zip(range(i_grid, regobj.grid.coords.shape[0]),
                             np.where(samples)[0]):
            if print_out:
                print("   Performing simulation #{}\n".format(i_grid+1))
            # read conductivities from grid
            res = train_set_data[s, :]
            #print res, regobj.grid.coords[i_grid], train_set_coords[s]

            # append result to solution matrix (RHS)
            if i_grid == 0:
                RES = res
            else:
                RES = np.vstack([RES, res])

        # increase grid counter by one for next iteration (to not repeat last simulation)
        i_grid = i_grid + 1

        # perform leave one out cross validation
        regobj.construct_gpc_matrix()

        coeffs, eps_gpc = regobj.expand(RES, return_error=True, alphas=alphas)
        if record_loocv:
            start = time.time()
            error_dictionary['LOOCV'].append(regobj.LOOCV(RES))
            print("Time to perform LOOCV:", time.time() - start)

        error_dictionary['N_simus'].append(i_grid)
        error_dictionary['CV'].append(eps_gpc)


        if validation_set_data is not None:
            err = error_validation_set(regobj, coeffs, validation_set_data, validation_set_coords)

            error_dictionary['mean_rel_error'].append(err['mean_rel_error'])
            error_dictionary['max_rel_error'].append(err['max_rel_error'])

            error_dictionary['mean_abs_error'].append(err['mean_abs_error'])
            error_dictionary['max_abs_error'].append(err['max_abs_error'])

        if print_out:
            print("    -> relerror_CV = {}\n".format(eps_gpc))
            if validation_set_data is not None:
                print("    -> relerror_Validation = {}\n".format(err['mean_rel_error']))

    return regobj, RES, error_dictionary


class regularized(reg):
    def expand(self, data, return_error=False, cv=10, alphas=np.logspace(-5, 3, 9)):
        self.N_out = data.shape[1]
        if len(data) < cv:
            cv = len(data)
        coef = np.zeros((self.A.shape[1], self.N_out))

        def relative_error(y, y_pred):
            return np.average(np.linalg.norm(y - y_pred, axis=1) / np.linalg.norm(y, axis=1))

        cv_grid = model_selection.GridSearchCV(
            linear_model.Ridge(fit_intercept=False),
            {'alpha': alphas},
            cv=cv, iid=False,
            scoring=metrics.make_scorer(relative_error, greater_is_better=False),
            return_train_score=False)

        # How to get the CV (relative) error
        if data.shape[1] > data.shape[0]:
            U, S, Vt = np.linalg.svd(data.T,
                                     full_matrices=False)
            reduced = (S[:, None] * Vt).T
            cv_grid.fit(
                self.A, reduced)
            m = cv_grid.best_estimator_
            coef = U.dot(m.coef_).T

        else:
            cv_grid.fit(
                self.A, data)
            m = cv_grid.best_estimator_
            coef = m.coef_.T

        error = -cv_grid.best_score_
        print(cv_grid.best_params_)
        if return_error:
            return coef, error
        else:
            return coef


def _expand_polynomial(active_set, old_set, to_expand, order_max):
    ''' Algorithm by Gerstner and Griebel '''
    active_set = [tuple(a) for a in active_set]
    old_set = [tuple(o) for o in old_set]
    to_expand = tuple(to_expand)
    active_set.remove(to_expand)
    old_set += [to_expand]
    expand = []
    for e in range(len(to_expand)):
        forward = np.asarray(to_expand, dtype=int)
        forward[e] += 1
        has_predecessors = True
        for e2 in range(len(to_expand)):
            if forward[e2] > 0:
                predecessor = forward.copy()
                predecessor[e2] -= 1
                predecessor = tuple(predecessor)
                has_predecessors *= predecessor in old_set
        if has_predecessors and np.sum(np.abs(forward)) <= order_max:
            expand += [tuple(forward)]
            active_set += [tuple(forward)]

    return active_set, old_set, expand


def _choose_to_expand(reg, active_set, old_set, coeffs, order_max):
    coeffs = np.linalg.norm(coeffs, axis=1)
    for idx in reg.poly_idx[coeffs.argsort()[::-1]]:
        if tuple(idx) in active_set:
            _, _, expand = _expand_polynomial(active_set, old_set, tuple(idx), order_max)
            if len(expand) > 0:
                return tuple(idx)

    raise ValueError('Could not find a polynomial in the expansion and in the active set')

def run_reg_adaptive_grid_validation(
    pdftype, pdfshape, limits, train_set_data,
    train_set_coords,
    validation_set_data=None,
    validation_set_coords=None,
    eps=1E-3,
    data_poly_ratio=1.2,
    alphas=np.logspace(-5, 3, 9),
    order_max=7,
    record_loocv=False,
    print_out=False,
    return_coeffs=False):

    """ 
    Adaptive regression approach based on leave one out cross validation error
    estimation
    
    Parameters
    ----------

    pdftype : list
              Type of probability density functions of input parameters,
              i.e. ["beta", "norm",...]
    pdfshape : list of lists
               Shape parameters of probability density functions
               s1=[...] "beta": p, "norm": mean
               s2=[...] "beta": q, "norm": std
               pdfshape = [s1,s2]
    limits : list of lists
             Upper and lower bounds of random variables (only "beta")
             a=[...] "beta": lower bound, "norm": n/a define 0
             b=[...] "beta": upper bound, "norm": n/a define 0
             limits = [a,b]
    func : callable func(x,*args)
           The objective function to be minimized.
    args : tuple, optional
           Extra arguments passed to func, i.e. f(x,*args).         
    order_start : int, optional
                  Initial gpc expansion order (maximum order)
    order_end : int, optional
                Maximum gpc expansion order to expand to
    eps : float, optional
          Relative mean error of leave one out cross validation
    print_out : boolean, optional
          Print output of iterations and subiterations (True/False)      

    Returns
    -------
    gobj : object
           gpc object
    res  : ndarray
           Funtion values at grid points of the N_out output variables
           size: [N_grid x N_out]        
    """
    
    # initialize iterators
    eps_gpc = eps+1
    i_grid = 0
    i_iter = 0
    DIM = len(pdftype)
    active_set = [tuple(0 for d in range(DIM))]
    old_set = []
    to_expand = tuple(0 for d in range(DIM))

    error_dictionary = {}

    if validation_set_data is not None:
        error_dictionary['mean_rel_error'] = []
        error_dictionary['max_rel_error'] = []

        error_dictionary['mean_abs_error'] = []
        error_dictionary['max_abs_error'] = []
    error_dictionary['CV'] = []
    if record_loocv:
        error_dictionary['LOOCV'] = []
    error_dictionary['N_simus'] = []
    N_train_set = train_set_coords.shape[0]
    train_set_to_choose = np.arange(N_train_set, dtype=int)

    def choose_from_train_set(N):
        chosen = np.random.choice(train_set_to_choose, int(N), replace=False)
        train_set = train_set_to_choose[~np.in1d(train_set_to_choose, chosen)]
        chosen_bool = np.zeros(N_train_set, dtype=bool)
        chosen_bool[chosen] = True
        return chosen_bool, train_set

    while (eps_gpc > eps):
        i_iter = i_iter + 1
        if print_out:
            print("Iteration #{}".format(i_iter))
            print("=============\n")

        if i_iter == 1:
            grid_init = randomgrid(pdftype, pdfshape, limits,
                                   np.ceil(data_poly_ratio*calc_Nc(0, DIM)))
            samples, train_set_to_choose = \
                choose_from_train_set(np.ceil(data_poly_ratio*calc_Nc(0, DIM)))
            grid_init.coords = train_set_coords[samples, :]
            grid_init.coords_norm = norm(train_set_coords[samples, :], pdftype, pdfshape, limits)
            regobj = regularized(pdftype, pdfshape, limits, 0*np.ones(DIM),
                                 order_max=0, interaction_order=DIM, grid=grid_init)

        else:
            active_set, old_set, expand = _expand_polynomial(active_set, old_set,
                                                             to_expand, order_max)
            print('Active Set', active_set)
            print('Old Set', old_set)
            print('Expand', expand)
            if data_poly_ratio * len(expand) > len(train_set_to_choose):
                break

            # add polynomials to gpc expansion
            regobj.enrich_polynomial_basis(np.asarray(expand, dtype=int))
            # generate new grid-points
            regobj.enrich_gpc_matrix_samples(data_poly_ratio)
            samples, train_set_to_choose = choose_from_train_set(regobj.grid.coords.shape[0] - i_grid)
            regobj.grid.coords[i_grid:, :] = train_set_coords[samples, :]
            regobj.grid.coords_norm[i_grid:, :] =\
                norm(train_set_coords[samples, :], pdftype, pdfshape, limits)

        # run repeated simulations
        for i_grid, s in zip(range(i_grid, regobj.grid.coords.shape[0]),
                             np.where(samples)[0]):
            if print_out:
                print("   Performing simulation #{}\n".format(i_grid+1))
            # read conductivities from grid
            res = train_set_data[s, :]
            # append result to solution matrix (RHS)
            if i_grid == 0:
                RES = res
            else:
                RES = np.vstack([RES, res])

        # increase grid counter by one for next iteration (to not repeat last simulation)
        i_grid = i_grid + 1

        # perform leave one out cross validation
        regobj.construct_gpc_matrix()

        coeffs, eps_gpc = regobj.expand(RES, return_error=True, alphas=alphas)
        if record_loocv:
            start = time.time()
            error_dictionary['LOOCV'].append(regobj.LOOCV(RES))
            print("Time to perform LOOCV:", time.time() - start)

        error_dictionary['N_simus'].append(i_grid)
        error_dictionary['CV'].append(eps_gpc)

        if validation_set_data is not None:
            err = error_validation_set(regobj, coeffs, validation_set_data, validation_set_coords)

            error_dictionary['mean_rel_error'].append(err['mean_rel_error'])
            error_dictionary['max_rel_error'].append(err['max_rel_error'])

            error_dictionary['mean_abs_error'].append(err['mean_abs_error'])
            error_dictionary['max_abs_error'].append(err['max_abs_error'])

        if print_out:
            print("    -> relerror_CV = {}\n".format(eps_gpc))
            if validation_set_data is not None:
                print("    -> relerror_Validation = {}\n".format(err['mean_rel_error']))

        to_expand = _choose_to_expand(regobj, active_set, old_set, coeffs, order_max)

    if return_coeffs:
        return regobj, RES, error_dictionary, coeffs
    else:
        return regobj, RES, error_dictionary

