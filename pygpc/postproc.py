# -*- coding: utf-8 -*-
"""
Functions that provide postprocessing implementations
"""

import numpy as np


def get_extracted_sobol_order(sobol, sobol_idx, order=1):
    """
    Extract Sobol indices with specified order from Sobol data.

    sobol_1st, sobol_idx_1st = extract_sobol_order(sobol, sobol_idx, order=1)

    Parameters
    ----------
    sobol: [N_sobol x N_out] np.ndarray
        Sobol indices of N_out output quantities
    sobol_idx: [N_sobol] list or np.ndarray of int
        list of parameter label indices belonging to Sobol indices
    order: int, optional, default=1
        Sobol index order to extract

    Returns
    -------
    sobol_n_order: np.ndarray
        n-th order Sobol indices of N_out output quantities

    sobol_idx_n_order: np.ndarray
        List of parameter label indices belonging to n-th order Sobol indices
    """

    # make mask of 1st order (linear) sobol indices
    mask = [index for index, sobol_element in enumerate(sobol_idx) if sobol_element.shape[0] == order]

    # extract from dataset
    sobol_n_order = sobol[mask, :]
    sobol_idx_n_order = np.vstack(sobol_idx[mask])

    # sort sobol indices according to parameter indices in ascending order
    sort_idx = np.argsort(sobol_idx_n_order, axis=0)[:, 0]
    sobol_n_order = sobol_n_order[sort_idx, :]
    sobol_idx_n_order = sobol_idx_n_order[sort_idx, :]

    return sobol_n_order, sobol_idx_n_order
