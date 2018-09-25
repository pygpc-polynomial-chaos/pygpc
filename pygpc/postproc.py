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
    sobol_idx: [N_sobol] list of np.ndarray
        List of parameter label indices belonging to Sobol indices
    order: int, optional, default=1
        Sobol index order to extract

    Returns
    -------
    sobol_1st: [N_sobol x N_out] np.ndarray
        1st order Sobol indices of N_out output quantities

    sobol_idx_1st: [N_sobol] list of np.ndarray
        List of parameter label indices belonging to 1st order Sobol indices
    """

    # make mask of 1st order (linear) sobol indices
    mask = np.asarray([i for i in range(len(sobol_idx)) if sobol_idx[i].shape[0] == order])

    # extract from dataset
    # TODO: rethink indexing
    sobol_1st = sobol[mask, :]
    sobol_idx_1st = np.asarray([sobol_idx[mask[i]] for i in range(len(mask))])[:, 0]

    # sort sobol indices according to parameter indices in ascending order
    sort_idx = np.argsort(sobol_idx_1st)
    sobol_1st = sobol_1st[sort_idx, :]
    sobol_idx_1st = sobol_idx_1st[sort_idx]

    return sobol_1st, sobol_idx_1st
