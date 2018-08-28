# -*- coding: utf-8 -*-
"""
Functions that provide input and output functionality
"""

import pickle
import h5py
import yaml

from .grid import *


def write_gpc_yml(obj, fname):
    """
    Write gPC infos about input pdfs, polynomials, grid etc. as .yml file.

    write_gpc_yml(obj, fname)

    Parameters:
    ---------------------------
    obj: object
        gpc object including infos to save
    fname: str
        filename
    """

    # fix extension
    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # write information to dictionary
    info = dict(random_vars=obj.random_vars,
                pdf_type=obj.pdftype.tolist(),
                pdf_shape=obj.pdfshape,
                limits=obj.limits,
                order=obj.order.tolist(),
                order_max=obj.order_max,
                interaction_order=obj.interaction_order,
                grid_coords=obj.grid.coords.tolist(),
                grid_coords_norm=obj.grid.coords_norm.tolist(),
                gpc_kind=obj.__class__.__name__,
                grid_kind=obj.grid.__class__.__name__)

    # add grid specific attributes to dictionary
    if obj.grid.__class__.__name__ == 'randomgrid':
        info['seed'] = obj.grid.seed

    if obj.grid.__class__.__name__ == 'tensgrid':
        info['gridtype'] = obj.grid.gridtype
        info['N'] = obj.grid.N
        info['weights'] = obj.grid.weights.tolist()

    if obj.grid.__class__.__name__ == 'sparsegrid':
        info['gridtype'] = obj.grid.gridtype
        info['level'] = obj.grid.level
        info['level_max'] = obj.grid.level_max
        info['order_sequence_type'] = obj.grid.order_sequence_type
        info['weights'] = obj.grid.weights.tolist()
        info['level_sequence'] = obj.grid.level_sequence
        info['order_sequence'] = obj.grid.order_sequence
        info['l_level'] = obj.grid.l_level.tolist()

    # write in file
    with open(fname, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


def read_gpc_yml(fname):
    """
    Read gPC infos about input pdfs, polynomials, grid etc. as .yml file and initialize gpc object.

    obj = read_gpc_yml(fname)

    Parameters:
    ---------------------------
    fname: str
        filename
    """

    # fix extension
    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # read yml file
    with open(fname, 'r') as f:
        info = yaml.load(f)

    # initialize grid object
    if info['grid_kind'] == 'randomgrid':
        grid = RandomGrid(pdf_type=info['pdftype'],
                          grid_shape=info['pdfshape'],
                          limits=info['limits'],
                          N=0,
                          seed=info['seed'])

    elif info['grid_kind'] == 'tensgrid':
        grid = TensGrid(pdf_type=info['pdftype'],
                        grid_type=info['gridtype'],
                        grid_shape=info['pdfshape'],
                        limits=info['limits'],
                        N=info['N'])
        grid.weights = np.asarray(info['weights'])

    elif info['grid_kind'] == 'sparsegrid':
        grid = SparseGrid(pdf_type=info['pdftype'],
                          grid_type=info['gridtype'],
                          grid_shape=info['pdfshape'],
                          limits=info['limits'],
                          level=info['level'],
                          level_max=info['level_max'],
                          interaction_order=info['interaction_order'],
                          order_sequence_type=info['order_sequence_type'],
                          make_grid=False)
        grid.weights = np.asarray(info['weights'])
        grid.level_sequence = info['level_sequence']
        grid.order_sequence = info['order_sequence']
        grid.l_level = np.asarray(info['l_level'])
    else:
        raise NotImplementedError

    grid.coords = np.asarray(info['grid_coords'])
    grid.coords_norm = np.asarray(info['grid_coords_norm'])

    # initialize gpc object
    obj = eval(info['gpc_kind'])(random_vars=info['random_vars'],
                                 pdftype=info['pdftype'],
                                 pdfshape=info['pdfshape'],
                                 limits=info['limits'],
                                 order=info['order'],
                                 order_max=info['order_max'],
                                 interaction_order=info['interaction_order'],
                                 grid=grid)
    return obj


def write_gpc_obj(obj, fname):
    """
    Write gPC object including infos about input pdfs, polynomials, grid etc. as pickle file.

    write_gpc_obj(obj, fname)

    Parameters:
    ---------------------------
    obj: object
        gpc object to save
    fname: str
        filename with .pkl extension
    """

    with open(fname, 'wb') as output:
        pickle.dump(obj, output, -1)


def read_gpc_obj(fname):
    """
    Read gPC object including infos about input pdfs, polynomials, grid etc.

    object = read_gpc_obj(fname)

    Parameters:
    ---------------------------
    fname: str
        filename with .pkl extension
    """

    with open(fname, 'rb') as f:
        return pickle.load(f)


def write_data_txt(data, fname):
    """
    Write data (quantity of interest) in .txt file (e.g. coeffs, mean, std, ...).

    write_data_txt(qoi, filename)

    Parameters:
    ---------------------------
    data: 2D np.array()
        data to save
    fname: str
        filename with .hdf5 extension
    """

    np.savetxt(fname, data, fmt='%.10e', delimiter='\t', newline='\n', header='', footer='')


def read_data_hdf5(fname, loc):
    """
    Read data from .hdf5 file (e.g. coeffs, mean, std, ...).

    load_data_hdf5(fname, loc)

    Parameters:
    ---------------------------
    fname: str
        filename with .hdf5 extension
    loc: str
        location (folder and name) in hdf5 file (e.g. data/phi)
    """

    with h5py.File(fname, 'r') as f:
        d = f[loc]
        return d


def write_data_hdf5(data, fname, loc):
    """
    Write quantity of interest in .hdf5 file (e.g. coeffs, mean, std, ...).

    write_data_hdf5(data, fname, loc)

    Parameters:
    ---------------------------
    data: 2D np.array()
        data to save
    fname: str
        filename with .hdf5 extension
    loc: str
        location (folder and name) in hdf5 file (e.g. data/phi)
    """

    with h5py.File(fname, 'a') as f:
        f.create_dataset(loc, data=data)


def write_sobol_idx_txt(sobol_idx, fname):
    """
    Write sobol_idx list in file.

    write_sobol_idx_txt(sobol_idx, filename)

    Parameters:
    ---------------------------
    sobol_idx: list of np.array [N_sobol]
        List of parameter label indices belonging to Sobol indices
    fname: str
        filename with .txt extension containing the saved sobol indices
    """

    f = open(fname, 'w')
    f.write('# Parameter index list of Sobol indices:\n')
    for i_line in range(len(sobol_idx)):
        for i_entry in range(len(sobol_idx[i_line])):
            if i_entry > 0:
                f.write(', ')
            f.write('{}'.format(sobol_idx[i_line][i_entry]))
        if i_line < list(range(len(sobol_idx))):
            f.write('\n')

    f.close()


def read_sobol_idx_txt(fname):
    """
    Read sobol_idx list from file.

    read_sobol_idx_txt(filename)

    Parameters:
    ---------------------------
    fname: str
        filename with .txt extension containing the saved sobol indices

    Returns:
    ---------------------------
    sobol_idx: list of np.array [N_sobol]
        List of parameter label indices belonging to Sobol indices
    """

    f = open(fname, 'r')

    line = f.readline().strip('\n')
    sobol_idx = []

    while line:

        # ignore comments in textfile
        if line[0] == '#':
            line = f.readline().strip('\n')
            continue

        else:
            # read comma separated indices and convert to nparray
            sobol_idx.append(np.array([int(x) for x in line.split(',') if x]))

        line = f.readline().strip('\n')

    return sobol_idx
