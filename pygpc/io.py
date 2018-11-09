# -*- coding: utf-8 -*-
"""
Functions that provide input and output functionality
"""

import pickle
import h5py
import yaml
import logging
from builtins import range

from .Grid import *


def write_gpc_yml(obj, fname):
    """
    Write gPC infos about input pdfs, polynomials, grid etc. as .yml file.

    write_gpc_yml(obj, fname)

    Parameters
    ----------
    obj: gPC or derived class
        class instance containing gpc data
    fname: str
        path to output file
    """

    # fix extension
    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # write information to dictionary
    info = dict(random_vars=obj.random_vars,
                pdf_type=obj.pdftype.tolist(),
                pdf_shape=obj.pdfshape,
                limits=obj.limits,
                dim=obj.dim,
                order=obj.order.tolist(),
                order_max=obj.order_max,
                interaction_order=obj.interaction_order,
                grid_coords=obj.grid.coords.tolist(),
                grid_coords_norm=obj.grid.coords_norm.tolist(),
                gpc_type=obj.__class__.__name__,
                grid_type=obj.grid.__class__.__name__)

    # add grid specific attributes to dictionary
    if obj.grid.__class__.__name__ == 'RandomGrid':
        info['seed'] = obj.grid.seed

    elif obj.grid.__class__.__name__ == 'TensorGrid':
        info['N'] = obj.grid.N
        info['weights'] = obj.grid.weights.tolist()

    elif obj.grid.__class__.__name__ == 'SparseGrid':
        info['level'] = obj.grid.level
        info['level_max'] = obj.grid.level_max
        info['order_sequence_type'] = obj.grid.order_sequence_type
        info['weights'] = obj.grid.weights.tolist()
        info['level_sequence'] = obj.grid.level_sequence
        info['order_sequence'] = obj.grid.order_sequence
        info['l_level'] = obj.grid.l_level.tolist()

    else:
        raise NotImplementedError

    # write in file
    with open(fname, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


def read_gpc_yml(fname):
    """
    Read gPC infos about input pdfs, polynomials, grid etc. as .yml file and initialize gpc object.

    obj = read_gpc_yml(fname)

    Parameters
    ----------
    fname: str
        path to input file
    """

    # fix extension
    if fname[-4:] != 'yaml' and fname[-3:] != 'yml':
        fname += '.yaml'

    # read yml file
    with open(fname, 'r') as f:
        info = yaml.load(f)

    # initialize grid object
    if info['grid_type'] == 'RandomGrid':
        grid = RandomGrid(pdf_type=info['pdf_type'],
                          grid_shape=info['pdf_shape'],
                          limits=info['limits'],
                          N=0,
                          seed=info['seed'])

    elif info['grid_type'] == 'TensorGrid':
        grid = TensorGrid(pdf_type=info['pdf_type'],
                          grid_type=info['grid_type'],
                          grid_shape=info['pdf_shape'],
                          limits=info['limits'],
                          N=info['N'])
        grid.weights = np.asarray(info['weights'])

    elif info['grid_type'] == 'SparseGrid':
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
    obj = eval(info['gpc_type'])(random_vars=info['random_vars'],
                                 pdftype=info['pdftype'],
                                 pdfshape=info['pdfshape'],
                                 limits=info['limits'],
                                 order=info['order'],
                                 order_max=info['order_max'],
                                 interaction_order=info['interaction_order'],
                                 grid=grid)
    return obj


def write_gpc_pkl(obj, fname):
    """
    Write gPC object including infos about input pdfs, polynomials, grid etc. as pickle file.

    write_gpc_obj(obj, fname)

    Parameters
    ----------
    obj: gPC or derived class
        class instance containing gpc data
    fname: str
        path to output file
    """

    with open(fname, 'wb') as output:
        pickle.dump(obj, output, -1)


def read_gpc_pkl(fname):
    """
    Read gPC object including infos about input pdfs, polynomials, grid etc.

    object = read_gpc_obj(fname)

    Parameters
    ----------
    fname: str
        path to input file
    """

    with open(fname, 'rb') as f:
        return pickle.load(f)


def write_data_txt(data, fname):
    """
    Write data (quantity of interest) in .txt file (e.g. coeffs, mean, std, ...).

    write_data_txt(data, fname)

    Parameters
    ----------
    data: np.ndarray
        data to save
    fname: str
        path to output file
    """

    np.savetxt(fname, data, fmt='%.10e', delimiter='\t', newline='\n', header='', footer='')


def read_data_hdf5(fname, loc):
    """
    Read data from .hdf5 file (e.g. coeffs, mean, std, ...).

    load_data_hdf5(fname, loc)

    Parameters
    ----------
    fname: str
        path to input file
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

    Parameters
    ----------
    data: np.ndarray
        data to save
    fname: str
        path to output file
    loc: str
        location (folder and name) in hdf5 file (e.g. data/phi)
    """

    with h5py.File(fname, 'a') as f:
        f.create_dataset(loc, data=data)


def write_sobol_idx_txt(sobol_idx, fname):
    """
    Write sobol_idx list in file.

    write_sobol_idx_txt(sobol_idx, filename)

    Parameters
    ----------
    sobol_idx: [N_sobol] list of np.ndarray
        list of parameter label indices belonging to Sobol indices
    fname: str
        path to output file
    """

    f = open(fname, 'w')
    f.write('# Parameter index list of Sobol indices:\n')
    for line in sobol_idx:
        for entry in line:
            if entry != line[0]:
                f.write(', ')
            f.write('{}'.format(entry))
        if line != sobol_idx[-1]:
            f.write('\n')

    f.close()


def read_sobol_idx_txt(fname):
    """
    Read sobol_idx list from file.

    read_sobol_idx_txt(fname)

    Parameters
    ----------
    fname: str
        path to input file

    Returns
    -------
    sobol_idx: [N_sobol] list of np.array
        list of parameter label indices belonging to Sobol indices
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
            # read comma separated indices and convert to np.ndarray
            sobol_idx.append(np.asarray([int(x) for x in line.split(',') if x]))

        line = f.readline().strip('\n')

    return sobol_idx


def write_value(message, verbose=True):
    """
    Function that prints out a message over the python logging module

    iprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """
    console_logger.info(message)


def write_log_sobol(fname, sobol_rel_order_mean, sobol_rel_1st_order_mean, sobol_extracted_idx_1st):
    """
    Write average ratios of Sobol indices into logfile.

    Parameters
    ----------
    fname: str
        Path of logfile
    sobol_rel_order_mean: np.ndarray
        Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,).
        (over all output quantities)
    sobol_rel_1st_order_mean: np.ndarray
        Average proportion of the random variables of the 1st order Sobol indices to the total variance.
        (over all output quantities)
    sobol_extracted_idx_1st: list of int [N_sobol_1st]
        Indices of extracted 1st order Sobol indices corresponding to SGPC.random_vars.

    Returns
    -------
    <File>: .txt file
        Logfile containing information about the average ratios of 1st order Sobol indices w.r.t. the total variance
    """
    # start log
    log = open(os.path.splitext(fname)[0] + '.txt', 'w')
    log.write("Sobol indices:\n")
    log.write("==============\n")
    log.write("\n")

    # print order ratios
    log.write("Ratio: order / total variance over all output quantities:\n")
    log.write("---------------------------------------------------------\n")
    for i in range(len(sobol_rel_order_mean)):
        log.write("Order {}: {:.4f}\n".format(i + 1, sobol_rel_order_mean[i]))

    log.write("\n")

    # print 1st order ratios of parameters
    log.write("Ratio: 1st order Sobol indices of parameters / total variance over all output quantities\n")
    log.write("----------------------------------------------------------------------------------------\n")

    random_vars = []
    max_len = max([len(self.random_vars[i]) for i in range(len(self.random_vars))])
    for i in range(len(sobol_rel_1st_order_mean)):
        log.write("{}{:s}: {:.4f}\n".format(
            (max_len - len(self.random_vars[sobol_extracted_idx_1st[i]])) * ' ',
            self.random_vars[sobol_extracted_idx_1st[i]],
            sobol_rel_1st_order_mean[i]))
        random_vars.append(self.random_vars[sobol_extracted_idx_1st[i]])

    log.close()


# initialize logger
file_logger = logging.getLogger('gPC')
file_logger.setLevel(logging.DEBUG)
file_logger_handler = logging.FileHandler('gPC.log')
file_logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
file_logger_handler.setFormatter(file_logger_formatter)
file_logger.addHandler(file_logger_handler)

console_logger = logging.getLogger('gPC_console_output')
console_logger.setLevel(logging.DEBUG)
console_logger_handler = logging.StreamHandler()
console_logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_logger_handler.setFormatter(console_logger_formatter)
console_logger.addHandler(console_logger_handler)

file_logger.disabled = False
console_logger.disabled = False


def activate_terminal_output():
    console_logger.disabled = False


def activate_logfile_output():
    file_logger.disabled = False


def deactivate_terminal_output():
    console_logger.disabled = True


def deactivate_logfile_output():
    file_logger.disabled = True


def iprint(message, verbose=True, tab=None):
    """
    Function that prints out a message over the python logging module

    iprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """

    if tab:
        message = '\t' * tab + message
    console_logger.info(message)


def wprint(message, verbose=True, tab=None):
    """
    Function that prints out a warning message over the python logging module

    wprint(message, verbose=True)

    Parameters
    ----------
    message: string
        string to print in standard output
    verbose: bool, optional, default=True
        determines if string is printed out
    """

    if tab:
        message = '\t' * tab + message
    console_logger.warning(message)