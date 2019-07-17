# -*- coding: utf-8 -*-
# from .Grid import *
import numpy as np
import os
import pickle
import h5py
import copy
import logging


def write_gpc_pkl(obj, fname):
    """
    Write gPC object including information about the Basis, Problem and Model as pickle file.

    write_gpc_obj(obj, fname)

    Parameters
    ----------
    obj: GPC or derived class
        Class instance containing the gPC information
    fname: str
        path to output file

    Returns
    -------
    <file>: .pkl file
        File containing the GPC object
    """

    # write .gpc object
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, -1)


def read_gpc_pkl(fname):
    """
    Read gPC object including information about input pdfs, polynomials, grid etc.

    object = read_gpc_obj(fname)

    Parameters
    ----------
    fname: str
        path to input file

    Returns
    -------
    obj: GPC Object
        GPC object containing instances of Basis, Problem and Model.
    """

    with open(fname, 'rb') as f:
        obj = pickle.load(f)

    return obj


def write_data_txt(data, fname):
    """
    Write data (quantity of interest) in .txt file (e.g. coeffs, mean, std, ...).

    write_data_txt(data, fname)

    Parameters
    ----------
    data: ndarray of float
        Data to save
    fname: str
        Path to output file

    Returns
    -------
    <file>: .txt file
        File containing the data (tab delimited)
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

    Returns
    -------
    data: ndarray of float
        Loaded data from .hdf5 file
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
        List of parameter label indices belonging to Sobol indices
    fname: str
        Path to output file

    Returns
    -------
    <file>: .txt file
        File containing the sobol index list.
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
        Path to input file

    Returns
    -------
    sobol_idx: [N_sobol] list of np.array
        List of parameter label indices belonging to Sobol indices
    """

    f = open(fname, 'r')

    line = f.readline().strip('\n')
    sobol_idx = []

    while line:

        # ignore comments in text file
        if line[0] == '#':
            line = f.readline().strip('\n')
            continue

        else:
            # read comma separated indices and convert to ndarray
            sobol_idx.append(np.asarray([int(x) for x in line.split(',') if x]))

        line = f.readline().strip('\n')

    return sobol_idx


def write_log_sobol(fname, random_vars, sobol_rel_order_mean, sobol_rel_1st_order_mean, sobol_extracted_idx_1st):
    """
    Write average ratios of Sobol indices into logfile.

    Parameters
    ----------
    fname: str
        Path of logfile
    random_vars: list of str
        Labels of random variables
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

    # random_vars = []
    max_len = max([len(random_vars[i]) for i in range(len(random_vars))])
    for i in range(len(sobol_rel_1st_order_mean)):
        log.write("{}{:s}: {:.4f}\n".format(
            (max_len - len(random_vars[sobol_extracted_idx_1st[i]])) * ' ',
            random_vars[sobol_extracted_idx_1st[i]],
            sobol_rel_1st_order_mean[i]))
        # random_vars.append(self.random_vars[sobol_extracted_idx_1st[i]])

    log.close()


# # initialize logger
# file_logger = logging.getLogger('gPC')
# file_logger.setLevel(logging.DEBUG)
# file_logger_handler = logging.FileHandler('gPC.log')
# file_logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
# file_logger_handler.setFormatter(file_logger_formatter)
# file_logger.addHandler(file_logger_handler)

console_logger = logging.getLogger('gPC_console_output')
console_logger.setLevel(logging.DEBUG)
console_logger_handler = logging.StreamHandler()
console_logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_logger_handler.setFormatter(console_logger_formatter)
console_logger.addHandler(console_logger_handler)

# file_logger.disabled = False
console_logger.disabled = False


# def activate_terminal_output():
#     console_logger.disabled = False
#
#
# def activate_logfile_output():
#     file_logger.disabled = False
#
#
# def deactivate_terminal_output():
#     console_logger.disabled = True
#
#
# def deactivate_logfile_output():
#     file_logger.disabled = True


def iprint(message, verbose=True, tab=None):
    """
    Function that prints out a message over the python logging module

    iprint(message, verbose=True)

    Parameters
    ----------
    message: string
        String to print in standard output
    verbose: bool, optional, default=True
        Determines if string is printed out
    tab: int
        Number of tabs before message
    """
    if verbose:
        if tab:
            message = '\t' * tab + message
        # console_logger.info(message)
        print(message)


def wprint(message, verbose=True, tab=None):
    """
    Function that prints out a warning message over the python logging module

    wprint(message, verbose=True)

    Parameters
    ----------
    message: string
        String to print in standard output
    verbose: bool, optional, default=True
        Determines if string is printed out
    tab: int
        Number of tabs before message
    """

    if verbose:
        if tab:
            message = '\t' * tab + message
        console_logger.warning(message)
