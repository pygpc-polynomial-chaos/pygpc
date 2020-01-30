import os
import re
import h5py
import pickle
import logging
import numpy as np
from collections import OrderedDict
from .misc import is_instance


def write_session(obj, fname, overwrite=True):
    """
    Saves a gpc session in pickle or hdf5 file formal depending on the
    file extension in fname (.pkl or .hdf5)

    Parameters
    ----------
    obj: Session object
        Session class instance containing the gPC information
    fname: str
        Path to output file
    overwrite: bool, optional, default: True
        Overwrite existing file

    Returns
    -------
    <file>: .hdf5 or .pkl file
        .hdf5 or .pkl file containing the gpc session
    """

    file_format = os.path.splitext(fname)[1]

    if file_format == ".pkl":
        write_session_pkl(obj, fname, overwrite=overwrite)

    elif file_format == ".hdf5":
        write_session_hdf5(obj, fname, overwrite=overwrite)

    else:
        raise IOError("Session can only be saved in .pkl or .hdf5 format.")


def write_session_pkl(obj, fname, overwrite=True):
    """
    Write Session object including information about the Basis, Problem and Model as pickle file.

    Parameters
    ----------
    obj: GPC or derived class
        Class instance containing the gPC information
    fname: str
        Path to output file

    Returns
    -------
    <file>: .pkl file
        File containing the GPC object
    """
    if not overwrite and os.path.exists(fname):
        Warning("File already exists.")
    else:
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, -1)


def write_session_hdf5(obj, fname, overwrite=True):
    """
    Write Session object including information about the Basis, Problem and Model as .hdf5 file.

    Parameters
    ----------
    obj: Session object
        Session class instance containing the gPC information
    fname: str
        Path to output file
    overwrite: bool, optional, default: True
        Overwrite existing file

    Returns
    -------
    <file>: .hdf5 file
        .hdf5 file containing the gpc session
    """
    from .Algorithm import Algorithm
    from .AbstractModel import AbstractModel
    from .GPC import GPC
    from .MEGPC import MEGPC

    if not overwrite and os.path.exists(fname):
        raise FileExistsError

    if overwrite and os.path.exists(fname):
        os.remove(fname)

    write_dict_to_hdf5(fn_hdf5=fname, data=obj.__dict__, folder="")


def read_session(fname):
    """
    Reads a gpc session in pickle or hdf5 file formal depending on the
    file extension in fname (.pkl or .hdf5)

    Parameters
    ----------
    fname: str
        path to input file

    Returns
    -------
    obj: Session Object
        Session object containing instances of Basis, Problem and Model etc.
    """

    file_format = os.path.splitext(fname)[1]

    if file_format == ".pkl":
        obj = read_session_pkl(fname)

    elif file_format == ".hdf5":
        obj = read_session_hdf5(fname)

    else:
        raise IOError("Session can only be read from .pkl or .hdf5 files.")

    return obj


def read_session_pkl(fname):
    """
    Read Session object in pickle format.

    Parameters
    ----------
    fname: str
        path to input file

    Returns
    -------
    obj: Session Object
        Session object containing instances of Basis, Problem and Model etc.
    """

    with open(fname, 'rb') as f:
        obj = pickle.load(f)

    return obj


# TODO: implement method to read session from .hdf5 file
def read_session_hdf5(fname):
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

    return None


def write_dict_to_hdf5(fn_hdf5, data, folder, verbose=False):
    """
    Takes dict and passes its keys to write_arr_to_hdf5()

    fn_hdf5:folder/
                  |--key1
                  |--key2
                  |...

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    data : dict
        Dictionary to save in .hdf5 file
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info
    """
    max_recursion_depth = 6

    # object (dict)
    if is_instance(data) and not isinstance(data, OrderedDict):

        t, dt = get_dtype(data)

        # create group and set type and dtype attributes
        with h5py.File(fn_hdf5, "a") as f:
            f.create_group(str(folder))
            f[str(folder)].attrs.__setitem__("type", t)
            f[str(folder)].attrs.__setitem__("dtype", dt)

        # write content
        for key in data.__dict__:
            if len(folder.split("/")) >= max_recursion_depth:
                data.__dict__[key] = "None"

            write_arr_to_hdf5(fn_hdf5=fn_hdf5,
                              arr_name=folder+"/"+key,
                              data=data.__dict__[key],
                              verbose=verbose)

    # mappingproxy (can not be saved)
    elif str(type(data)) == "<class 'mappingproxy'>":
        data = "mappingproxy"
        write_arr_to_hdf5(fn_hdf5=fn_hdf5,
                          arr_name="mappingproxy",
                          data=data,
                          verbose=verbose)

    # list or tuple
    elif type(data) is list or type(data) is tuple:
        t, dt = get_dtype(data)

        # create group and set type and dtype attributes
        with h5py.File(fn_hdf5, "a") as f:
            f.create_group(str(folder))
            f[str(folder)].attrs.__setitem__("type", t)
            f[str(folder)].attrs.__setitem__("dtype", dt)

        for idx, lst in enumerate(data):
            if len(folder.split("/")) >= max_recursion_depth:
                lst = "None"

            write_arr_to_hdf5(fn_hdf5=fn_hdf5,
                              arr_name=folder+"/"+str(idx),
                              data=lst,
                              verbose=verbose)

    # dict or OrderedDict
    else:
        t, dt = get_dtype(data)

        # create group and set type and dtype attributes
        with h5py.File(fn_hdf5, "a") as f:
            try:
                f.create_group(str(folder))
                f[str(folder)].attrs.__setitem__("type", t)
                f[str(folder)].attrs.__setitem__("dtype", dt)
            except ValueError:
                pass

        for key in list(data.keys()):
            if len(folder.split("/")) >= max_recursion_depth:
                data[key] = "None"

            write_arr_to_hdf5(fn_hdf5=fn_hdf5,
                              arr_name=folder+"/"+str(key),
                              data=data[key],
                              verbose=verbose)


def write_arr_to_hdf5(fn_hdf5, arr_name, data, overwrite_arr=True,verbose=False):
    """
    Takes an array and adds it to an .hdf5 file

    If data is list of dict, write_dict_to_hdf5() is called for each dict with adapted hdf5-folder name
    Otherwise, data is casted to np.ndarray and dtype of unicode data casted to '|S'.

    Parameters:
    -----------
    fn_hdf5 : str
        Filename of .hdf5 file
    arr_name : str
        Complete path in .hdf5 file with array name
    data : ndarray, list or dict
        Data to write
    overwrite_arr : bool, optional, default: True
        Overwrite existing array
    verbose : bool, optional, default: False
        Print information
    """
    max_recursion_depth = 6

    # dict or OrderedDict
    if isinstance(data, dict) or isinstance(data, OrderedDict):
        if len(arr_name.split("/")) >= max_recursion_depth:
            data = np.array("None")
        else:
            write_dict_to_hdf5(fn_hdf5=fn_hdf5,
                               data=data,
                               folder=arr_name,
                               verbose=verbose)
            return

    # list of dictionaries:
    elif isinstance(data, list) and len(data) > 0 and (isinstance(data[0], dict) or is_instance(data[0])):
        t, dt = get_dtype(data)

        # create group and set type and dtype attributes
        with h5py.File(fn_hdf5, "a") as f:
            f.create_group(str(arr_name))
            f[str(arr_name)].attrs.__setitem__("type", t)
            f[str(arr_name)].attrs.__setitem__("dtype", dt)

        for idx, lst in enumerate(data):
            if len(arr_name.split("/")) >= max_recursion_depth:
                data = np.array("None")
            else:
                write_dict_to_hdf5(fn_hdf5=fn_hdf5,
                                   data=lst,
                                   folder=arr_name+"/"+str(idx),
                                   verbose=verbose)
                return

    # object
    elif is_instance(data):
        if len(arr_name.split("/")) >= max_recursion_depth:
            data = np.array("None")
        else:
            t, dt = get_dtype(data)

            # create group and set type and dtype attributes
            with h5py.File(fn_hdf5, "a") as f:
                f.create_group(str(arr_name))
                f[str(arr_name)].attrs.__setitem__("type", t)
                f[str(arr_name)].attrs.__setitem__("dtype", dt)

            write_dict_to_hdf5(fn_hdf5=fn_hdf5,
                               data=data.__dict__,
                               folder=arr_name,
                               verbose=verbose)
            return

    # list or tuple
    elif type(data) is list or type(data) is tuple:
        if len(arr_name.split("/")) >= max_recursion_depth:
            data = np.array("None")
        else:
            t, dt = get_dtype(data)

            # create group and set type and dtype attributes
            with h5py.File(fn_hdf5, "a") as f:
                f.create_group(str(arr_name))
                f[str(arr_name)].attrs.__setitem__("type", t)
                f[str(arr_name)].attrs.__setitem__("dtype", dt)

            data_dict = dict()
            for idx, lst in enumerate(data):
                data_dict[idx] = lst

            write_dict_to_hdf5(fn_hdf5=fn_hdf5,
                               data=data_dict,
                               folder=arr_name,
                               verbose=verbose)

        return

    elif not isinstance(data, np.ndarray):
        if len(arr_name.split("/")) >= max_recursion_depth:
            data = np.array("None")
        else:
            data = np.array(data)

    # np.arrays of np.arrays
    elif data.dtype == 'O' and len(data) > 1:
        if len(arr_name.split("/")) >= max_recursion_depth:
            return
        else:
            t, dt = get_dtype(data)

            # create group and set type and dtype attributes
            with h5py.File(fn_hdf5, "a") as f:
                f.create_group(str(arr_name))
                f[str(arr_name)].attrs.__setitem__("type", t)
                f[str(arr_name)].attrs.__setitem__("dtype", dt)

            data = data.tolist()
            write_dict_to_hdf5(fn_hdf5=fn_hdf5,
                               data=data,
                               folder=arr_name,
                               verbose=verbose)
            return


    # do some type casting from numpy/pd -> h5py
    # date column from experiment.csv is O
    # plotsetting["view"] is O list of list of different length
    # coil1 and coil2 columns names from experiment.csv is <U8
    # coil_mean column name from experiment.csv is <U12
    if data.dtype == 'O' or data.dtype.kind == 'U':
        data = data.astype('|S')

        if verbose:
            print("Converting array " + arr_name + " to string")

    t, dt = get_dtype(data)

    with h5py.File(fn_hdf5, 'a') as f:
        # create data_set
        if overwrite_arr:
            try:
                del f[arr_name]
            except KeyError:
                pass

        f.create_dataset(arr_name, data=data)
        f[str(arr_name)].attrs.__setitem__("type", t)
        f[str(arr_name)].attrs.__setitem__("dtype", dt)

    return


def get_dtype(obj):
    """
    Get type and datatype of object

    Parameters
    ----------
    obj : Object
        Input object (any)

    Returns
    -------
    type : str
        Type of object (e.g. 'class')
    dtype : str
        Datatype of object (e.g. 'numpy.ndarray')
    """
    type_str = str(type(obj))
    type_attr = re.match(pattern=r"\<(.*?)\ '", string=type_str).group(1)
    dtype_attr = re.findall(pattern=r"'(.*?)'", string=type_str)[0]

    return type_attr, dtype_attr


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
