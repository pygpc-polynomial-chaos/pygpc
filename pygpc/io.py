import os
import re
import sys
import h5py
import uuid
import pickle
import inspect
import logging
import numpy as np
from .misc import is_instance
from collections import OrderedDict
from importlib import import_module


def write_session(obj, fname, folder="session", overwrite=True):
    """
    Saves a gpc session in pickle or hdf5 file formal depending on the
    file extension in fname (.pkl or .hdf5)

    Parameters
    ----------
    obj : Session object
        Session class instance containing the gPC information
    fname : str
        Path to output file (.pkl or .hdf5)
    folder : str, optional, default: "session"
        Path in .hdf5 file (for .hdf5 format only)
    overwrite : bool, optional, default: True
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
        write_session_hdf5(obj, fname, folder, overwrite=overwrite)

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
    overwrite : bool, optional, default: True
        Overwrite existing file

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


def write_session_hdf5(obj, fname, folder="session", overwrite=True):
    """
    Write Session object including information about the Basis, Problem and Model as .hdf5 file.

    Parameters
    ----------
    obj : Session object
        Session class instance containing the gPC information
    fname : str
        Path to output file
    folder : str, optional, default: "session"
        Path in .hdf5 file
    overwrite : bool, optional, default: True
        Overwrite existing file

    Returns
    -------
    <file>: .hdf5 file
        .hdf5 file containing the gpc session
    """

    if overwrite and os.path.exists(fname):
        os.remove(fname)

    write_dict_to_hdf5(fn_hdf5=fname, data=obj.__dict__, folder=folder)


def read_session(fname, folder="session"):
    """
    Reads a gpc session in pickle or hdf5 file formal depending on the
    file extension in fname (.pkl or .hdf5)

    Parameters
    ----------
    fname : str
        path to input file
    folder : str, optional, default: "session"
        Path in .hdf5 file

    Returns
    -------
    obj : Session Object
        Session object containing instances of Basis, Problem and Model etc.
    """

    file_format = os.path.splitext(fname)[1]

    if file_format == ".pkl":
        obj = read_session_pkl(fname)

    elif file_format == ".hdf5":
        obj = read_session_hdf5(fname=fname, folder=folder)

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


def read_session_hdf5(fname, folder="session", verbose=False):
    """
    Read gPC object including information about input pdfs, polynomials, grid etc.

    object = read_gpc_obj(fname)

    Parameters
    ----------
    fname : str
        path to input file
    folder : str, optional, default: "session"
        Path in .hdf5 file
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    obj: GPC Object
        GPC object containing instances of Basis, Problem and Model.
    """
    from .Problem import Problem
    from .Session import Session
    from .RandomParameter import RandomParameter

    # model
    model = read_model_from_hdf5(fn_hdf5=fname, folder=folder + "/model", verbose=verbose)

    # parameters
    parameters_unsorted = read_parameters_from_hdf5(fn_hdf5=fname, folder=folder + "/problem/parameters",
                                                    verbose=verbose)

    problem_dict = read_group_from_hdf5(fn_hdf5=fname, folder=folder + "/problem", verbose=verbose)

    parameters = OrderedDict()
    parameters_random = OrderedDict()

    for p in problem_dict["parameters_keys"]:
        parameters[p] = parameters_unsorted[p]

        if isinstance(parameters_unsorted[p], RandomParameter):
            parameters_random[p] = parameters_unsorted[p]

    # problem(model, parameters)
    problem = Problem(model, parameters)

    # options
    options = read_group_from_hdf5(fn_hdf5=fname, folder=folder + "/algorithm/options")

    # validation
    try:
        validation = read_validation_from_hdf5(fn_hdf5=fname, folder=folder + "/validation", verbose=verbose)
    except KeyError:
        validation = None

    # grid
    grid = read_grid_from_hdf5(fn_hdf5=fname, folder=folder + "/grid", verbose=verbose)

    # algorithm
    module = import_module(".Algorithm", package="pygpc")
    algorithm_dict = read_group_from_hdf5(fn_hdf5=fname, folder=folder + "/algorithm", verbose=verbose)
    alg = getattr(module, algorithm_dict["attrs"]["dtype"].split(".")[-1])
    args = inspect.getfullargspec(alg).args[1:]

    args_dict = dict()
    args = [a for a in args if a != "gpc"]
    for a in args:
        args_dict[a] = locals()[a]

    algorithm = alg(**args_dict)

    # gpc
    gpc_raw_list = read_group_from_hdf5(fn_hdf5=fname, folder=folder + "/gpc", verbose=verbose)
    module = import_module(".Algorithm", package="pygpc")

    gpc_list = [0 for _ in range(len(gpc_raw_list))]
    for i_gpc, gpc_raw in enumerate(gpc_raw_list):

        # read and initialize classifier if present
        if "classifier" in gpc_raw.keys():
            classifier = read_classifier_from_hdf5(fn_hdf5=fname,
                                                   folder=folder + "/gpc/{}/classifier".format(i_gpc),
                                                   verbose=verbose)

        # read SGPC object if present (sub-gpc)
        if "gpc" in gpc_raw.keys():
            gpc = read_sgpc_from_hdf5(fn_hdf5=fname,
                                      folder=folder + "/gpc/{}/gpc".format(i_gpc),
                                      verbose=verbose)

        # get gpc class (SGPC or MEGPC)
        g = getattr(module, gpc_raw["attrs"]["dtype"].rsplit(".", 1)[1])
        del gpc_raw["attrs"]

        # SGPC
        if "SGPC" in g.__module__:
            gpc_list = read_sgpc_from_hdf5(fn_hdf5=fname,
                                           folder=folder + "/gpc/{}".format(i_gpc),
                                           verbose=verbose)
        # MEGPC with sub-gpcs
        else:
            # get input parameters of gpc
            args = inspect.getfullargspec(g).args[1:]

            args_dict = dict()
            for a in args:
                args_dict[a] = locals()[a]

            # initialize gpc
            gpc_list[i_gpc] = g(**args_dict)

            # loop over entries and save in self (if we have it in locals() we take this,
            # e.g. gpc, grid, validation etc)
            for key in gpc_raw:
                if key in locals():
                    setattr(gpc_list[i_gpc], key, locals()[key])
                else:
                    setattr(gpc_list[i_gpc], key, gpc_raw[key])

    # session(algorithm)
    session = Session(algorithm=algorithm)

    # read session hdf5 content
    session_dict = read_group_from_hdf5(fn_hdf5=fname, folder=folder, verbose=verbose)

    for key in session_dict:
        if key in locals():
            setattr(session, key, locals()[key])
        else:
            setattr(session, key, session_dict[key])

    # add path of .hdf5 script to python which generated the session (needed in case of relative imports)
    sys.path.append(os.path.split(session_dict["fn_script"])[0])

    # set gpc type in session
    session.set_gpc(gpc_list)

    return session


def read_problem_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads problem from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    problem : Problem object
        Problem
    """
    from .Problem import Problem

    # read content of problem
    problem_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)

    # model
    model = read_model_from_hdf5(fn_hdf5=fn_hdf5, folder=folder + "/model", verbose=verbose)

    # parameters
    parameters_unsorted = read_parameters_from_hdf5(fn_hdf5=fn_hdf5, folder=folder + "/parameters", verbose=False)

    # sort parameters
    parameters = OrderedDict()

    for p in problem_dict["parameters_keys"]:
        parameters[p] = parameters_unsorted[p]

    # initialize problem
    problem = Problem(model, parameters)

    return problem


def read_classifier_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads classifier from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    classifier : classifier object
        Classifier
    """
    classifier_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)
    module = import_module(".Classifier", package="pygpc")
    c = getattr(module, classifier_dict["attrs"]["dtype"].rsplit(".", 1)[1])

    # get input parameters of classifier
    args = inspect.getfullargspec(c).args[1:]

    args_dict = dict()
    for a in args:
        args_dict[a] = classifier_dict[a]

    init_classifier = True

    # for some reason the domains may be swapped in very rare cases so we do the init again
    while init_classifier:
        # initialize classifier
        classifier = c(**args_dict)

        # ensure that domains are not swapped
        classifier.domains = classifier_dict["domains"]
        classifier.update(coords=classifier_dict["coords"],
                          results=classifier_dict["results"])

        if np.sum(classifier.predict(coords=classifier_dict["coords"]) == classifier_dict["domains"])/ \
                len(classifier_dict["domains"]) > 0.95:
            init_classifier = False

    return classifier


def read_basis_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads Basis from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    basis : Basis object
        basis
    """

    basis_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)
    module_basis = import_module(".Basis", package="pygpc")
    module_basis_function = import_module(".BasisFunction", package="pygpc")
    b = getattr(module_basis, basis_dict["attrs"]["dtype"].rsplit(".", 1)[1])

    # get arguments to initialize basis
    args = inspect.getfullargspec(b).args[1:]

    # collect arguments from hdf5 file content
    args_dict = dict()
    for a in args:
        args_dict[a] = basis_dict[a]

    # initialize basis
    basis = b(**args_dict)

    # write content in self
    for key in basis_dict:
        if key != "b":
            setattr(basis, key,  basis_dict[key])

    b = [[0 for _ in range(basis_dict["dim"])] for _ in range(basis_dict["n_basis"])]
    for i_basis, b_lst in enumerate(basis_dict["b"]):
        for i_dim, b_ in enumerate(b_lst):
            # get type of basis function
            bf = getattr(module_basis_function, b_["attrs"]["dtype"].rsplit(".", 1)[1])

            # read content of hdf5
            bf_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5,
                                           folder=folder + "/b/{}/{}".format(i_basis, i_dim),
                                           verbose=verbose)

            # get arguments to initialize basis function
            args = inspect.getfullargspec(bf).args[1:]

            # collect arguments from hdf5 file content
            args_dict = dict()
            for a in args:
                args_dict[a] = bf_dict[a]

            # initialize basis function
            b[i_basis][i_dim] = bf(**args_dict)

    # extend basis
    basis.extend_basis(b)

    return basis


def read_sgpc_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads SGPC from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    sgpc : SGPC object or list of SGPC objects
        SGPC
    """

    sgpc_raw_list = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)
    module = import_module(".SGPC", package="pygpc")

    if type(sgpc_raw_list) is not list:
        sgpc_raw_list = [sgpc_raw_list]
        sub_gpc = False
    else:
        sub_gpc = True

    sgpc_list = [0 for _ in range(len(sgpc_raw_list))]

    for i_gpc, sgpc_raw in enumerate(sgpc_raw_list):
        if sub_gpc:
            hdf5_loc = "/{}/".format(i_gpc)
        else:
            hdf5_loc = "/"

        # get gpc by type
        g = getattr(module, sgpc_raw["attrs"]["dtype"].rsplit(".", 1)[1])

        # get input parameters of classifier
        args = inspect.getfullargspec(g).args[1:]

        args_dict = dict()
        for a in args:
            if a == "problem":
                args_dict[a] = read_problem_from_hdf5(fn_hdf5=fn_hdf5,
                                                      folder=folder + hdf5_loc + a,
                                                      verbose=verbose)
            elif a == "validation":
                args_dict[a] = read_validation_from_hdf5(fn_hdf5=fn_hdf5,
                                                         folder=folder + hdf5_loc + a,
                                                         verbose=verbose)
            else:
                args_dict[a] = sgpc_raw[a]

        # initialize SGPC object
        sgpc_list[i_gpc] = g(**args_dict)

        # write objects in self
        for key in sgpc_raw:
            try:
                dtype = sgpc_raw[key]["attrs"]["dtype"]

                if "pygpc.Basis" in dtype:
                    basis = read_basis_from_hdf5(fn_hdf5=fn_hdf5,
                                                 folder=folder + hdf5_loc + key,
                                                 verbose=verbose)
                    setattr(sgpc_list[i_gpc], key, basis)

                elif "pygpc.Grid" in dtype:
                    grid = read_grid_from_hdf5(fn_hdf5=fn_hdf5,
                                               folder=folder + hdf5_loc + key,
                                               verbose=verbose)
                    setattr(sgpc_list[i_gpc], key, grid)

                elif "pygpc.Problem" in dtype:
                    problem = read_problem_from_hdf5(fn_hdf5=fn_hdf5,
                                                     folder=folder + hdf5_loc + key,
                                                     verbose=verbose)
                    setattr(sgpc_list[i_gpc], key, problem)

                else:
                    setattr(sgpc_list[i_gpc], key, sgpc_raw[key])

            except (KeyError, IndexError, TypeError):
                setattr(sgpc_list[i_gpc], key, sgpc_raw[key])

    return sgpc_list


def read_model_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads model from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    model : Model object
        Model
    """

    model_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)
    sys.path.append(os.path.split(model_dict["fname"])[0])
    module = import_module(os.path.splitext(os.path.split(model_dict["fname"])[1])[0])
    model = getattr(module, model_dict["attrs"]["dtype"].rsplit(".", 1)[1])()

    return model


def read_parameters_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads parameters from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    parameters : OrderedDict
        OrdererDict containing the parameters (random and deterministic)
    """
    parameters = OrderedDict()
    parameters_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)
    module = import_module(".RandomParameter", package="pygpc")

    for p in parameters_dict:
        if (type(parameters_dict[p]) is dict or type(parameters_dict[p]) is OrderedDict) and \
                "RandomParameter" in parameters_dict[p]["attrs"]["dtype"]:
            rp = getattr(module, parameters_dict[p]["attrs"]["dtype"].split(".")[-1])
            args = inspect.getfullargspec(rp).args[1:]

            args_dict = dict()
            for a in args:
                args_dict[a] = parameters_dict[p][a]

            parameters[p] = rp(**args_dict)

        else:
            parameters[p] = parameters_dict[p]

    return parameters


def read_grid_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads and initializes grid from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    grid : Grid object
        Grid
    """

    grid_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder, verbose=verbose)

    module = import_module(".Grid", package="pygpc")
    g = getattr(module, grid_dict["attrs"]["dtype"].split(".")[-1])

    # get arguments of grid function
    args = inspect.getfullargspec(g).args[1:]

    # read content of hdf5 and relate to arguments
    args_dict = dict()
    for a in args:
        if a in ["coords", "coords_norm", "coords_gradient", "coords_gradient_norm", "weights"]:
            args_dict[a] = grid_dict["_" + a]
        elif a == "parameters_random":
            parameters_random = read_parameters_from_hdf5(fn_hdf5=fn_hdf5,
                                                          folder=folder + "/parameters_random",
                                                          verbose=verbose)
            args_dict[a] = parameters_random
        else:
            args_dict[a] = grid_dict[a]

    # regenerate unique grid IDs
    args_dict["coords_id"] = [uuid.uuid4() for _ in range(grid_dict["n_grid"])]

    if args_dict["coords_gradient"] is not None:
        args_dict["coords_gradient_id"] = args_dict["coords_id"]

    grid = g(**args_dict)

    return grid


def read_validation_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Reads and initializes ValidatioSet from hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    validation : ValidationSet object
        ValidationSet
    """
    from .ValidationSet import ValidationSet
    validation_dict = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=folder)

    if validation_dict is None:
        validation = None
    else:
        args_dict = dict()
        args = inspect.getfullargspec(ValidationSet).args[1:]

        for a in args:
            if a == "grid":
                args_dict[a] = read_grid_from_hdf5(fn_hdf5=fn_hdf5, folder=folder + "/grid", verbose=verbose)
            else:
                args_dict[a] = validation_dict[a]

        validation = ValidationSet(**args_dict)

    return validation


def read_group_from_hdf5(fn_hdf5, folder, verbose=False):
    """
    Read data from group (folder) in hdf5 file

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    data : dict or list or OrderedDict
        Folder content
    """
    f = h5py.File(fn_hdf5, "r")

    attrs = dict()
    for a in f[folder].attrs:
        attrs[a] = f[folder].attrs.__getitem__(a)

    data = dict()

    if isinstance(f[folder], h5py.Group) and len(f[folder].keys()) > 0:
        for key in f[folder].keys():
            if folder != "/":
                data["attrs"] = attrs
            data[key] = read_array_from_hdf5(fn_hdf5=fn_hdf5,
                                             arr_name=folder + "/" + key)

        if folder != "/":
            if data["attrs"]["dtype"] == "list":
                data = [data[key] for key in data if key != "attrs"]

            elif data["attrs"]["dtype"] == "dict":
                del data["attrs"]

            elif data["attrs"]["dtype"] == "collections.OrderedDict":
                data_ordered = OrderedDict()

                for key in data:
                    if key != "attrs":
                        data_ordered[key] = data[key]

                data = data_ordered

    else:
        data = None

    return data


def read_array_from_hdf5(fn_hdf5, arr_name, verbose=False):
    """

    Parameters
    ----------
    fn_hdf5 : str
        Filename of .hdf5 file to write in
    folder : str
        Folder inside .hdf5 file where dict is saved
    verbose : bool, optional, default: False
        Print output info

    Returns
    -------
    data

    attrs

    """
    f = h5py.File(fn_hdf5, "r")

    if isinstance(f[arr_name], h5py.Group):
        data = read_group_from_hdf5(fn_hdf5=fn_hdf5, folder=arr_name)

    else:
        data = f[arr_name][()]

    if type(data) == np.bytes_:
        data = str(data.astype(str))

    if type(data) == str and (data == "None" or data == "N/A"):
        data = None

    return data


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
    max_recursion_depth = 12

    # object (dict)
    if is_instance(data) and not isinstance(data, OrderedDict):

        t, dt = get_dtype(data)

        # do not save uuids in hdf5
        if dt == "uuid.UUID":
            return

        else:

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


def write_arr_to_hdf5(fn_hdf5, arr_name, data, overwrite_arr=True, verbose=False):
    """
    Takes an array and adds it to an .hdf5 file

    If data is list of dict, write_dict_to_hdf5() is called for each dict with adapted hdf5-folder name
    Otherwise, data is casted to np.ndarray and dtype of unicode data casted to '|S'.

    Parameters
    ----------
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
    max_recursion_depth = 12

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

        # do not save uuids in hdf5
        if dt == "uuid.UUID":
            return

        else:
            # create group and set type and dtype attributes
            with h5py.File(fn_hdf5, "a") as f:
                f.create_group(str(arr_name))
                f[str(arr_name)].attrs.__setitem__("type", t)
                f[str(arr_name)].attrs.__setitem__("dtype", dt)

            for idx, lst in enumerate(data):
                if len(arr_name.split("/")) >= max_recursion_depth:
                    lst = np.array("None")

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
            data = np.array(["None"])

        t, dt = get_dtype(data)

        # do not save uuids in hdf5
        if dt == "uuid.UUID":
            return

        else:
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
