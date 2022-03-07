import numpy as np
import h5py
from .io import read_session
from .MEGPC import *


def check_file_consistency(fn_hdf5):
    """
    Test gPC output files for consistency.

    Parameters
    ----------
    fn_hdf5 : str
        Filename of gPC results (.hdf5) file

    Returns
    -------
    file_status : boolean
        File consistency
    error_msg : list of str
        Error messages if files are not consistent
    """
    error_msg = []
    file_status = True

    # get list of filenames of gPC .pkl files
    try:
        with h5py.File(fn_hdf5, "r") as f:
            try:
                fn_session = os.path.join(os.path.split(fn_hdf5)[0], f["misc/fn_session"][0].astype(str))
                fn_session_folder = f["misc/fn_session_folder"][0].astype(str)
            except KeyError:
                error_msg.append("misc/fn_gpc_pkl not found in results file: {}".format(fn_hdf5))
                file_status = False

    except FileNotFoundError:
        error_msg.append("gPC results file not found: {}".format(fn_hdf5))
        file_status = False
        return file_status, error_msg

    # check for gPC object file
    ###########################
    try:
        session = read_session(fname=fn_session, folder=fn_session_folder)
    except FileNotFoundError:
        error_msg.append("gPC session not found in file: {}".format(fn_session))
        file_status = False
        return file_status, error_msg

    # check for gPC results file and kind of simulation
    ###################################################

    with h5py.File(fn_hdf5, "r") as f:
        qoi_keys = [""]
        try:
            if isinstance(f["coeffs/"], h5py.Group):
                if np.array(["qoi" in s for s in list(f["coeffs/"].keys())]).any():
                    qoi_keys = list(f["coeffs"].keys())
                    qoi_idx = [int(key.split("qoi_")[1]) for key in qoi_keys]
            else:
                qoi_keys = [""]
        except KeyError:
            pass

    if session.gpc_type == "megpc":
        dom_keys = ["dom_{}".format(int(i)) for i in range(len(np.unique(session.gpc[0].domains)))]

    else:
        dom_keys = [""]

    # check for general .hdf5 file content
    ######################################
    with h5py.File(fn_hdf5, "r") as f:
        for target in ["grid/coords", "grid/coords_norm", "model_evaluations/results"]:
            try:
                if type(f[target][:]) is not np.ndarray:
                    error_msg.append(target + " is not a numpy array")
                    file_status = False
            except KeyError:
                error_msg.append(target + " not found in results file: {}".format(fn_hdf5))
                file_status = False

        if session.projection or (session.gradient and session.algorithm.options["gradient_calculation"] == "FD_fwd"):
            for target in ["grid/coords_gradient", "grid/coords_gradient_norm", "model_evaluations/gradient_results"]:
                try:
                    if type(f[target][()]) is not np.ndarray:
                        error_msg.append(target + " is not a numpy array")
                        file_status = False
                except KeyError:
                    error_msg.append(target + " not found in results file: {}".format(fn_hdf5))
                    file_status = False

        try:
            if not(type(f["misc/error_type"][...][()]) is str or type(f["misc/error_type"][...][()]) is bytes):
                error_msg.append("misc/error_type is not str or bytes")
                file_status = False
        except KeyError:
            error_msg.append("misc/error_type not found in results file: {}".format(fn_hdf5))
            file_status = False

    # check for specific .hdf5 file content
    #######################################
    with h5py.File(fn_hdf5, "r") as f:

        for i_qoi, q_idx in enumerate(qoi_keys):

            for target in ["domains"]:
                if not(target == "domains" and not session.gpc_type == "megpc"):
                    try:
                        h5_path = target + "/" + q_idx
                        if type(f[h5_path][()]) not in [np.ndarray, np.float64]:
                            error_msg.append(h5_path + " is not a numpy array")
                            file_status = False

                    except KeyError:
                        error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                        file_status = False

            for d_idx in dom_keys:

                # error can be domain specific or not in case of ME gPC (depending on the algorithm)
                for target in ["error"]:
                    try:
                        h5_path = target + "/" + q_idx + "/" + d_idx
                        tmp = f[h5_path][...]

                    except KeyError:
                        try:
                            h5_path = target + "/" + q_idx

                            if type(f[h5_path][()]) not in [np.ndarray, np.float64]:
                                error_msg.append(h5_path + " is not a numpy array")
                                file_status = False

                        except KeyError:
                            error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                            file_status = False

                for target in ["coeffs", "gpc_matrix"]:
                    try:
                        h5_path = target + "/" + q_idx + "/" + d_idx
                        if type(f[h5_path][()]) not in [np.ndarray]:
                            error_msg.append(h5_path + " is not a numpy array")
                            file_status = False

                    except KeyError:
                        error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                        file_status = False

                if session.gradient and session.algorithm.options["gradient_calculation"] == "FD_fwd":
                    for target in ["gpc_matrix_gradient"]:
                        try:
                            h5_path = target + "/" + q_idx + "/" + d_idx
                            if type(f[h5_path][()]) not in [np.ndarray]:
                                error_msg.append(h5_path + " is not a numpy array")
                                file_status = False
                        except KeyError:
                            error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                            file_status = False

                if session.projection:
                    for target in ["p_matrix"]:
                        try:
                            h5_path = target + "/" + q_idx + "/" + d_idx
                            if type(f[h5_path][()]) not in [np.ndarray]:
                                error_msg.append(h5_path + " is not a numpy array")
                                file_status = False
                        except KeyError:
                            error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                            file_status = False

    return file_status, error_msg
