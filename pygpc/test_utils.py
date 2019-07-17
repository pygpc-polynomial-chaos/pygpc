import numpy as np
import h5py
from .io import read_gpc_pkl
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
    gpc = []
    error_msg = []
    file_status = True
    projection = False

    # get list of filenames of gPC .pkl files
    try:
        with h5py.File(fn_hdf5, "r") as f:
            try:
                fn_gpc_pkl = [fn.astype(str) for fn in list(f["misc/fn_gpc_pkl"][:].flatten())]
            except KeyError:
                error_msg.append("misc/fn_gpc_pkl not found in results file: {}".format(fn_hdf5))
                file_status = False

    except FileNotFoundError:
        error_msg.append("gPC results file not found: {}".format(fn_hdf5))
        file_status = False
        return file_status, error_msg

    # check for gPC object file
    ###########################
    for fn in fn_gpc_pkl:
        try:
            gpc.append(read_gpc_pkl(os.path.join(os.path.split(fn_hdf5)[0], fn)))
        except FileNotFoundError:
            error_msg.append("gPC object file not found: {}".format(fn))
            file_status = False
            return file_status, error_msg

    # check for gPC results file and kind of simulation
    ###################################################

    with h5py.File(fn_hdf5, "r") as f:
        try:
            if "qoi" in list(f["coeffs"].keys())[0]:
                qoi_keys = list(f["coeffs"].keys())
                qoi_idx = [int(key.split("qoi_")[1]) for key in qoi_keys]
                qoi_specific = True
        except AttributeError:
            qoi_keys = [""]
            qoi_specific = False

    for i_qoi in range(len(gpc)):
        if isinstance(gpc[i_qoi], MEGPC):
            multi_element_gpc = True
            dom_keys = ["dom_{}".format(int(i)) for i in range(len(np.unique(gpc[i_qoi].domains)))]

        else:
            multi_element_gpc = False
            dom_keys = [""]

    # check for projection approach
    if qoi_specific and multi_element_gpc:
        if str(type(gpc[0].gpc[0].p_matrix)) != "<class 'NoneType'>":
            projection = True
    elif not qoi_specific and not multi_element_gpc:
        if str(type(gpc[0].p_matrix)) != "<class 'NoneType'>":
            projection = True
    elif not qoi_specific and multi_element_gpc:
        if str(type(gpc[0].gpc[0].p_matrix)) != "<class 'NoneType'>":
            projection = True
    elif qoi_specific and not multi_element_gpc:
        if str(type(gpc[0].p_matrix)) != "<class 'NoneType'>":
            projection = True

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

        if gpc[0].gradient or projection:
            for target in ["grid/coords_gradient", "grid/coords_gradient_norm", "model_evaluations/gradient_results"]:
                try:
                    if type(f[target][()]) is not np.ndarray:
                        error_msg.append(target + " is not a numpy array")
                        file_status = False
                except KeyError:
                    error_msg.append(target + " not found in results file: {}".format(fn_hdf5))
                    file_status = False

        try:
            if type(f["misc/error_type"][()]) is not str:
                error_msg.append("misc/error_type is not a str")
                file_status = False
        except KeyError:
            error_msg.append("misc/error_type not found in results file: {}".format(fn_hdf5))
            file_status = False

    # check for specific .hdf5 file content
    #######################################
    with h5py.File(fn_hdf5, "r") as f:

        for i_qoi, q_idx in enumerate(qoi_keys):

            for target in ["domains"]:
                if not(target == "domains" and not multi_element_gpc):
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
                        tmp = f[h5_path][()]

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

                if gpc[0].gradient or projection:
                    for target in ["gpc_matrix_gradient"]:
                        try:
                            h5_path = target + "/" + q_idx + "/" + d_idx
                            if type(f[h5_path][()]) not in [np.ndarray]:
                                error_msg.append(h5_path + " is not a numpy array")
                                file_status = False
                        except KeyError:
                            error_msg.append(h5_path + " not found in results file: {}".format(fn_hdf5))
                            file_status = False

                if projection:
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
