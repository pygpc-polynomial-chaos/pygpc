import numpy as np
from .MEGPC import *
from .SGPC import *
from .GPC import *
import pickle
from .io import write_session_pkl


class Session(object):
    """
    GPC Session class
    """

    def __init__(self, algorithm):
        """
        Constructor; Initializes a gPC Session

        Parameters
        ----------
        algorithm : Algorithm Object
            Algorithm object containing the Problem object, the Model object
        """
        self.gpc = None
        self.qoi_specific = None
        self.gpc_type = None
        self.problem = None
        self.model = None
        self.algorithm = algorithm
        self.model = self.algorithm.problem.model
        self.matlab_model = self.algorithm.options["matlab_model"]
        self.fn_results = os.path.splitext(self.algorithm.options["fn_results"])[0]
        self.validation = None
        self.projection = None
        self.gradient = None
        self.n_cpu = self.algorithm.options["n_cpu"]

        # safe the original problem and random parameters
        self.problem = self.algorithm.problem
        self.parameters_random = self.algorithm.problem.parameters_random

    def set_gpc(self, gpc):
        """ Get properties of gPC Object """
        with h5py.File(os.path.splitext(self.fn_results)[0] + ".hdf5", "r") as f:
            try:
                if type(f["coeffs"][()]) is np.ndarray:
                    self.qoi_specific = False
            except AttributeError:

                if np.array([True for s in list(f["coeffs/"]) if "qoi" in s]).any():
                    self.qoi_specific = True
                else:
                    self.qoi_specific = False

        if type(gpc) is list:
            self.gpc = gpc
        else:
            self.gpc = [gpc]

        if isinstance(self.gpc[0], MEGPC):
            self.gpc_type = "megpc"
        else:
            self.gpc_type = "sgpc"

        # check for projection approach
        if (self.qoi_specific and self.gpc_type == "megpc") or \
                (not self.qoi_specific and self.gpc_type == "megpc"):
            if str(type(self.gpc[0].gpc[0].p_matrix)) != "<class 'NoneType'>":
                self.projection = True

        elif (not self.qoi_specific and not self.gpc_type == "megpc") or \
                (self.qoi_specific and not self.gpc_type == "megpc"):
            if str(type(self.gpc[0].p_matrix)) != "<class 'NoneType'>":
                self.projection = True
        else:
            self.projection = False

        self.gradient = self.gpc[0].gradient

    def run(self):
        """
        Runs the gPC session by calling the algorithm and saves the Session object
        """
        gpc, coeffs, results = self.algorithm.run()
        self.set_gpc(gpc)

        if type(coeffs) is list and not self.qoi_specific:
            coeffs = coeffs[0]

        if self.gpc[0].validation:
            self.validation = self.gpc[0].validation

        write_session_pkl(self, self.fn_results + ".pkl")

        return self, coeffs, results

