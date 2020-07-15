import numpy as np
import pickle
import __main__ as main
from .io import write_session
from .MEGPC import *
from .SGPC import *
from .GPC import *


class Session(object):
    """
    GPC Session class

    Parameters
    ----------
    algorithm : Algorithm Object
        Algorithm object containing the Problem object, the Model object
    """

    def __init__(self, algorithm):
        """
        Constructor; Initializes a gPC Session
        """
        self.gpc = None
        self.grid = None
        self.model = None
        self.problem = None
        self.gpc_type = None
        self.gradient = None
        self.validation = None
        self.projection = None
        self.qoi_specific = None
        self.algorithm = algorithm
        self.model = self.algorithm.problem.model
        self.n_cpu = self.algorithm.options["n_cpu"]
        self.matlab_model = self.algorithm.options["matlab_model"]

        if self.algorithm.options["fn_results"] is not None:
            self.fn_results = os.path.splitext(self.algorithm.options["fn_results"])[0]
        else:
            self.fn_results = None

        if self.algorithm.options["fn_session"] is not None:
            self.fn_session = self.algorithm.options["fn_session"]
            self.fn_session_folder = self.algorithm.options["fn_session_folder"]
        else:
            self.fn_session = None
            self.fn_session_folder = None

        try:
            self.fn_script = main.__file__
        except AttributeError:
            self.fn_script = "jupter notebook"

        # safe the original problem and random parameters
        self.problem = self.algorithm.problem
        self.parameters_random = self.algorithm.problem.parameters_random

    def set_gpc(self, gpc):
        """
        Determine and set properties of gPC Object returned from algorithms

        Parameters
        ----------
        gpc : MEGPC or SGPC object or list of MEGPC or SGPC objects
            GPC objects
        """
        # if fn_results is not absolute, try if it is relative wrt the path of the executing script
        if self.fn_results is not None and os.path.isabs(self.fn_results):
            self.fn_results = os.path.join(os.path.split(self.fn_script)[0], self.fn_results)

        # determine qoi specificity from algorithm
        self.qoi_specific = self.algorithm.qoi_specific

        # # determine qoi specificity from coeffs structure in results file
        # with h5py.File(os.path.splitext(self.fn_results)[0] + ".hdf5", "r") as f:
        #     try:
        #         if type(f["coeffs"][()]) is np.ndarray:
        #             self.qoi_specific = False
        #     except AttributeError:
        #
        #         if np.array([True for s in list(f["coeffs/"]) if "qoi" in s]).any():
        #             self.qoi_specific = True
        #         else:
        #             self.qoi_specific = False

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
        else:
            self.projection = False

        self.gradient = self.gpc[0].gradient

    def run(self):
        """
        Runs the gPC session by calling the algorithm and saves the Session object
        in .hdf5 results file in the "session/" folder or as .pkl file
        """
        gpc, coeffs, results = self.algorithm.run()
        self.set_gpc(gpc)

        if type(coeffs) is list and not self.qoi_specific:
            coeffs = coeffs[0]

        if self.gpc[0].validation:
            self.validation = self.gpc[0].validation

        self.grid = self.gpc[-1].grid

        try:
            self.grid_original = self.gpc[-1].grid_original
        except AttributeError:
            pass

        if self.fn_session is not None:
            if os.path.splitext(self.fn_session)[1] == ".hdf5":
                write_session(self, fname=self.fn_session, folder=self.fn_session_folder, overwrite=False)
            else:
                write_session(self, fname=self.fn_session, overwrite=True)

        return self, coeffs, results
