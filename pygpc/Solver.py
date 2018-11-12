import numpy as np


class Solver:
    """
    Solver class
    """

    def __init__(self, p):
        """
        Constructor; Initialize Solver class

        Parameters
        ----------
        p : dict
            Solver parameters
            - p["method"] ... Solver method (only applicable for Sgpc.Reg)
            - p["..."] ............

        Notes
        -----
        Moore-Penrose: Inverts gPC matrix optimally in least square sense gpc.A.shape[0] >= gpc.A.shape[1]

        OMP:
        """
        self.p = p

    # TODO: @Konstantin: implement here reg and quad solver to determine the gpc coefficients
    def __call__(self, gpc):
        """
        Determine gpc coefficients.

        Parameters
        ----------
        gpc : instance of gpc object
            Gpc input object under consideration.

        Returns
        -------
        coeffs : ndarray of float [N_basis x N_out]
            GPC coefficients for each output quantity
        """

