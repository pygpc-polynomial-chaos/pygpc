from .testfunctions.testfunctions import *
from .RandomParameter import *
from .Problem import *
from collections import OrderedDict

class Test(object):
    """
    Test function objects
    """
    def __init__(self):
        """
        Initializes Test function objects
        """
        pass

class Peaks(Test):
    """
    Peaks test function
    """
    def __init__(self):
        """
        Initializes Peaks test function
        """
        super(Peaks, self).__init__()

        # define model
        self.model = Peaks

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x3"] = 0.

        # define problem
        self.problem = Problem(self.model[-1], self.parameters[-1])

class HyperbolicTangent(Test):
    """
    HyperbolicTangent test function
    """
    def __init__(self):
        """
        Initializes HyperbolicTangent test function
        """
        super(HyperbolicTangent, self).__init__()

        # define model
        self.model = HyperbolicTangent

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model[-1], self.parameters[-1])

class MovingParticleFrictionForce(Test):
    """
    MovingParticleFrictionForce test function
    """
    def __init__(self):
        """
        Initializes MovingParticleFrictionForce test function
        """
        super(MovingParticleFrictionForce, self).__init__()

        # define model
        self.model = MovingParticleFrictionForce

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["xi"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model[-1], self.parameters[-1])

class SurfaceCoverageSpecies(Test):
    """
    HyperbolicTangent test function
    """
    def __init__(self):
        """
        Initializes SurfaceCoverageSpecies test function
        """
        super(SurfaceCoverageSpecies, self).__init__()

        # define model
        self.model = SurfaceCoverageSpecies

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["xi"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model[-1], self.parameters[-1])