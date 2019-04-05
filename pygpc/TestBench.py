from .testfunctions.testfunctions import *
from .RandomParameter import *
from .Problem import *


class TestBench(object):
    """
    TestBench for gPC algorithms
    """

    def __init__(self, algorithm_label, tests):
        """
        Initializes the Testbench class object instance

        Parameters
        ----------
        algorithm_label : str
            Algorithm to benchmark
        test_labels : test objects
            Tests to perform ["2D", "ND", "discontinuous"]
        """
        self.algorithm_label = algorithm_label
        self.test = []

        # create tests
        for t in tests:
            self.test.append(t(t))


class Test(object):
    """
    Test for gPC algorithms
    """
    def __init__(self):
        """
        Initializes Test class
        """
        pass


class Test2D(Test):
    """
    Test including 2D testfunctions
    """
    def __init__(self):
        """
        Initializes Test2D class
        """
        super(Test2D, self).__init__()

        self.model = []
        self.parameters = []
        self.problem = []

        # Peaks
        #######################################################################
        # define model
        self.model.append(Peaks)

        # define parameters
        self.parameters.append(OrderedDict())
        self.parameters[-1]["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters[-1]["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters[-1]["x3"] = 0.

        # define problem
        self.problem.append(Problem(self.model[-1], self.parameters[-1]))


class TestDiscontinuous(Test):
    """
    Test including discontinuous testfunctions
    """
    def __init__(self):
        """
        Initializes TestDiscontinuous class
        """
        super(TestDiscontinuous, self).__init__()

        self.model = []
        self.parameters = []
        self.problem = []

        # HyperbolicTangent
        #######################################################################
        # define model
        self.model.append(HyperbolicTangent)

        # define parameters
        self.parameters.append(OrderedDict())
        self.parameters[-1]["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])
        self.parameters[-1]["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem.append(Problem(self.model[-1], self.parameters[-1]))

        # MovingParticleFrictionForce
        #######################################################################
        # define model
        self.model.append(MovingParticleFrictionForce)

        # define parameters
        self.parameters.append(OrderedDict())
        self.parameters[-1]["xi"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem.append(Problem(self.model[-1], self.parameters[-1]))


class TestND(Test):
    """
    Test including 2D testfunctions
    """
    def __init__(self):
        """
        Initializes Test2D class
        """
        super(TestND, self).__init__()

        self.model = []
        self.parameters = []
        self.problem = []
