# -*- coding: utf-8 -*-
from .testfunctions import *
from .RandomParameter import *
from .Problem import *
from collections import OrderedDict


class Test(object):
    """
    Test function objects
    """
    def __init__(self, dim):
        """
        Initializes Test function objects
        """
        self.dim = dim
        self.gpc = None


#############################################
# Low-Dimensional Continuous Testfunctions  #
#############################################

class Peaks(Test):
    """
    Peaks test function
    """
    def __init__(self):
        """
        Initializes Peaks test function
        """
        super(Peaks, self).__init__(dim=2)

        # define model
        self.model = testfunctions.Peaks

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x3"] = 0.

        # define problem
        self.problem = Problem(self.model, self.parameters)


class Franke(Test):
    """
    Franke test function
    """
    def __init__(self):
        """
        Initializes Franke test function
        """
        super(Franke, self).__init__(dim=2)

        # define model
        self.model = testfunctions.Franke

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class Lim2002(Test):
    """
    Lim2002 test function
    """
    def __init__(self):
        """
        Initializes Lim2002 test function
        """
        super(Lim2002, self).__init__(dim=2)

        # define model
        self.model = testfunctions.Lim2002

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class Ishigami(Test):
    """
    Ishigami test function
    """
    def __init__(self, dim=2):
        """
        Initializes Ishigami test function
        """
        super(Ishigami, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.Ishigami

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])

        if dim > 2:
            self.parameters["x3"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-np.pi, np.pi])
        else:
            self.parameters["x3"] = 0.

        self.parameters["a"] = 7.
        self.parameters["b"] = 0.1

        # define problem
        self.problem = Problem(self.model, self.parameters)


##########################################
# N-Dimensional Continuous Testfunctions #
##########################################

class ManufactureDecay(Test):
    """
    ManufactureDecay test function
    """
    def __init__(self, dim=2):
        """
        Initializes ManufactureDecay test function
        """
        super(ManufactureDecay, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.ManufactureDecay

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GenzContinuous(Test):
    """
    GenzContinuous test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzContinuous test function
        """
        super(GenzContinuous, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzContinuous

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GenzCornerPeak(Test):
    """
    GenzCornerPeak test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzCornerPeak test function
        """
        super(GenzCornerPeak, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzContinuous

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GenzGaussianPeak(Test):
    """
    GenzGaussianPeak test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzGaussianPeak test function
        """
        super(GenzGaussianPeak, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzGaussianPeak

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GenzOscillatory(Test):
    """
    GenzOscillatory test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzOscillatory test function
        """
        super(GenzOscillatory, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzOscillatory

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GenzProductPeak(Test):
    """
    GenzProductPeak test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzProductPeak test function
        """
        super(GenzProductPeak, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzProductPeak

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class Ridge(Test):
    """
    Ridge test function
    """
    def __init__(self, dim=2):
        """
        Initializes Ridge test function
        """
        super(Ridge, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.Ridge

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-4., 4.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class SphereFun(Test):
    """
    SphereFun test function
    """
    def __init__(self, dim=2):
        """
        Initializes SphereFun test function
        """
        super(SphereFun, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.SphereFun

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class GFunction(Test):
    """
    GFunction test function
    """
    def __init__(self, dim=2):
        """
        Initializes GFunction test function
        """
        super(GFunction, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GFunction

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        self.parameters["a"] = (np.arange(dim) + 1 - 2.) / 2

        # define problem
        self.problem = Problem(self.model, self.parameters)


class OakleyOhagan2004(Test):
    """
    OakleyOhagan2004 test function
    """
    def __init__(self):
        """
        Initializes OakleyOhagan2004 test function
        """
        super(OakleyOhagan2004, self).__init__(dim=15)

        # define model
        self.model = testfunctions.OakleyOhagan2004

        # define parameters
        self.parameters = OrderedDict()

        for i in range(self.dim):
            self.parameters["x{}".format(i)] = Norm(pdf_shape=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class Welch1992(Test):
    """
    Welch1992 test function
    """
    def __init__(self):
        """
        Initializes Welch1992 test function
        """
        super(Welch1992, self).__init__(dim=20)

        # define model
        self.model = testfunctions.Welch1992

        # define parameters
        self.parameters = OrderedDict()

        for i in range(self.dim):
            self.parameters["x{}".format(i+1)] = Beta(pdf_shape=[1., 1.], pdf_limits=[-0.5, 0.5])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class WingWeight(Test):
    """
    WingWeight test function
    """
    def __init__(self):
        """
        Initializes WingWeight test function
        """
        super(WingWeight, self).__init__(dim=10)

        # define model
        self.model = testfunctions.WingWeight

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[150., 200.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[220., 300.])
        self.parameters["x3"] = Beta(pdf_shape=[1., 1.], pdf_limits=[6., 10.])
        self.parameters["x4"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-10., 10.])
        self.parameters["x5"] = Beta(pdf_shape=[1., 1.], pdf_limits=[16., 45.])
        self.parameters["x6"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0.5, 1.])
        self.parameters["x7"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0.08, 0.18])
        self.parameters["x8"] = Beta(pdf_shape=[1., 1.], pdf_limits=[2.5, 6.])
        self.parameters["x9"] = Beta(pdf_shape=[1., 1.], pdf_limits=[1700., 2500.])
        self.parameters["x10"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0.025, 0.08])

        # define problem
        self.problem = Problem(self.model, self.parameters)


###############################################
# Low Dimensional Discontinuous Testfunctions #
###############################################

class HyperbolicTangent(Test):
    """
    HyperbolicTangent test function
    """
    def __init__(self):
        """
        Initializes HyperbolicTangent test function
        """
        super(HyperbolicTangent, self).__init__(dim=2)

        # define model
        self.model = testfunctions.HyperbolicTangent

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["x1"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])
        self.parameters["x2"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class MovingParticleFrictionForce(Test):
    """
    MovingParticleFrictionForce test function
    """
    def __init__(self):
        """
        Initializes MovingParticleFrictionForce test function
        """
        super(MovingParticleFrictionForce, self).__init__(dim=1)

        # define model
        self.model = testfunctions.MovingParticleFrictionForce

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["xi"] = Beta(pdf_shape=[1., 1.], pdf_limits=[-1., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)


class SurfaceCoverageSpecies(Test):
    """
    SurfaceCoverageSpecies test function
    """
    def __init__(self, dim):
        """
        Initializes SurfaceCoverageSpecies test function
        """
        super(SurfaceCoverageSpecies, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.SurfaceCoverageSpecies

        # define parameters
        self.parameters = OrderedDict()
        self.parameters["rho_0"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
        self.parameters["beta"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 20.])

        if dim > 2:
            self.parameters["alpha"] = Beta(pdf_shape=[1., 1.], pdf_limits=[0.1, 2.])
        else:
            self.parameters["alpha"] = 1.

            # define problem
        self.problem = Problem(self.model, self.parameters)


#############################################
# N-Dimensional Discontinuous Testfunctions #
#############################################
class GenzDiscontinuous(Test):
    """
    GenzDiscontinuous test function
    """
    def __init__(self, dim=2):
        """
        Initializes GenzDiscontinuous test function
        """
        super(GenzDiscontinuous, self).__init__(dim=dim)

        # define model
        self.model = testfunctions.GenzDiscontinuous

        # define parameters
        self.parameters = OrderedDict()

        for i in range(dim):
            self.parameters["x{}".format(i)] = Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])

        # define problem
        self.problem = Problem(self.model, self.parameters)