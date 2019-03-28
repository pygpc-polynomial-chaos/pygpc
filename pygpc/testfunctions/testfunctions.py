# -*- coding: utf-8 -*-
import numpy as np
import warnings
import scipy.special
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pygpc.AbstractModel import AbstractModel


def plot_testfunction(testfunction_name, *args):
    x = []
    for arg in args:
        x.append(arg)

    p = dict()

    if len(x) == 2:
        x1, x2 = np.meshgrid(x[0], x[1])
        p["x1"] = x1.flatten()
        p["x2"] = x2.flatten()

    else:
        p["x1"] = x[0]

    model = []
    exec ("model = {}(p, None)".format(testfunction_name))

    y = model.simulate(None)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = plt.pcolor(x1, x2, np.reshape(y, (100, 100), order='f'), cmap="jet")

    plt.title("{} function".format(model.__class__.__name__))
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show()


class Peaks(AbstractModel):
    """
    Two-dimensional peaks function.

    y = Peaks(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        Parameter 1
    p["x2"]: float or ndarray of float [n_grid]
        Parameter 2
    p["x3"]: float or ndarray of float [n_grid]
        Parameter 3

    Returns
    -------
    y: ndarray of float [n_grid x n_out]
        Output data
    misc: dict or list of dict [n_grid]
        Additional data, will be saved under its keys in the .hdf5 file during gPC simulations for every grid point
    """

    def __init__(self, p, context):
        super(Peaks, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):
        y = (3.0 * (1 - self.p["x1"]) ** 2. * np.exp(-(self.p["x1"] ** 2) - (self.p["x3"] + 1) ** 2)
             - 10.0 * (self.p["x1"] / 5.0 - self.p["x1"] ** 3 - self.p["x3"] ** 5)
             * np.exp(-self.p["x1"] ** 2 - self.p["x3"] ** 2) - 1.0 / 3
             * np.exp(-(self.p["x1"] + 1) ** 2 - self.p["x3"] ** 2)) + self.p["x2"]

        additional_data = {"additional_data/list_mult_int": [1, 2, 3],
                           "additional_data/list_single_float": [0.2],
                           "additional_data/list_single_str": ["test"],
                           "additional_data/list_mult_str": ["test1", "test2"],
                           "additional_data/single_float": 0.2,
                           "additional_data/single_int": 2,
                           "additional_data/single_str": "test"}

        # two output variables for testing
        if y.size > 1:
            y_out = np.array([y, 2*y]).transpose()
            additional_data = y.size * [additional_data]
        else:
            y_out = np.array([y, 2*y])

        return y_out, additional_data


class HyperbolicTangent(AbstractModel):
    """
    Two-dimensional hyperbolic tangent function [1] to simulate discontinuities. Discontinuity at x1 = 0.

    y = HyperbolicTangent(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        Parameter 1
    p["x2"]: float or ndarray of float [n_grid]
        Parameter 2

    Returns
    -------
    y: ndarray of float [n_grid x n_out]
        Output data

    Notes
    -----
    .. [1] Ahlfeld, R., Montomoli, F., Carnevale, M., Salvadore, S. (2018).
       Autonomous Uncertainty Quantification for Discontinuous Models Using Multivariate Pade Approximations.
       Journal of Turbomachinery, 104, 041004.
    """

    def __init__(self, p, context):
        super(HyperbolicTangent, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):
        y = np.array(np.tanh(10. * self.p["x1"]) + 0.2 * np.sin(10. * self.p["x1"]) + 0.3 * self.p["x2"] +
                     0.1 * np.sin(5. * self.p["x1"]))

        if len(y) > 1:
            y = y[:, np.newaxis]

        return y


class MovingParticleFrictionForce(AbstractModel):
    """
    Differential equation describing a particle moving under the influence of a
    potential field and of a friction force [1].

    d^2 X / dt^2 + f * dX/dt = -35/2 * X^3 + 15/2 * X

    with:

    x1 = X
    x2 = dX/dt

    we get a system of two 1st order ODE

    dx1/dt = x2
    dx2/dt + f*x2 = -35/2 * x1^3 + 15/2 * x1

    Discontinuity at randomly perturbed initial value x0 = X0 + delta_X * xi = 0.05 - 0.2 * 0.25
    and two stable fixed points:
    - X = -sqrt(15/35) for xi < -0.25
    - X = +sqrt(15/35) for xi > -0.25

    xi is uniform distributed [-1, 1]

    Mean value: 0.163663
    Standard deviation: 0.633865691

    y = MovingParticleFrictionForce(x)

    Parameters
    ----------
    p["x1"]: ndarray of float [1]
        Pertubation xi of initial value x0 (x0 = X0 + xi)

    Returns
    -------
    y: ndarray of float [1 x 1]
        X(t=10.)

    Notes
    -----
    .. [1] Le Maitre, O.P., Knio, O.M., Najm, H.N., Ghanem, R.G. (2004).
       Uncertainty propagation using Wiener-Haar expansions.
       Journal of Computational Physics, 197, 28-57.
    """

    def __init__(self, p, context):
        super(MovingParticleFrictionForce, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        # System of 1st order DEQ
        def deq(x, t, f):
            return x[1], -35. / 2. * x[0] ** 3. + 15. / 2. * x[0] - f * x[1]

        # Initial values
        x0 = 0.05
        delta_x = 0.2
        x0 = [x0 + delta_x * self.p["x1"], 0.]

        # Friction coefficient
        f = 2.

        # Simulation parameters
        dt = 0.001
        t_end = 10.
        t = np.arange(0, t_end, dt)

        # Solve
        y = odeint(deq, x0, t, args=(f,), hmin=dt)
        y_out = np.array([[y[-1, 0]]])

        return y_out


class SurfaceCoverageSpecies(AbstractModel):
    """
    Differential equation describing the time-evolution of the surface coverage rho [0, 1] for a given species [1].
    This problem has one or two fixed points according to the value of the recombination rate beta and it exhibits
    smooth dependence on the other parameters. The statistics of the solution at t=1 are investigated considering
    uncertainties in the initial coverage rho_0 and in the reaction parameter beta. Additionally uncertainty
    in the surface absorption rate alpha can be considered to make the problem 3-dimensional.
    Gamma=0.01 denotes the desorption rate.

    drho/dt = alpha * (1 - rho) - gamma * rho - beta * (rho - 1)^2 * rho

    y = SurfaceCoverageSpecies(x)

    Parameters
    ----------
    p["rho_0"]: ndarray of float [1]
        Initial value rho(t=0) (uniform distributed [0, 1])
    p["beta"]: ndarray of float [1]
        Recombination rate (uniform distributed [0, 20])
    p["alpha"]: ndarray of float [1]
        Surface absorption rate (1 or uniform distributed [0.1, 2])

    Returns
    -------
    y: ndarray of float [1 x 1]
        rho(t->1)

    Notes
    -----
    .. [1] Le Maitre, O.P., Najm, H.N., Ghanem, R.G., Knio, O.M., (2004).
       Multi-resolution analysis of Wiener-type uncertainty propagation schemes.
       Journal of Computational Physics, 197, 502-531.
    """

    def __init__(self, p, context):
        super(SurfaceCoverageSpecies, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        # System of 1st order DEQ
        def deq(rho, t, alpha, beta, gamma):
            return alpha * (1. - rho) - gamma * rho - beta * (rho - 1)**2 * rho

        # Constants
        gamma = 0.01

        # Simulation parameters
        dt = 0.01
        t_end = 1.
        t = np.arange(0, t_end, dt)

        # Solve
        y = odeint(deq, self.p["rho_0"], t, args=(self.p["alpha"], self.p["beta"], gamma,))
        y_out = np.array([y[-1]])

        return y_out


class Franke(AbstractModel):
    """
    Franke function [1] with 2 parameters. It is often used in regression or interpolation analysis.
    It is defined in the interval [0, 1] x [0, 1]. Hampton and Doostan used in the framework of BASE-PC [2].

    y = Franke(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["x2"]: float or ndarray of float [n_grid]
        Second parameter [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot

       x1 = np.linspace(0, 1, 100)
       x2 = np.linspace(0, 1, 100)
       plot("Franke", x1, x2)

    .. [1] Franke, R., (1979) A Critical Comparison of some Methods for Interpolation of Scattered Data,
       Tech. rep., DTIC Document

    .. [2] Hampton, J., Doostan, A., (2018), Basis adaptive sample efficient polynomial chaos (BASE-PC),
       Journal of Computational Physics, 371, 20-49.


    """

    def __init__(self, p, context):
        super(Franke, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        y = 3. / 4. * np.exp(-((9 * self.p["x1"] - 2) ** 2) / 4. - ((9 * self.p["x2"] - 2) ** 2) / 4.) + \
            3. / 4. * np.exp(-((9 * self.p["x1"] + 1) ** 2) / 49. - (9 * self.p["x2"] + 1) / 10.) + \
            1. / 2. * np.exp(-((9 * self.p["x1"] - 7) ** 2) / 4. - ((9 * self.p["x2"] - 3) ** 2) / 4.) - \
            1. / 5. * np.exp(-(9 * self.p["x1"] - 4) ** 2 - (9 * self.p["x2"] - 7) ** 2)

        if type(y) is not np.ndarray:
            y = np.array([y])

        y_out = y[:, np.newaxis]

        return y_out


class ManufactureDecay(AbstractModel):
    """
    N-dimensional manufacture decay function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    Hampton and Doostan used in the framework of BASE-PC [1].

    y = ManufactureDecay(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Hampton, J., Doostan, A., (2018), Basis adaptive sample efficient polynomial chaos (BASE-PC),
       Journal of Computational Physics, 371, 20-49.
    """

    def __init__(self, p, context):
        super(ManufactureDecay, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        # determine sum in exponent
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += np.sin(i+1) * self.p[key] / (i+1.)

        # determine output
        y = np.exp(2 - s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzContinuous(AbstractModel):
    """
    N-dimensional "continuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzContinuous(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/cont.html
    """

    def __init__(self, p, context):
        super(GenzContinuous, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum in exponent
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * np.abs(self.p[key] - u[i])

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzCornerPeak(AbstractModel):
    """
    N-dimensional "CornerPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzCornerPeak(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/copeak.html
    """

    def __init__(self, p, context):
        super(GenzCornerPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = (1 + s) ** -(n + 1)

        y_out = y[:, np.newaxis]

        return y_out


class GenzDiscontinuous(AbstractModel):
    """
    N-dimensional "Discontinuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzDiscontinuous(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/disc.html
    """

    def __init__(self, p, context):
        super(GenzDiscontinuous, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        mask = np.zeros((len(self.p[self.p.keys()[0]]), n))

        for i, key in enumerate(self.p.keys()):
            mask[:, i] = self.p[key] > u[i]
        mask = mask.any(axis=1)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = np.exp(s)
        y[mask] = 0.

        y_out = y[:, np.newaxis]

        return y_out


class GenzGaussianPeak(AbstractModel):
    """
    N-dimensional "GaussianPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzGaussianPeak(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/gaussian.html
    """

    def __init__(self, p, context):
        super(GenzGaussianPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i]**2 * (self.p[key] - u[i])**2

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzOscillatory(AbstractModel):
    """
    N-dimensional "Oscillatory" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzOscillatory(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/oscil.html
    """

    def __init__(self, p, context):
        super(GenzOscillatory, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = np.cos(2 * np.pi * u[0] + s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzProductPeak(AbstractModel):
    """
    N-dimensional "ProductPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    y = GenzProductPeak(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/prpeak.html
    """

    def __init__(self, p, context):
        super(GenzProductPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine output
        y = np.ones(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            y *= 1 / (a[i] ** (-2) + (self.p[key] - u[i]) ** 2)

        y_out = y[:, np.newaxis]

        return y_out


class Lim2002(AbstractModel):
    """
    Two-dimensional test function of Lim et al. (2002) [1].

    This function is a polynomial in two dimensions, with terms up to degree
    5. It is nonlinear, and it is smooth despite being complex, which is
    common for computer experiment functions (Lim et al., 2002).

    f(x) = 9 + 5/2*x1 - 35/2*x2 + 5/2*x1*x2 + 19*x2^2 - 15/2*x1^3 - 5/2*x1*x2^2 - 11/2*x2^4 + x1^3*x2^2

    Parameters
    ----------
    p["x"]: [N x 2] np.ndarray
        Input data, xi is element of [0, 1], for all i = 1, 2

    Returns
    -------
    y: [N x 1] np.ndarray
        Output data

    Notes
    -----
    .. [1] Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design
       and analysis of computer experiments when the output is highly correlated
       over the input space. Canadian Journal of Statistics, 30(1), 109-126.
    """

    def __init__(self, p, context):
        super(Lim2002, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        y = (9 + 5.0 / 2 * self.p["x1"] - 35.0 / 2 * self.p["x2"] + 5.0
             / 2 * self.p["x1"] * self.p["x2"] + 19 * self.p["x2"] ** 2
             - 15.0 / 2 * self.p["x1"] ** 3 - 5.0 / 2 * self.p["x1"] * self.p["x2"] ** 2
             - 11.0 / 2 * self.p["x2"] ** 4 + self.p["x1"] ** 3 * self.p["x2"] ** 2)
    
        return y


class Ishigami(AbstractModel):
    """
    Three-dimensional test function of Ishigami.

    The Ishigami function of Ishigami & Homma (1990) [1] is used as an example
    for uncertainty and sensitivity analysis methods, because it exhibits
    strong nonlinearity and nonmonotonicity. It also has a peculiar
    dependence on x3, as described by Sobol' & Levitan (1999) [2].

    f(x) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)

    Parameters
    ----------
    p["x"]: [N x 3] np.ndarray
        input data,
        xi ~ Uniform[-pi, pi], for all i = 1, 2, 3
    p["a"]: float
        shape parameter
    p["b"]: float
        shape parameter

    Returns
    -------
    y: [N x 1] np.ndarray
        output data

    Notes
    -----
    .. [1] Ishigami, T., Homma, T. (1990, December). An importance quantification
       technique in uncertainty analysis for computer models. In Uncertainty
       Modeling and Analysis, 1990. Proceedings., First International Symposium
       on (pp. 398-403). IEEE.

    .. [2] Sobol', I.M., Levitan, Y.L. (1999). On the use of variance reducing
       multipliers in Monte Carlo computations of a global sensitivity index.
       Computer Physics Communications, 117(1), 52-61.
    """

    def __init__(self, p, context):
        super(Ishigami, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):
        y = (np.sin(self.p["x1"]) + self.p["a"] * np.sin(self.p["x2"]) ** 2
             + self.p["b"] * self.p["x3"] ** 4 * np.sin(self.p["x1"]))
    
        return y


class Sphere0Fun(AbstractModel):
    """
    N-dimensional sphere function with zero mean.

    Parameters
    ----------
    p["x"]:  ndarray of float [N_input x N_dims]
        input data
    p["a"]: ndarray of float [N_dims]
        lower bound of input data
    p["b"]: ndarray of float [N_dims]
        upper bound of input data

    Returns
    -------
    y: [N_input] np.ndarray
        output data
    """

    def __init__(self, p, context):
        super(Sphere0Fun, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        try:
            n = self.p["x"].shape[1]
        except IndexError:
            n = 1
            self.p["x"] = np.array([self.p["x"]])

        # zero mean
        c2 = (1.0 * n * (self.p["b"] ** 3 - self.p["a"] ** 3)) / (3 * (self.p["b"] - self.p["a"]))

        # sphere function
        y = (np.sum(np.square(self.p["x"]), axis=1) - c2)
    
        return y


class SphereFun(AbstractModel):
    """
    N-dimensional sphere function.

    Parameters
    ----------
    p["x"]: ndarray of float [N_input x N_dims]
        Input data

    Returns
    -------
    y: ndarray of float [N_input x 1]
        Output data
    """

    def __init__(self, p, context):
        super(SphereFun, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        y = (np.sum(np.square(self.p["x"]), axis=1))
    
        return y


class GFunction(AbstractModel):
    """
    N-dimensional g-function used by Saltelli and Sobol (1995) [1].

    This test function is used as an integrand for various numerical
    estimation methods, including sensitivity analysis methods, because it
    is fairly complex, and its sensitivity indices can be expressed
    analytically. The exact value of the integral with this function as an
    integrand is 1.

    Parameters
    ----------
    p["x"]: ndarray of float [N_input x N_dims]
        Input data
    p["a"]: ndarray of float [N_dims]
        Importance factor of dimensions

    Returns
    -------
    y: ndarray of float [N_input x 1]
        Output data

    Notes
    -----
    .. [1] Saltelli, Andrea; Sobol, I. M. (1995): Sensitivity analysis for nonlinear
       mathematical models: numerical experience. In: Mathematical models and
       computer experiment 7 (11), S. 16-28.
    """

    def __init__(self, p, context):
        super(GFunction, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        try:
            self.p["x"].shape[1]
        except IndexError:
            self.p["x"] = np.array([self.p["x"]])

        y = (np.prod((np.abs(4.0 * self.p["x"] - 2) + self.p["a"]) / (1.0 + self.p["a"]), axis=1))
    
        return y


class OakleyOhagan2004(AbstractModel):
    """
    15-dimensional test function of Oakley and O'Hagan (2004) [1].

    This function's a-coefficients are chosen so that 5 of the input
    variables contribute significantly to the output variance, 5 have a
    much smaller effect, and the remaining 5 have almost no effect on the
    output variance.

    Parameters
    ----------
    p["x"]: ndarray of float [N_input x 15]
        Input data, xi ~ N(mu=0, sigma=1), for all i = 1, 2,..., 15.

    Returns
    -------
    y: ndarray of float [N_input x 1]
        Output data

    Notes
    -----
    .. [1] Oakley, J. E., & O'Hagan, A. (2004). Probabilistic sensitivity analysis
       of complex models: a Bayesian approach. Journal of the Royal Statistical
       Society: Series B (Statistical Methodology), 66(3), 751-769.
    """
    def __init__(self, p, context):
        super(OakleyOhagan2004, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        # load coefficients
        m = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_M.txt')
        a1 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a1.txt')
        a2 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a2.txt')
        a3 = np.loadtxt('../pckg/data/oakley_ohagan_2004/oakley_ohagan_2004_a3.txt')

        # function
        y = (np.dot(self.p["x"], a1) + np.dot(np.sin(self.p["x"]), a2)
             + np.dot(np.cos(self.p["x"]), a3) + np.sum(np.multiply(np.dot(self.p["x"], m), self.p["x"]), axis=1))

        return y


class Welch1992(AbstractModel):
    """
    20-dimensional test function of Welch et al. (1992) [1].

    For input variable screening purposes, it can be found that some input
    variables of this function have a very high effect on the output,
    compared to other input variables. As Welch et al. (1992) [1] point out,
    interactions and nonlinear effects make this function challenging.

    Parameters
    ----------
    p["x1...x20"]: float
        Input data, xi ~ U(-0.5, 0.5), for all i = 1,..., 20.

    Returns
    -------
    y: ndarray of float [1]
        Output data

    Notes
    -----
    .. [1] Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., Morris, M. D. (1992).
       Screening, predicting, and computer experiments. Technometrics, 34(1), 15-25.
    """

    def __init__(self, p, context):
        super(Welch1992, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        y = (5.0 * self.p["x12"] / (1 + self.p["x1"]) + 5 * (self.p["x4"] - self.p["x20"]) ** 2
             + self.p["x5"] + 40 * self.p["x19"] ** 3 + 5 * self.p["x19"] + 0.05 * self.p["x2"]
             + 0.08 * self.p["x3"] - 0.03 * self.p["x6"] + 0.03 * self.p["x7"]
             - 0.09 * self.p["x9"] - 0.01 * self.p["x10"] - 0.07 * self.p["x11"]
             + 0.25 * self.p["x13"] ** 2 - 0.04 * self.p["x14"]
             + 0.06 * self.p["x15"] - 0.01 * self.p["x17"] - 0.03 * self.p["x18"])
    
        return y


class WingWeight(AbstractModel):
    """
    10-dimensional test function which models a light aircraft wing from Forrester et al. (2008) [1]

    Parameters
    ----------
    p["x1"]: float
        x1(Sw) [150, 200]
    p["x2"]: float
        x2(Wfw) [220, 300]
    p["x3"]: float
        x3(A) [6, 10]
    p["x4"]: float
        x4(Lambda) [-10, 10]
    p["x5"]: float
        x5(q) [16, 45]
    p["x6"]: float
        x6(lambda) [0.5, 1]
    p["x7"]: float
        x7(tc) [0.08, 0.18]
    p["x8"]: float
        x8(Nz) [2.5, 6]
    p["x9"]: float
        x9(Wdg) [1700, 2500]
    p["x10"]: float
        x10(Wp) [0.025, 0.08]

    Returns
    -------
    y: float
        output data

    Notes
    -----
    .. [1] Forrester, A., Sobester, A., & Keane, A. (2008).
       Engineering design via surrogate modelling: a practical guide. John Wiley & Sons.
    """

    def __init__(self, p, context):
            super(WingWeight, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):
        y = (0.036 * self.p["x1"] ** 0.758 * self.p["x2"] ** 0.0035
             * (self.p["x3"] / np.cos(self.p["x4"]) ** 2) ** 0.6
             * self.p["x5"] ** 0.006 * self.p["x6"] ** 0.04
             * (100 * self.p["x7"] / np.cos(self.p["x4"])) ** -0.3
             * (self.p["x8"] * self.p["x9"]) ** 0.49
             + self.p["x1"] * self.p["x10"])
    
        return y


class SphereModel(AbstractModel):
    """
    Calculates the electric potential in a 3-layered sphere caused by point-like electrodes
    after Rush and Driscoll (1969) [1].

    Parameters
    ----------
    p["sigma_1"]: float
        Conductivity of the innermost layer, in (S/m)
    p["sigma_2"]: float
        Conductivity of the intermediate layer, in (S/m)
    p["sigma_3"]: float
        Conductivity of the outermost layer, in (S/m)
    p["radii"]: list [3]
        Radius of each of the 3 layers (innermost to outermost), in (mm)
    p["anode_pos"]: ndarray of float [3 x 1]
        Position of the anode_pos, in (mm)
    p["cathode_pos"]: ndarray of float [3 x 1]
        Position of cathode_pos, in (mm)
    p["p"]: ndarray of float [N x 3]
        Positions where the potential should be calculated, in (mm)

    Returns
    -------
    potential: ndarray of float [N x 1]
        Values of the electric potential, in (V)

    Notes
    -----
    .. [1] Rush, S., & Driscoll, D. A. (1969). EEG electrode sensitivity-an application of reciprocity.
       IEEE transactions on biomedical engineering, (1), 15-22.
    """

    def __init__(self, p, context):
        super(SphereModel, self).__init__(p, context)

        # number of of legendre polynomials to use
        self.nbr_polynomials = 50

    def validate(self):
        pass

    def simulate(self, process_id):

        assert len(self.p["R"]) == 3
        assert self.p["R"][0] < self.p["R"][1] < self.p["R"][2]
        assert len(self.p["anode_pos"]) == 3
        assert len(self.p["cathode_pos"]) == 3
        assert self.p["points"].shape[1] == 3

        b_over_s = float(self.p["sigma_1"]) / float(self.p["sigma_2"])
        s_over_t = float(self.p["sigma_2"]) / float(self.p["sigma_3"])
        radius_brain = self.p["R"][0] * 1e-3
        radius_skull = self.p["R"][1] * 1e-3
        radius_skin = self.p["R"][2] * 1e-3

        r = np.linalg.norm(self.p["points"], axis=1) * 1e-3
        theta = np.arccos(self.p["points"][:, 2] * 1e-3 / r)
        phi = np.arctan2(self.p["points"][:, 1], self.p["points"][:, 0])

        p_r = np.vstack((r, theta, phi)).T

        cathode_pos = (np.sqrt(self.p["cathode_pos"][0]**2 +
                               self.p["cathode_pos"][1]**2 +
                               self.p["cathode_pos"][2]**2) * 1e-3,
                       np.arccos(self.p["cathode_pos"][2] /
                                 np.sqrt(self.p["cathode_pos"][0]**2 +
                                         self.p["cathode_pos"][1]**2 +
                                         self.p["cathode_pos"][2]**2)),
                       np.arctan2(self.p["cathode_pos"][1], self.p["cathode_pos"][0]))

        anode_pos = (np.sqrt(self.p["anode_pos"][0]**2 + self.p["anode_pos"][1]**2 + self.p["anode_pos"][2]**2) * 1e-3,
                     np.arccos(self.p["anode_pos"][2] /
                               np.sqrt(self.p["anode_pos"][0] ** 2 +
                                       self.p["anode_pos"][1]**2 +
                                       self.p["anode_pos"][2]**2)),
                     np.arctan2(self.p["anode_pos"][1], self.p["anode_pos"][0]))

        def a(n):
            return ((2 * n + 1)**3 / (2 * n)) / (((b_over_s + 1) * n + 1) * ((s_over_t + 1) * n + 1) +
                                                 (b_over_s - 1) * (s_over_t - 1) * n * (n + 1) *
                                                 (radius_brain / radius_skull)**(2 * n + 1) +
                                                 (s_over_t - 1) * (n + 1) * ((b_over_s + 1) * n + 1) *
                                                 (radius_skull / radius_skin)**(2 * n + 1) +
                                                 (b_over_s - 1) * (n + 1) * ((s_over_t + 1) * (n + 1) - 1) *
                                                 (radius_brain / radius_skin)**(2 * n + 1))

        # All of the bellow is modified: division by radius_skin moved to the
        # coefficients calculations due to numerical constraints
        # THIS IS DIFFERENT FROM THE PAPER (there's a sum instead of difference)
        def s(n):
            return (a(n)) * ((1 + b_over_s) * n + 1) / (2 * n + 1)

        def u(n):
            return (a(n) * radius_skin) * n * (1 - b_over_s) * \
                   radius_brain**(2 * n + 1) / (2 * n + 1)

        def t(n):
            return (a(n) / ((2 * n + 1)**2)) * \
                   (((1 + b_over_s) * n + 1) * ((1 + s_over_t) * n + 1) +
                    n * (n + 1) * (1 - b_over_s) * (1 - s_over_t) * (radius_brain / radius_skull)**(2 * n + 1))

        def w(n):
            return ((n * a(n) * radius_skin) / ((2 * n + 1)**2)) * \
                   ((1 - s_over_t) * ((1 + b_over_s) * n + 1) * radius_skull**(2 * n + 1) +
                    (1 - b_over_s) * ((1 + s_over_t) * n + s_over_t) * radius_brain**(2 * n + 1))

        brain_region = np.where(p_r[:, 0] <= radius_brain)[0]
        skull_region = np.where(
            (p_r[:, 0] > radius_brain) * (p_r[:, 0] <= radius_skull))[0]
        skin_region = np.where((p_r[:, 0] > radius_skull)
                               * (p_r[:, 0] <= radius_skin))[0]
        inside_sphere = np.where((p_r[:, 0] <= radius_skin))[0]
        outside_sphere = np.where((p_r[:, 0] > radius_skin))[0]

        cos_theta_a = np.cos(cathode_pos[1]) * np.cos(p_r[:, 1]) +\
            np.sin(cathode_pos[1]) * np.sin(p_r[:, 1]) * \
            np.cos(p_r[:, 2] - cathode_pos[2])
        cos_theta_b = np.cos(anode_pos[1]) * np.cos(p_r[:, 1]) +\
            np.sin(anode_pos[1]) * np.sin(p_r[:, 1]) * \
            np.cos(p_r[:, 2] - anode_pos[2])

        potentials = np.zeros((self.p["points"].shape[0]), dtype='float64')

        coefficients = np.zeros((self.nbr_polynomials, self.p["points"].shape[0]), dtype='float64')

        # accelerate
        for ii in range(1, self.nbr_polynomials):
            ni = float(ii)
            coefficients[ii, brain_region] = np.nan_to_num(
                a(ni) * ((p_r[brain_region, 0] / radius_skin)**ni))

            coefficients[ii, skull_region] = np.nan_to_num(s(ni) * (p_r[skull_region, 0] / radius_skin)**ni +
                                                           u(ni) * (p_r[skull_region, 0] * radius_skin)**(-ni - 1))

            coefficients[ii, skin_region] = np.nan_to_num(t(ni) * (p_r[skin_region, 0] / radius_skin)**ni
                                                          + w(ni) * (p_r[skin_region, 0] * radius_skin)**(-ni - 1))

        potentials[inside_sphere] = np.nan_to_num(
            np.polynomial.legendre.legval(cos_theta_a[inside_sphere], coefficients[:, inside_sphere], tensor=False) -
            np.polynomial.legendre.legval(cos_theta_b[inside_sphere], coefficients[:, inside_sphere], tensor=False))

        potentials *= 1.0 / (2 * np.pi * self.p["sigma_3"] * radius_skin)

        potentials[outside_sphere] = 0.0

        return potentials


class PotentialHomogeneousDipole(AbstractModel):
    """
    Calculates the surface potential generated by a dipole inside a homogeneous conducting sphere after Yao (2000).

    Parameters
    ----------
    p["sphere_radius"]: float
        Radius of sphere in (mm)
    p["conductivity"]: float
        Conductivity of medium in (S/m)
    p["dipole_pos"]: ndarray of float [3 x 1]
        Position of dipole in (mm)
    p["dipole_moment"]: ndarray of float [3 x 1]
        Moment of dipole in (Cm)
    p["detector_positions"]: ndarray of float [n x 3]
        Position of detectors, will be projected into the sphere surface in (mm)

    Returns
    -------
    potential: [n x 1] np.ndarray
       Potential at the points

    Notes
    -----
    .. [1] Yao, D. (2000). Electric potential produced by a dipole in a homogeneous conducting sphere.
       IEEE Transactions on Biomedical Engineering, 47(7), 964-966.
    """

    def __init__(self, p, context):
        super(PotentialHomogeneousDipole, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        self.p["detector_positions"] = np.atleast_2d(self.p["detector_positions"])
        assert self.p["detector_positions"].shape[1] == 3
        assert np.linalg.norm(self.p["dipole_pos"]) < self.p["sphere_radius"]

        self.p["sphere_radius"] = np.float128(self.p["sphere_radius"] * 1e-3)
        self.p["dipole_pos"] = np.array(self.p["dipole_pos"], dtype=np.float128) * 1e-3
        self.p["dipole_moment"] = np.array(self.p["dipole_moment"], dtype=np.float128)
        self.p["detector_positions"] = np.array(self.p["detector_positions"], dtype=np.float128) * 1e-3

        rs = self.p["sphere_radius"]
        r0 = np.linalg.norm(self.p["dipole_pos"])
        r = np.linalg.norm(self.p["detector_positions"], axis=1)
        rp = np.linalg.norm(self.p["dipole_pos"] - self.p["detector_positions"], axis=1)

        if not np.allclose(r, rs):
            warnings.warn('Some points are not in the surface!!')

        if np.isclose(r0, 0):
            cos_phi = np.zeros(len(self.p["detector_positions"]), dtype=np.float128)
        else:
            cos_phi = self.p["dipole_pos"].dot(self.p["detector_positions"].T) / \
                      (np.linalg.norm(self.p["dipole_pos"]) * np.linalg.norm(self.p["detector_positions"], axis=1))

        second_term = 1. / (rp[:, None] * rs ** 2) * (
                self.p["detector_positions"] + (self.p["detector_positions"] * r0 * cos_phi[:, None] -
                                                rs * self.p["dipole_pos"]) /
                (rs + rp - r0 * cos_phi)[:, None])

        potential = self.p["dipole_moment"].dot((2 * (self.p["detector_positions"] - self.p["dipole_pos"]) /
                                                 (rp ** 3)[:, None] + second_term).T).T

        potential /= 4 * np.pi * self.p["conductivity"]

        return potential


class BfieldOutsideSphere(AbstractModel):
    """
    Calculates the B-field outside a sphere, does not depend on conductivity after Jarvas (1987).
    Dipole in SI units, positions in (mm)

    Parameters
    ----------
    p["sphere_radius"]: float
        Radius of sphere in (mm)
    p["dipole_pos"]: ndarray of float [3 x 1]
        Position of dipole in (mm)
    p["dipole_moment"]: ndarray of float [3 x 1]
        Moment of dipole in (Ams)
    p["detector_positions"]: ndarray of float [n x 3]
        Position of detectors, must lie outside sphere

    Returns
    -------
    B: ndarray of float [N x 3]
        B-fields in detector positions

    Notes
    -----
    .. [1] Sarvas, J. (1987). Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem.
       Physics in Medicine & Biology, 32(1), 11.
    """

    def __init__(self, p, context):
        super(BfieldOutsideSphere, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        pos = np.array(self.p["dipole_pos"], dtype=float) * 1e-3
        moment = np.array(self.p["dipole_moment"], dtype=float)
        detector = np.array(self.p["detector_positions"], dtype=float) * 1e-3

        assert np.all(np.linalg.norm(self.p["detector_positions"], axis=1) > self.p["sphere_radius"]), \
            "All points must be outside the sphere"

        assert np.all(np.linalg.norm(self.p["dipole_pos"]) < self.p["sphere_radius"]), \
            "Dipole must be outside sphere"

        b = np.zeros(self.p["detector_positions"].shape, dtype=float)

        for ii, r in enumerate(detector):
            norm_r = np.linalg.norm(r)

            r_0 = pos
            # norm_r0 = np.linalg.norm(pos)

            a = r - r_0
            norm_a = np.linalg.norm(a)

            f = norm_a * (norm_r * norm_a + norm_r ** 2 - r_0.dot(r))

            grad_f = (norm_r ** (-1) * norm_a ** 2 + norm_a ** (-1) * a.dot(r) + 2 * norm_a + 2 * norm_r) * r - \
                     (norm_a + 2 * norm_r + norm_a ** (-1) * a.dot(r)) * r_0

            b[ii, :] = (4 * np.pi * 1e-7) / \
                       (4 * np.pi * f ** 2) * (f * np.cross(moment, r_0) - np.dot(np.cross(moment, r_0), r) * grad_f)

        return b


class TMSEfieldSphere(AbstractModel):
    """
    Calculate the E-field in a sphere caused by external magnetic dipoles after Heller and van Hulsteyn (1992).
    The results are independent of conductivity.

    Parameters
    ----------
    p["dipole_pos"]: ndarray of float [M x 3]
        Position of dipoles, must be outside sphere
    p["dipole_moment"]: ndarray of float [m x 3]
        Moment of dipoles
    p["didt"]: float
        Variation rate of current in the coil
    p["positions"]: ndarray of float [N x 3]
        Position where fields should be calculated, must lie inside sphere in (mm)

    Returns
    -------
    E: [N x 3] np.ndarray
        E-fields at detector positions

    Notes
    -----
    .. [1] Heller, L., & van Hulsteyn, D. B. (1992). Brain stimulation using electromagnetic sources:
       theoretical aspects. Biophysical Journal, 63(1), 129-138.
    """

    def __init__(self, p, context):
        super(TMSEfieldSphere, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id):

        if self.p["dipole_pos"].shape != self.p["dipole_moment"].shape:
            raise ValueError('List of dipole position and moments should have the same'
                             'lengths')
        mu0_4pi = 1e-7

        e = np.zeros(self.p["positions"].shape, dtype=float)
        dp = np.atleast_2d(self.p["dipole_pos"])
        dm = np.atleast_2d(self.p["dipole_moment"])

        r1 = self.p["positions"]

        for m, r2 in zip(dm, dp):
            a = r2 - r1
            norm_a = np.linalg.norm(a, axis=1)[:, None]

            # norm_r1 = np.linalg.norm(r1, axis=1)[:, None]
            norm_r2 = np.linalg.norm(r2)

            r2_dot_a = np.sum(r2 * a, axis=1)[:, None]
            f = norm_a * (norm_r2 * norm_a + r2_dot_a)
            grad_f = (norm_a ** 2 / norm_r2 + 2 * norm_a + 2 * norm_r2 + r2_dot_a / norm_a) * r2 - \
                     (norm_a + 2 * norm_r2 + r2_dot_a / norm_a) * r1
            e += -self.p["didt"] * mu0_4pi / f ** 2 * (
                 f * np.cross(r1, m) - np.cross(np.sum(m * grad_f, axis=1)[:, None] * r1, r2))

        return e


class PotentialDipole3Layers(AbstractModel):
    """
    Calculates the electric potential in a 3-layered sphere caused by a dipole after Ary et al. (1981).

    Parameters
    ----------
    p["radii"]: list [3]
        Radius of each of the 3 layers (innermost to outermost) in (mm)
    p["cond_brain_scalp"]: float
        Conductivity of the brain and scalp layers in (S/m)
    p["cond_skull"]: float
        Conductivity of the skull layer in (S/m)
    p["dipole_pos"]: ndarray of float [3 x 1]
        Position of the dipole, in (mm)
    p["dipole_moment"]: ndarray of float [3 x 1]
        Moment of dipole, in (Cm)
    p["surface_points"]: ndarray of float [N x 3]
        List of positions where the potential should be calculated in (mm)

    Returns
    -------
    potential: ndarray of float [N x 1]
        Values of the electric potential, in (V)

    Notes
    -----
    .. [1] Ary, J. P., Klein, S. A., & Fender, D. H. (1981). Location of sources of evoked scalp potentials:
       corrections for skull and scalp thicknesses. IEEE Transactions on Biomedical Engineering, (6), 447-452.
       eq. 2 and 2a
    """

    def __init__(self, p, context):
        super(PotentialDipole3Layers, self).__init__(p, context)

        # Number of of legendre polynomials to use (default = 100)
        self.nbr_polynomials = 100

    def validate(self):
        pass

    def simulate(self, process_id):

        assert len(self.p["radii"]) == 3
        assert self.p["radii"][0] < self.p["radii"][1] < self.p["radii"][2]
        assert len(self.p["dipole_moment"]) == 3
        assert len(self.p["dipole_pos"]) == 3
        assert self.p["surface_points"].shape[1] == 3
        assert np.linalg.norm(self.p["dipole_pos"]) < self.p["radii"][0], "Dipole must be inside inner sphere"

        xi = float(self.p["cond_skull"]) / float(self.p["cond_brain_scalp"])
        r = float(self.p["radii"][2] * 1e-3)
        f1 = float(self.p["radii"][0] * 1e-3) / r
        f2 = float(self.p["radii"][1] * 1e-3) / r
        b = np.linalg.norm(self.p["dipole_pos"]) * 1e-3 / r

        if not np.allclose(np.linalg.norm(self.p["surface_points"], axis=1), r * 1e3):
            warnings.warn('Some points are not in the surface!!')

        if np.isclose(b, 0):
            r_dir = np.array(self.p["dipole_moment"], dtype=float)
            r_dir /= np.linalg.norm(r_dir)
        else:
            r_dir = self.p["dipole_pos"] / np.linalg.norm(self.p["dipole_pos"])

        m_r = np.dot(self.p["dipole_moment"], r_dir)
        cos_alpha = self.p["surface_points"].dot(r_dir) / r * 1e-3

        t_dir = self.p["dipole_moment"] - m_r * r_dir

        # if the dipole is radial only
        if np.isclose(np.linalg.norm(self.p["dipole_moment"]), np.abs(np.dot(r_dir, self.p["dipole_moment"]))):
            # try to set an axis in x, if the dipole is not in x
            if not np.allclose(np.abs(r_dir.dot([1, 0, 0])), 1):
                t_dir = np.array([1., 0., 0.], dtype=float)
            # otherwise, set it in y
            else:
                t_dir = np.array([0., 1., 0.], dtype=float)
            t_dir = t_dir - r_dir.dot(t_dir)

        t_dir /= np.linalg.norm(t_dir)
        t2_dir = np.cross(r_dir, t_dir)
        m_t = np.dot(self.p["dipole_moment"], t_dir)
        beta = np.arctan2(self.p["surface_points"].dot(t2_dir), self.p["surface_points"].dot(t_dir))
        cos_beta = np.cos(beta)

        def d(n):
            d_n = ((n + 1) * xi + n) * ((n * xi) / (n + 1) + 1) + \
                  (1 - xi) * ((n + 1) * xi + n) * (f1 ** (2 * n + 1) - f2 ** (2 * n + 1)) - \
                  n * (1 - xi) ** 2 * (f1 / f2) ** (2 * n + 1)
            return d_n

        potential = np.zeros(self.p["surface_points"].shape[0], dtype='float64')

        p = np.zeros((2, self.nbr_polynomials + 1, self.p["surface_points"].shape[0]), dtype='float64')

        for ii, ca in enumerate(cos_alpha):
            p[:, :, ii], _ = scipy.special.lpmn(1, self.nbr_polynomials, ca)

        for ii in range(1, self.nbr_polynomials + 1):
            ni = float(ii)
            potential += np.nan_to_num(
                (2 * ni + 1) / ni * b ** (ni - 1) * ((xi * (2 * ni + 1) ** 2) / (d(ni) * (ni + 1))) *
                (ni * m_r * p[0, ii, :] - m_t * p[1, ii, :] * cos_beta))

        potential /= 4 * np.pi * self.p["cond_brain_scalp"] * r ** 2

        return potential
