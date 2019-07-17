# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import scipy.special
from scipy.integrate import odeint
from collections import OrderedDict
from pygpc.AbstractModel import AbstractModel

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    warnings.warn("If you want to use plot functionality from pygpc, "
                  "please install matplotlib (pip install matplotlib).")
    pass


def plot_testfunction(testfunction_name, parameters, constants=None, output_idx=0):
    """
    Plot 1D or 2D testfunctions for documentation.

    Parameters
    ----------
    testfunction_name : str
        Name of testfunction AbstractModel class
    parameters : OrdererdDict
        Dictionary containing the 1D coordinates as ndarrays, where the testfunction is evaluated (will be tensorized)
    constants : OrderedDict (optional)
        Dictionary containing the (remaining) parameters treated as constants
    output_idx : int or list of int
        Indices of output quantity to plot

    Returns
    -------
    <plot> : matplotlib figure
        Plot showing the QoI of the testfunction in 1D or 2D
    """
    if type(output_idx) is not list:
        output_idx = [output_idx]

    n_qoi = len(output_idx)

    # setup parameters
    p_names = list(parameters.keys())
    p = OrderedDict()

    if len(p_names) == 2:
        x1, x2 = np.meshgrid(parameters[p_names[0]], parameters[p_names[1]])
        p[p_names[0]] = x1.flatten()
        p[p_names[1]] = x2.flatten()

    else:
        p[p_names[0]] = parameters[p_names[0]].flatten()

    # add constants
    if type(constants) is dict or type(constants) is OrderedDict:
        c_names = constants.keys()
        for c_name in c_names:
            p[c_name] = np.tile(constants[c_name], (len(p[p_names[0]]), 1))

    model = eval("{}().set_parameters(p)".format(testfunction_name))

    y = model.simulate()

    fig, ax = plt.subplots(n_qoi, figsize=(6, ((n_qoi-1)*0.85+1) * 5), sharex=False, sharey=False)

    if type(ax) is not np.ndarray:
        ax = np.array([ax])

    for o in output_idx:
        # omit "additional_data" if present
        if type(y) is tuple:
            y = y[0]

        if len(p_names) == 2:
            im = ax[o].pcolor(x1,
                              x2,
                              np.reshape(y[:, o],
                                         (len(parameters[p_names[1]]), len(parameters[p_names[0]])),
                                         order='c'),
                              cmap="jet")
            ax[o].set_ylabel(r"${}$".format(p_names[1]), fontsize=12)
            fig.colorbar(im, ax=ax[o], orientation='vertical')

        else:
            ax[o].plot(parameters[p_names[0]], y)
            ax[o].set_ylabel(r"$y({})$".format(p_names[0]), fontsize=12)

        ax[o].set_xlabel(r"${}$".format(p_names[0]), fontsize=12)

    ax[0].set_title("{} function".format(model.__class__.__name__))
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

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       constants = OrderedDict()
       constants["x3"] = 0.
       plot("Peaks", parameters, constants)
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        if type(self.p["x1"]) is np.ndarray:
            self.p["x1"] = self.p["x1"].flatten()
        if type(self.p["x2"]) is np.ndarray:
            self.p["x2"] = self.p["x2"].flatten()
        if type(self.p["x3"]) is np.ndarray:
            self.p["x3"] = self.p["x3"].flatten()

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
            y_out = np.array([y, 2 * y]).transpose()
            additional_data = y.size * [additional_data]
        else:
            y_out = np.array([y, 2 * y])

        return y_out, additional_data


class DiscontinuousRidgeManufactureDecayGenzDiscontinuous(AbstractModel):
    """
    N-dimensional discontinuous test function. The first QOI corresponds to the
    DiscontinuousRidgeManufactureDecay function and the second QOI to GenzDiscontinuous.

    y = DiscontinuousRidgeManufactureDecayGenzDiscontinuous(x)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        Parameter 1 [0, 1]
    p["x2"]: float or ndarray of float [n_grid]
        Parameter 2 [0, 1]
    p["x3"]: float or ndarray of float [n_grid]
        Parameter 3 [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x n_out]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("DiscontinuousRidgeManufactureDecayGenzDiscontinuous", parameters, output_idx=[0, 1])
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y_1 = DiscontinuousRidgeManufactureDecay(self.p).simulate()
        y_2 = BinaryDiscontinuousSphere(self.p).simulate()

        y = np.hstack((y_1, y_2))

        return y


class HyperbolicTangent(AbstractModel):
    """
    Two-dimensional hyperbolic tangent function [1] to simulate discontinuities. Discontinuity at x1 = 0.

    .. math::
       y(x_1, x_2) = \\tanh(10 x_1) + 0.2 \sin(10 x_1) + 0.3 x_2 + 0.1 \sin(5 x_1)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        Parameter 1 [-1, 1]
    p["x2"]: float or ndarray of float [n_grid]
        Parameter 2 [-1, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(-1, 1, 100)
       parameters["x2"] = np.linspace(-1, 1, 100)

       plot("HyperbolicTangent", parameters)

    .. [1] Ahlfeld, R., Montomoli, F., Carnevale, M., Salvadore, S. (2018).
       Autonomous Uncertainty Quantification for Discontinuous Models Using Multivariate Pade Approximations.
       Journal of Turbomachinery, 104, 041004.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        y = np.array(np.tanh(10. * self.p["x1"]) + 0.2 * np.sin(10. * self.p["x1"]) + 0.3 * self.p["x2"] +
                     0.1 * np.sin(5. * self.p["x1"]))

        if len(y) > 1:
            y = y[:, np.newaxis]

        return y


class MovingParticleFrictionForce(AbstractModel):
    """
    Differential equation describing a particle moving under the influence of a
    potential field and of a friction force [1].

    .. math:: \\frac{d^2 x}{dt^2} + f \\frac{dx}{dt} = -\\frac{35}{2} x^3 + \\frac{15}{2} x

    with:

    .. math:: x_1 = x
    .. math:: x_2 = \\frac{dx}{dt}

    we get a system of two 1st order ODE

    .. math:: \\frac{d x_1}{dt} = x_2
    .. math:: \\frac{d x_2}{dt} = -\\frac{35}{2} x_1^3 + \\frac{15}{2} x_1 - f x_2

    Discontinuity at randomly perturbed initial value x0 = X0 + delta_X * xi = 0.05 - 0.2 * 0.25
    and two stable fixed points:

    .. math:: x = -\sqrt{15/35} \; \mathrm{for} \; \\xi < -0.25
    .. math:: x = +\sqrt{15/35} \; \mathrm{for} \; \\xi > -0.25

    xi is uniform distributed [-1, 1]

    Mean value: 0.163663
    Standard deviation: 0.633865691

    Parameters
    ----------
    p["xi"]: ndarray of float [1]
        Pertubation xi of initial value x0 (x0 = X0 + xi) [-1, 1]

    Returns
    -------
    y: ndarray of float [1 x 1]
        x(t=10.)

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["xi"] = np.linspace(-1, 1, 100)

       plot("MovingParticleFrictionForce", parameters)

    .. [1] Le Maitre, O.P., Knio, O.M., Najm, H.N., Ghanem, R.G. (2004).
       Uncertainty propagation using Wiener-Haar expansions.
       Journal of Computational Physics, 197, 28-57.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # System of 1st order DEQ
        def deq(x, t, f):
            return x[1], -35. / 2. * x[0] ** 3. + 15. / 2. * x[0] - f * x[1]

        # Initial values
        x0 = 0.05
        delta_x = 0.2

        # Friction coefficient
        f = 2.

        # Simulation parameters
        dt = 0.001
        t_end = 10.
        t = np.arange(0, t_end, dt)

        # Solve
        y_out = np.zeros((len(self.p["xi"]), 1))

        for i in range(len(y_out)):
            x0_init = [x0 + delta_x * self.p["xi"].flatten()[i], 0.]
            y = odeint(deq, x0_init, t, args=(f,), hmin=dt)
            y_out[i, 0] = np.array([[y[-1, 0]]])

        return y_out


class SurfaceCoverageSpecies(AbstractModel):
    """
    Differential equation describing the time-evolution of the surface coverage rho [0, 1] for a given species [1].
    This problem has one or two fixed points according to the value of the recombination rate beta and it exhibits
    smooth dependence on the other parameters. The statistics of the solution at t=1 are investigated considering
    uncertainties in the initial coverage rho_0 and in the reaction parameter beta. Additionally uncertainty
    in the surface absorption rate alpha can be considered to make the problem 3-dimensional.
    Gamma=0.01 denotes the desorption rate.

    .. math:: \\frac{d\\rho}{dt} = \\alpha (1 - \\rho) - \\gamma \\rho - \\beta (\\rho - 1)^2 \\rho

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
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["rho_0"] = np.linspace(0, 1, 100)
       parameters["beta"] = np.linspace(0, 20, 100)

       constants = OrderedDict()
       constants["alpha"] = 1.

       plot("SurfaceCoverageSpecies", parameters, constants)

    .. [1] Le Maitre, O.P., Najm, H.N., Ghanem, R.G., Knio, O.M., (2004).
       Multi-resolution analysis of Wiener-type uncertainty propagation schemes.
       Journal of Computational Physics, 197, 502-531.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # System of 1st order DEQ
        def deq(rho, t, alpha, beta, gamma):
            return alpha * (1. - rho) - gamma * rho - beta * (rho - 1) ** 2 * rho

        # Constants
        gamma = 0.01

        # Simulation parameters
        dt = 0.01
        t_end = 1.
        t = np.arange(0, t_end, dt)

        # Solve
        y_out = np.zeros((len(self.p["rho_0"]), 1))

        for i in range(len(y_out)):
            y = odeint(deq, self.p["rho_0"].flatten()[i], t,
                       args=(self.p["alpha"].flatten()[i], self.p["beta"].flatten()[i], gamma,))
            y_out[i, 0] = np.array([y[-1]])

        y_out = np.hstack((y_out, 2.*y_out))

        return y_out


class Franke(AbstractModel):
    """
    Franke function [1] with 2 parameters. It is often used in regression or interpolation analysis.
    It is defined in the interval [0, 1] x [0, 1]. Hampton and Doostan used in the framework of BASE-PC [2].

    .. math::
       y = \\frac{3}{4} \exp{\\left(-\\frac{(9 x_1 - 2)^2}{4} - \\frac{(9 x_2 - 2)^2}{4}\\right)} +
       \\frac{3}{4} \exp{\\left(-\\frac{(9 x_1 + 1)^2}{49} - \\frac{(9 x_2 + 1)}{10}\\right)} +
       \\frac{1}{2} \exp{\\left(-\\frac{(9 x_1 - 7)^2}{4} - \\frac{(9 x_2 - 3)^2}{4}\\right)} +
       \\frac{1}{5} \exp{\\left(-\\frac{(9 x_1 - 4)^2}{4} - (9 x_2 - 7)^2\\right)}

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
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("Franke", parameters)

    .. [1] Franke, R., (1979) A Critical Comparison of some Methods for Interpolation of Scattered Data,
       Tech. rep., DTIC Document

    .. [2] Hampton, J., Doostan, A., (2018), Basis adaptive sample efficient polynomial chaos (BASE-PC),
       Journal of Computational Physics, 371, 20-49.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
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

    .. math:: y = \\exp{\\left(2 - \sum_{i=1}^{N} x_i \\frac{\sin(i+1)}{i + 1}\\right)}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("ManufactureDecay", parameters)

    .. [1] Hampton, J., Doostan, A., (2018), Basis adaptive sample efficient polynomial chaos (BASE-PC),
       Journal of Computational Physics, 371, 20-49.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # determine sum in exponent
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += np.sin(i + 1) * self.p[key] / (i + 1.)

        # determine output
        y = np.exp(2 - s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzContinuous(AbstractModel):
    """
    N-dimensional "continuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math::  y = \\exp{\\left(- \sum_{i=1}^{N} a_i | x_i - u_i | \\right)}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzContinuous", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/cont.html
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum in exponent
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * np.abs(self.p[key] - u[i])

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzCornerPeak(AbstractModel):
    """
    N-dimensional "CornerPeak" Genz function [1,2]. It is defined in the interval [0, 1] x ... x [0, 1].
    Used by [3] as testfunction.

    .. math:: y = \\left( 1 + \sum_{i=1}^N a_i x_i\\right)^{-(N + 1)}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzCornerPeak", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/copeak.html

    .. [3] Jakeman, J. D., Eldred, M. S., & Sargsyan, K. (2015).
       Enhancing ℓ1-minimization estimates of polynomial chaos expansions using basis selection.
       Journal of Computational Physics, 289, 18-34.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # set constants
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = (1 + s) ** -(n + 1)

        y_out = y[:, np.newaxis]

        return y_out


class GenzDiscontinuous(AbstractModel):
    """
    N-dimensional "Discontinuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math:: y = \exp\\left( \sum_{i=1}^N a_i x_i\\right) \quad \mathrm{if} \quad x_i < u_i \quad \mathrm{else} \quad 0

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzDiscontinuous", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/disc.html
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        mask = np.zeros((len(self.p[list(self.p.keys())[0]]), n))

        for i, key in enumerate(self.p.keys()):
            mask[:, i] = (self.p[key] > u[i]).squeeze()
        mask = mask.any(axis=1)

        # determine sum
        s = np.zeros((np.array(self.p[list(self.p.keys())[0]]).size, 1))

        for i, key in enumerate(self.p.keys()):
            if self.p[key].ndim == 1:
                self.p[key] = self.p[key][:, np.newaxis]
            s += a[i] * self.p[key]

        # determine output
        y = np.exp(s)
        y[mask] = 0.

        return y


class GenzGaussianPeak(AbstractModel):
    """
    N-dimensional "GaussianPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math:: y = \exp\\left( - \sum_{i=1}^{N} a_i ^2 (x_i - u_i)^2\\right)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzGaussianPeak", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/gaussian.html
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] ** 2 * (self.p[key] - u[i]) ** 2

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzOscillatory(AbstractModel):
    """
    N-dimensional "Oscillatory" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math:: y = \cos \\left( 2 \pi u_1 + \sum_{i=1}^{N}a_i x_i \\right)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzOscillatory", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/oscil.html
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = np.cos(2 * np.pi * u[0] + s)

        y_out = y[:, np.newaxis]

        # y_out = np.hstack((y_out, 2*y_out))

        return y_out


class GenzProductPeak(AbstractModel):
    """
    N-dimensional "ProductPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math:: y = \prod_{i=1}^{N} \\left( a_i^{-2} + (x_i - u_i)^2 \\right)^{-1}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("GenzProductPeak", parameters)

    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.

    .. [2] https://www.sfu.ca/~ssurjano/prpeak.html
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine output
        y = np.ones(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            y *= 1 / (a[i] ** (-2) + (self.p[key] - u[i]) ** 2)

        y_out = y[:, np.newaxis]

        return y_out


class Ridge(AbstractModel):
    """
    N-dimensional "Ridge" function [1] (and also used as testfunction therein).
    Typically defined in the interval [-4, 4] x ... x [-4, 4].

    .. math::
       y = \sum_{i=1}^{N}x_i + 0.25 \\left( \sum_{i=1}^{N}x_i \\right)^2 + 0.025 \\left( \sum_{i=1}^{N}x_i \\right)^3

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in e.g. [-4, 4]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in e.g. [-4, 4]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in e.g. [-4, 4]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(-4, 4, 100)
       parameters["x2"] = np.linspace(-4, 4, 100)

       plot("Ridge", parameters)

    .. [1] Tsilifis, P., Huan, X., Safta, C., Sargsyan, K., Lacaze, G., Oefelein, J. C., Najm, H. N., Ghanem, R. G.
       (2019). Compressive sensing adaptation for polynomial chaos expansions.
       Journal of Computational Physics, 380, 29-47.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        n = len(self.p.keys())

        # determine sum
        s = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += self.p[key]

        # determine output
        y = s + 0.25 * s**2 + 0.025 * s**3

        y_out = y[:, np.newaxis]

        return y_out


class Lim2002(AbstractModel):
    """
    Two-dimensional test function of Lim et al. (2002) [1] (eq. (27)).

    This function is a polynomial in two dimensions, with terms up to degree
    5. It is nonlinear, and it is smooth despite being complex, which is
    common for computer experiment functions.

    .. math::
       y = 9 + \\frac{5}{2} x_1 - \\frac{35}{2} x_2 + \\frac{5}{2} x_1 x_2 + 19 x_2^2 -
       \\frac{15}{2} x_1^3 - \\frac{5}{2} x_1 x_2^2 - \\frac{11}{2} x_2^4 + x_1^3 x_2^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["x2"]: float or ndarray of float [n_grid]
        Second parameter defined in [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       plot("Lim2002", parameters)

    .. [1] Lim, Y. B., Sacks, J., Studden, W. J., & Welch, W. J. (2002). Design
       and analysis of computer experiments when the output is highly correlated
       over the input space. Canadian Journal of Statistics, 30(1), 109-126.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        y = 9 + 2.5 * self.p["x1"] - 17.5 * self.p["x2"] + 2.5 * self.p["x1"] * self.p["x2"] + 19 * self.p["x2"] ** 2 - 7.5 * self.p["x1"] ** 3 - 2.5 * self.p["x1"] * self.p["x2"] ** 2 - 5.5 * self.p["x2"] ** 4 + (self.p["x1"] ** 3) * (self.p["x2"] ** 2)

        # y = (9 + 5.0 / 2 * self.p["x1"] - 35.0 / 2 * self.p["x2"] + 5.0 / 2 * self.p["x1"] * self.p["x2"] +
        #      19 * self.p["x2"] ** 2 - 15.0 / 2 * self.p["x1"] ** 3 - 5.0 / 2 * self.p["x1"] * self.p["x2"] ** 2 -
        #      11.0 / 2 * self.p["x2"] ** 4 + self.p["x1"] ** 3 * self.p["x2"] ** 2)

        if type(y) is not np.ndarray:
            y = np.array([y])

        y_out = y[:, np.newaxis]

        return y_out


class Ishigami(AbstractModel):
    """
    Three-dimensional test function of Ishigami.

    The Ishigami function of Ishigami & Homma (1990) [1] is used as an example
    for uncertainty and sensitivity analysis methods, because it exhibits
    strong nonlinearity and nonmonotonicity. It also has a peculiar
    dependence on x3, as described by Sobol' & Levitan (1999) [2].
    The values of a and b used by Crestaux et al. (2007) [3] and Marrel et al. (2009) [4] are: a = 7 and b = 0.1.

    .. math:: y = \sin(x_1) + a \sin(x_2)^2 + b x_3^4 \sin(x_1)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-pi, pi]
    p["x2"]: float or ndarray of float [n_grid]
        Second parameter defined in [-pi, pi]
    p["x3"]: float or ndarray of float [n_grid]
        Third parameter defined in [-pi, pi]
    p["a"]: float
        shape parameter (a=7)
    p["b"]: float
        shape parameter (b=0.1)

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(-np.pi, np.pi, 100)
       parameters["x2"] = np.linspace(-np.pi, np.pi, 100)

       constants = OrderedDict()
       constants["a"] = 7.
       constants["b"] = 0.1
       constants["x3"] = 0.

       plot("Ishigami", parameters, constants)

    .. [1] Ishigami, T., Homma, T. (1990, December). An importance quantification
       technique in uncertainty analysis for computer models. In Uncertainty
       Modeling and Analysis, 1990. Proceedings., First International Symposium
       on (pp. 398-403). IEEE.

    .. [2] Sobol', I.M., Levitan, Y.L. (1999). On the use of variance reducing
       multipliers in Monte Carlo computations of a global sensitivity index.
       Computer Physics Communications, 117(1), 52-61.

    .. [3] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
       Polynomial chaos expansion for uncertainties quantification and sensitivity analysis [PowerPoint slides].
       Retrieved from SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.

    .. [4] Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009).
       Calculations of sobol indices for the gaussian process metamodel.
       Reliability Engineering & System Safety, 94(3), 742-751.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        if self.p["x1"] is not np.ndarray:
            self.p["x1"] = np.array(self.p["x1"])

        if self.p["x2"] is not np.ndarray:
            self.p["x2"] = np.array(self.p["x2"])

        if self.p["x3"] is not np.ndarray:
            self.p["x3"] = np.array(self.p["x3"])

        if self.p["a"] is not np.ndarray:
            self.p["a"] = np.array(self.p["a"])

        if self.p["b"] is not np.ndarray:
            self.p["b"] = np.array(self.p["b"])

        y = (np.sin(self.p["x1"].flatten()) + self.p["a"].flatten() * np.sin(self.p["x2"].flatten()) ** 2
             + self.p["b"].flatten() * self.p["x3"].flatten() ** 4 * np.sin(self.p["x1"].flatten()))

        if type(y) is not np.ndarray:
            y = np.array([y])

        y_out = y[:, np.newaxis]

        return y_out


class SphereFun(AbstractModel):
    """
    N-dimensional sphere function with zero mean.

    .. math:: y = \sum_{i=1}^{N}x_i^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [-1, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-1, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter [-1, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(-1, 1, 100)
       parameters["x2"] = np.linspace(-1, 1, 100)

       plot("SphereFun", parameters)
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # determine output
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            y += self.p[key] ** 2

        y_out = y[:, np.newaxis]

        return y_out


class GFunction(AbstractModel):
    """
    N-dimensional g-function used by Saltelli and Sobol (1995) [1].

    This test function is used as an integrand for various numerical
    estimation methods, including sensitivity analysis methods, because it
    is fairly complex, and its sensitivity indices can be expressed
    analytically. The exact value of the integral with this function as an
    integrand is 1. For each index i, a lower value of a_i indicates a higher
    importance of the input variable xi.

    .. math:: \prod_{i=1}^{N}\\frac{|4 x_i - 2| + a_i}{1 + a_i}

    The recommended values of a_i by Crestaux et al. (2007) [2] are:

    .. math:: a_i = \\frac{i-2}{2} \quad \mathrm{for\;all} \quad i=1,...,d

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter [0, 1]
    p["a"]: ndarray of float [N_dims]
        Importance factors of dimensions

    Returns
    -------
    y: ndarray of float [N_input x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)

       constants = OrderedDict()
       constants["a"] =  (np.arange(2)+1-2.)/2.

       plot("GFunction", parameters, constants)

    .. [1] Saltelli, Andrea; Sobol, I. M. (1995): Sensitivity analysis for nonlinear
       mathematical models: numerical experience. In: Mathematical models and
       computer experiment 7 (11), pp. 16-28.

    .. [2] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
       Polynomial chaos expansion for uncertainties quantification and sensitivity analysis [PowerPoint slides].
       Retrieved from SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # determine output
        y = np.ones(np.array(self.p[list(self.p.keys())[0]]).size)

        for i, key in enumerate(self.p.keys()):
            if "x" in key:
                y *= (np.abs(4.0 * self.p[key] - 2) + self.p["a"][:, i]) / (1.0 + self.p["a"][:, i])

        y_out = y[:, np.newaxis]

        return y_out


class BinaryDiscontinuousSphere(AbstractModel):
    """
    N-dimensional testfunction containing a spherical discontinuity.
    Inside the sphere the output is 2 and outside of the sphere it is 1.

    .. math::
       y = \\begin{cases}
       2, & \\text{if } \\sqrt{\\sum_{i=1}^{N}(x_i-0.5)^2} \\leq 0.25 \\\\
       1, & \\text{otherwise}
       \\end{cases}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 500)
       parameters["x2"] = np.linspace(0, 1, 500)

       plot("BinaryDiscontinuousSphere", parameters)
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        x = np.vstack([self.p[key].squeeze() for key in self.p.keys()])

        y = np.ones(x.shape[1])
        y[np.linalg.norm(x-0.5, axis=0) <= 0.25] = 2.

        y_out = y[:, np.newaxis]

        return y_out


class ContinuousDiscontinuousSphere(AbstractModel):
    """
    N-dimensional testfunction containing a spherical discontinuity.
    Inside the sphere the output corresponds to the ManufactureDecay function (shifted by -2)
    and outside of the sphere it correspond to the GenzOscillatory testfunction (scaled by 2).

    .. math::
       y = \\begin{cases}
       \\text{ManufactureDecay}(x) - 2, & \\text{if } \\sqrt{\\sum_{i=1}^{N}x_i^2} \\leq 0.25 \\\\
       \\text{GenzOscillatory}(x) * 2, & \\text{otherwise}
       \\end{cases}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 500)
       parameters["x2"] = np.linspace(0, 1, 500)

       plot("ContinuousDiscontinuousSphere", parameters)
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for key in self.p.keys():
            if self.p[key].ndim == 1:
                self.p[key] = self.p[key][:, np.newaxis]

        x = np.hstack([self.p[key] for key in self.p.keys()])

        y = np.zeros(x.shape[0])
        mask = (np.linalg.norm(x-0.5, axis=1) <= 0.25).flatten()

        p_1 = OrderedDict()
        p_2 = OrderedDict()

        for i, key in enumerate(self.p.keys()):
            p_1[key] = x[mask, i]
            p_2[key] = x[np.logical_not(mask), i]

        model_1 = ManufactureDecay().set_parameters(p_1)
        model_2 = GenzOscillatory().set_parameters(p_2)

        y_1 = model_1.simulate() - 2
        y_2 = model_2.simulate() * 2

        y[mask] = y_1.flatten()
        y[np.logical_not(mask)] = y_2.flatten()

        y_out = y[:, np.newaxis]

        return y_out


class DiscontinuousRidgeManufactureDecay(AbstractModel):
    """
    N-dimensional testfunction containing a linear discontinuity.
    On the one side the output corresponds to the Ridge function
    and on the other side it correspond to the ManufactureDecay testfunction.

    .. math::
       y = \\begin{cases}
       \\text{ManufactureDecay}(x), & \\text{if } \\sum_{i=1}^{N}x_i \\leq 1 \\\\
       \\text{Ridge}(x), & \\text{otherwise}
       \\end{cases}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter [0, 1]

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. plot::

       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict

       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 250)
       parameters["x2"] = np.linspace(0, 1, 250)

       plot("DiscontinuousRidgeManufactureDecay", parameters)
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for key in self.p.keys():
            if self.p[key].ndim == 1:
                self.p[key] = self.p[key][:, np.newaxis]

        x = np.hstack([self.p[key] for key in self.p.keys()])

        y = np.zeros(x.shape[0])
        mask = (np.sum(x, axis=1) <= 1.).flatten()

        p_1 = OrderedDict()
        p_2 = OrderedDict()

        for i, key in enumerate(self.p.keys()):
            p_1[key] = x[mask, i]
            p_2[key] = x[np.logical_not(mask), i]

        model_1 = ManufactureDecay().set_parameters(p_1)
        model_2 = Ridge().set_parameters(p_2)

        y_1 = model_1.simulate()
        y_2 = model_2.simulate()

        y[mask] = y_1.flatten()
        y[np.logical_not(mask)] = y_2.flatten()

        y_out = y[:, np.newaxis]

        return y_out


class OakleyOhagan2004(AbstractModel):
    """
    15-dimensional test function of Oakley and O'Hagan (2004) [1].

    This function's a-coefficients are chosen so that 5 of the input
    variables contribute significantly to the output variance, 5 have a
    much smaller effect, and the remaining 5 have almost no effect on the
    output variance.

    .. math::
       y = \mathbf{a}_1^T\mathbf{x} + \mathbf{a}_2^T \sin(\mathbf{x}) + \mathbf{a}_3^T \cos(\mathbf{x}) +
       \mathbf{x}^T\mathbf{M}\mathbf{x}

    The parameter vectors a and matrix M are in /pygpc/pck/data/oakley_ohagan_2004.

    Parameters
    ----------
    p["x1...15"]: ndarray of float [n_grid]
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

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # load coefficients
        folder = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0],
                              "pckg", "data", "oakley_ohagan_2004")
        m = np.loadtxt(os.path.join(folder, "oakley_ohagan_2004_M.txt"))
        a1 = np.loadtxt(os.path.join(folder, "oakley_ohagan_2004_a1.txt"))
        a2 = np.loadtxt(os.path.join(folder, "oakley_ohagan_2004_a2.txt"))
        a3 = np.loadtxt(os.path.join(folder, "oakley_ohagan_2004_a3.txt"))

        x = np.zeros((self.p[list(self.p.keys())[0]].size, 15))

        for i, key in enumerate(self.p.keys()):
            x[:, i] = self.p[key]

        # function
        y = (np.dot(x, a1) + np.dot(np.sin(x), a2)
             + np.dot(np.cos(x), a3) + np.sum(np.multiply(np.dot(x, m), x), axis=1))

        y_out = y[:, np.newaxis]

        return y_out


class Welch1992(AbstractModel):
    """
    20-dimensional test function of Welch et al. (1992) [1].

    For input variable screening purposes, it can be found that some input
    variables of this function have a very high effect on the output,
    compared to other input variables. As Welch et al. (1992) [1] point out,
    interactions and nonlinear effects make this function challenging.

    .. math::
       y = \\frac{5 x_{12}}{1 + x_1} + 5 (x_4 - x_{20})^2 + x_5 + 40 x_{19}^3 + 5 x_{19} + 0.05 x_2
       + 0.08 x_3 - 0.03 x_6 + 0.03 x_7 - 0.09 x_9 - 0.01 x_{10} -

       0.07 x_{11} + 0.25 x_{13}^2 - 0.04 x_{14} + 0.06 x_{15} - 0.01 x_{17} - 0.03 x_{18}

    Parameters
    ----------
    p["x1...x20"]: float
        Input data, xi ~ U(-0.5, 0.5), for all i = 1,..., 20.

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. [1] Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., Morris, M. D. (1992).
       Screening, predicting, and computer experiments. Technometrics, 34(1), 15-25.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        y = (5.0 * self.p["x12"] / (1 + self.p["x1"]) + 5 * (self.p["x4"] - self.p["x20"]) ** 2
             + self.p["x5"] + 40 * self.p["x19"] ** 3 + 5 * self.p["x19"] + 0.05 * self.p["x2"]
             + 0.08 * self.p["x3"] - 0.03 * self.p["x6"] + 0.03 * self.p["x7"]
             - 0.09 * self.p["x9"] - 0.01 * self.p["x10"] - 0.07 * self.p["x11"]
             + 0.25 * self.p["x13"] ** 2 - 0.04 * self.p["x14"]
             + 0.06 * self.p["x15"] - 0.01 * self.p["x17"] - 0.03 * self.p["x18"])

        y_out = y[:, np.newaxis]

        return y_out


class WingWeight(AbstractModel):
    """
    10-dimensional test function which models a light aircraft wing from Forrester et al. (2008) [1]

    .. math::
       y = \\frac{0.036 x_1^{0.758} x_2^{0.0035} x_3}{\cos(x_4)^2)^{0.6}} x_5^{0.006} x_6^{0.04}
       \\left( \\frac{100 x_7}{\cos(x_4)}\\right)^{-0.3} (x_8 x_9)^{0.49} + x_1 x_{10}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        x1(Sw) [150, 200]
    p["x2"]: float or ndarray of float [n_grid]
        x2(Wfw) [220, 300]
    p["x3"]: float or ndarray of float [n_grid]
        x3(A) [6, 10]
    p["x4"]: float or ndarray of float [n_grid]
        x4(Lambda) [-10, 10]
    p["x5"]: float or ndarray of float [n_grid]
        x5(q) [16, 45]
    p["x6"]: float or ndarray of float [n_grid]
        x6(lambda) [0.5, 1]
    p["x7"]: float or ndarray of float [n_grid]
        x7(tc) [0.08, 0.18]
    p["x8"]: float or ndarray of float [n_grid]
        x8(Nz) [2.5, 6]
    p["x9"]: float or ndarray of float [n_grid]
        x9(Wdg) [1700, 2500]
    p["x10"]: float or ndarray of float [n_grid]
        x10(Wp) [0.025, 0.08]

    Returns
    -------
    y: float or ndarray of float [n_grid x 1]
        Output data

    Notes
    -----
    .. [1] Forrester, A., Sobester, A., & Keane, A. (2008).
       Engineering design via surrogate modelling: a practical guide. John Wiley & Sons.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        y = 0.036 * self.p["x1"] ** 0.758 * self.p["x2"] ** 0.0035 * \
             (self.p["x3"] / np.cos(self.p["x4"]) ** 2) ** 0.6 * \
             self.p["x5"] ** 0.006 * self.p["x6"] ** 0.04 * \
             (100 * self.p["x7"] / np.cos(self.p["x4"]))**(-0.3) * \
             (self.p["x8"] * self.p["x9"])**0.49 + self.p["x1"] * self.p["x10"]

        y_out = y[:, np.newaxis]

        return y_out


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
    potential: ndarray of float [1 x n_out]
        Values of the electric potential, in (V)

    Notes
    -----
    .. [1] Rush, S., & Driscoll, D. A. (1969). EEG electrode sensitivity-an application of reciprocity.
       IEEE transactions on biomedical engineering, (1), 15-22.
    """
    def __init__(self):
        self.nbr_polynomials = 50

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

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

        cathode_pos = (np.sqrt(self.p["cathode_pos"][0] ** 2 +
                               self.p["cathode_pos"][1] ** 2 +
                               self.p["cathode_pos"][2] ** 2) * 1e-3,
                       np.arccos(self.p["cathode_pos"][2] /
                                 np.sqrt(self.p["cathode_pos"][0] ** 2 +
                                         self.p["cathode_pos"][1] ** 2 +
                                         self.p["cathode_pos"][2] ** 2)),
                       np.arctan2(self.p["cathode_pos"][1], self.p["cathode_pos"][0]))

        anode_pos = (
            np.sqrt(self.p["anode_pos"][0] ** 2 + self.p["anode_pos"][1] ** 2 + self.p["anode_pos"][2] ** 2) * 1e-3,
            np.arccos(self.p["anode_pos"][2] /
                      np.sqrt(self.p["anode_pos"][0] ** 2 +
                              self.p["anode_pos"][1] ** 2 +
                              self.p["anode_pos"][2] ** 2)),
            np.arctan2(self.p["anode_pos"][1], self.p["anode_pos"][0]))

        def a(n):
            return ((2 * n + 1) ** 3 / (2 * n)) / (((b_over_s + 1) * n + 1) * ((s_over_t + 1) * n + 1) +
                                                   (b_over_s - 1) * (s_over_t - 1) * n * (n + 1) *
                                                   (radius_brain / radius_skull) ** (2 * n + 1) +
                                                   (s_over_t - 1) * (n + 1) * ((b_over_s + 1) * n + 1) *
                                                   (radius_skull / radius_skin) ** (2 * n + 1) +
                                                   (b_over_s - 1) * (n + 1) * ((s_over_t + 1) * (n + 1) - 1) *
                                                   (radius_brain / radius_skin) ** (2 * n + 1))

        # All of the bellow is modified: division by radius_skin moved to the
        # coefficients calculations due to numerical constraints
        # THIS IS DIFFERENT FROM THE PAPER (there's a sum instead of difference)
        def s(n):
            return (a(n)) * ((1 + b_over_s) * n + 1) / (2 * n + 1)

        def u(n):
            return (a(n) * radius_skin) * n * (1 - b_over_s) * \
                   radius_brain ** (2 * n + 1) / (2 * n + 1)

        def t(n):
            return (a(n) / ((2 * n + 1) ** 2)) * \
                   (((1 + b_over_s) * n + 1) * ((1 + s_over_t) * n + 1) +
                    n * (n + 1) * (1 - b_over_s) * (1 - s_over_t) * (radius_brain / radius_skull) ** (2 * n + 1))

        def w(n):
            return ((n * a(n) * radius_skin) / ((2 * n + 1) ** 2)) * \
                   ((1 - s_over_t) * ((1 + b_over_s) * n + 1) * radius_skull ** (2 * n + 1) +
                    (1 - b_over_s) * ((1 + s_over_t) * n + s_over_t) * radius_brain ** (2 * n + 1))

        brain_region = np.where(p_r[:, 0] <= radius_brain)[0]
        skull_region = np.where(
            (p_r[:, 0] > radius_brain) * (p_r[:, 0] <= radius_skull))[0]
        skin_region = np.where((p_r[:, 0] > radius_skull)
                               * (p_r[:, 0] <= radius_skin))[0]
        inside_sphere = np.where((p_r[:, 0] <= radius_skin))[0]
        outside_sphere = np.where((p_r[:, 0] > radius_skin))[0]

        cos_theta_a = np.cos(cathode_pos[1]) * np.cos(p_r[:, 1]) + \
                      np.sin(cathode_pos[1]) * np.sin(p_r[:, 1]) * \
                      np.cos(p_r[:, 2] - cathode_pos[2])
        cos_theta_b = np.cos(anode_pos[1]) * np.cos(p_r[:, 1]) + \
                      np.sin(anode_pos[1]) * np.sin(p_r[:, 1]) * \
                      np.cos(p_r[:, 2] - anode_pos[2])

        potentials = np.zeros((self.p["points"].shape[0]), dtype='float64')

        coefficients = np.zeros((self.nbr_polynomials, self.p["points"].shape[0]), dtype='float64')

        # accelerate
        for ii in range(1, self.nbr_polynomials):
            ni = float(ii)
            coefficients[ii, brain_region] = np.nan_to_num(
                a(ni) * ((p_r[brain_region, 0] / radius_skin) ** ni))

            coefficients[ii, skull_region] = np.nan_to_num(s(ni) * (p_r[skull_region, 0] / radius_skin) ** ni +
                                                           u(ni) * (p_r[skull_region, 0] * radius_skin) ** (-ni - 1))

            coefficients[ii, skin_region] = np.nan_to_num(t(ni) * (p_r[skin_region, 0] / radius_skin) ** ni
                                                          + w(ni) * (p_r[skin_region, 0] * radius_skin) ** (-ni - 1))

        potentials[inside_sphere] = np.nan_to_num(
            np.polynomial.legendre.legval(cos_theta_a[inside_sphere], coefficients[:, inside_sphere], tensor=False) -
            np.polynomial.legendre.legval(cos_theta_b[inside_sphere], coefficients[:, inside_sphere], tensor=False))

        potentials *= 1.0 / (2 * np.pi * self.p["sigma_3"] * radius_skin)

        potentials[outside_sphere] = 0.0

        potentials = potentials[np.newaxis, :]

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
    potential: ndarray of float [1 x n_out]
       Potential at the points

    Notes
    -----
    .. [1] Yao, D. (2000). Electric potential produced by a dipole in a homogeneous conducting sphere.
       IEEE Transactions on Biomedical Engineering, 47(7), 964-966.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

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

        potential = potential[np.newaxis, :]

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
    B: ndarray of float [1 x 3*N]
        B-fields in detector positions

    Notes
    -----
    .. [1] Sarvas, J. (1987). Basic mathematical and electromagnetic concepts of the biomagnetic inverse problem.
       Physics in Medicine & Biology, 32(1), 11.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
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

        b = b.flatten()[np.newaxis, :]

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
    E: ndarray of float [1 x 3*N]
        E-fields at detector positions

    Notes
    -----
    .. [1] Heller, L., & van Hulsteyn, D. B. (1992). Brain stimulation using electromagnetic sources:
       theoretical aspects. Biophysical Journal, 63(1), 129-138.
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

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

        e = e.flatten()[np.newaxis, :]

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
    potential: ndarray of float [1 x n_out]
        Values of the electric potential, in (V)

    Notes
    -----
    .. [1] Ary, J. P., Klein, S. A., & Fender, D. H. (1981). Location of sources of evoked scalp potentials:
       corrections for skull and scalp thicknesses. IEEE Transactions on Biomedical Engineering, (6), 447-452.
       eq. 2 and 2a
    """

    def __init__(self):
        pass

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

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

        potential = potential[np.newaxis, :]

        return potential
