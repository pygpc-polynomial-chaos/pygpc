import os
import copy
import inspect
import warnings
import numpy as np
import scipy.special
from scipy.integrate import odeint
from collections import OrderedDict
from pygpc.AbstractModel import AbstractModel
# from pygpc.testfunctions.euber_libV3 import*

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass


def plot_testfunction(testfunction_name: object, parameters: object, constants: object = None, output_idx: object = 0,
                      plot_3d=True) -> object:
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
    plot_3d : bool, optional, default: True
        Plot function in 3d or 2d

    Returns
    -------
    <plot> : matplotlib figure
        Plot showing the QoI of the testfunction in 1D or 2D
    """

    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 12}

    matplotlib.rc('font', **font)

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

    fig = plt.figure(figsize=(6, ((n_qoi-1)*0.85+1) * 5))

    for i, o in enumerate(output_idx):
        if plot_3d:
            ax = fig.add_subplot(n_qoi, 1, i+1, projection='3d')
        else:
            ax = fig.add_subplot(n_qoi, 1, i+1)

        # omit "additional_data" if present
        if type(y) is tuple:
            y = y[0]

        if len(p_names) == 2:
            if plot_3d:
                im = ax.plot_surface(x1,
                                     x2,
                                     np.reshape(y[:, o],
                                                (len(parameters[p_names[1]]), len(parameters[p_names[0]])),
                                                order='c'),
                                     cmap="jet")
            else:
                im = ax.pcolor(x1,
                               x2,
                               np.reshape(y[:, o],
                                          (len(parameters[p_names[1]]), len(parameters[p_names[0]])),
                                          order='c'),
                               cmap="jet")

            ax.set_ylabel(r"${}$".format(p_names[1]), fontsize=13)
            fig.colorbar(im, ax=ax, orientation='vertical')

        else:
            ax.plot(parameters[p_names[0]], y)
            ax.set_ylabel(r"$y({})$".format(p_names[0]), fontsize=13)

        ax.set_xlabel(r"${}$".format(p_names[0]), fontsize=13)

    ax.set_title("{} function".format(model.__class__.__name__))
    plt.tight_layout()
    plt.show()


class Dummy(AbstractModel):
    """
    Dummy model

    Parameters
    ----------
    p["..."]: float or ndarray of float [n_grid]
        Any first parameter
    p["..."]: float or ndarray of float [n_grid]
        Any second parameter

    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Any output
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        pass


class Ackley(AbstractModel):
    """
    N-dimensional Ackley function [1][2][3][4].
    The Ackley function is widely used for testing optimization algorithms.
    In its two-dimensional form, as shown in the plot above, it is characterized
    by a nearly flat outer region, and a large hole at the centre.
    The function poses a risk for optimization algorithms, particularly
    hillclimbing algorithms, to be trapped in one of its many local minima.

    Recommended variable values are: a = 20, b = 0.2 and c = 0.5*pi.

    .. math::
       y = -a\\exp{\\left(-b\\sqrt{\\frac{1}{d}\\sum_{i=1}^{N} x_i^2}\\right)} -
       \\exp{\\left(\\frac{1}{d}\\sum_{i=1}^{N} \\cos{(cx_i)}\\right)} + a + \\exp{(1)}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-32.768, 32.768]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-32.768, 32.768]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [-32.768, 32.768]

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
       parameters["x1"] = np.linspace(-32.768, 32.768, 100)
       parameters["x2"] = np.linspace(-32.768, 32.768, 100)

       constants = OrderedDict()
       constants["a"] = 20.
       constants["b"] = 0.2
       constants["c"] = 0.5*np.pi

       plot("Ackley", parameters, constants, plot_3d=False)

    .. [1] Adorio, E. P., & Diliman, U. P. MVF - Multivariate Test Functions Library in C
       for Unconstrained Global Optimization (2005). Retrieved June 2013,
       from http://http://www.geocities.ws/eadorio/mvf.pdf

    .. [2] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf

    .. [3] Back, T. (1996). Evolutionary algorithms in theory and practice: evolution strategies,
       evolutionary programming, genetic algorithms. Oxford University Press on Demand

    .. [4] https://www.sfu.ca/~ssurjano/ackley.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # set constants
        p = copy.deepcopy(self.p)
        a = self.p["a"]
        b = self.p["b"]
        c = self.p["c"]
        del p["a"], p["b"], p["c"]

        n = len(p.keys())

        # determine sum in exponent
        s1 = np.zeros(np.array(p[list(p.keys())[0]]).size)
        s2 = np.zeros(np.array(p[list(p.keys())[0]]).size)

        for i, key in enumerate(p.keys()):
            s1 += p[key]**2
            s2 += np.cos(c*p[key])

        s1 = -a * np.exp(-b * np.sqrt(1/n * s1))
        s2 = np.exp(1/n * s2)

        # determine output
        y = s1 - s2 + a + np.exp(1)

        y_out = y[:, np.newaxis]

        return y_out


class BukinFunctionNumber6(AbstractModel):
    """
    2-dimensional Bukin Function N. 6 [1][2].
    The sixth Bukin Function has many local minima, all of which lie in a ridge.

    .. math::
       y = 100\\sqrt{\\mid x_2 - 0.01 x_1^2 \\mid} + 0.01\\mid x_1 + 10 \\mid

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-15, -5]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-3, 3]

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
       parameters["x1"] = np.linspace(-15, -5, 100)
       parameters["x2"] = np.linspace(-3, 3, 100)

       constants = None

       plot("BukinFunctionNumber6", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Functions Index. Retrieved June 2013, from
       http://infinity77.net/global_optimization/test_functions.html#test-functions-index.

    .. [2] https://www.sfu.ca/~ssurjano/bukin6.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = 100 * np.sqrt(abs(self.p["x2"] - 0.01 * self.p["x1"] ** 2)) + 0.01 * abs(self.p["x1"] + 10)

        y_out = y[:, np.newaxis]

        return y_out


class CrossinTrayFunction(AbstractModel):
    """
    2-dimensional Cross-in-Tray Function [1][2].
    The Cross-in-Tray function has multiple global minima.
    It is shown here with a smaller domain in the second plot,
    so that its characteristic "cross" will be visible.

    .. math::
      y = -0.0001\\left(\\mid\\sin(x_1)\\sin(x_2)\\exp\\left(\\mid 100-\\frac{\\sqrt{x_1^2 + x_2^2}}{\\pi}\\mid\\right)\\mid + 1\\right)^{0.1}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-10, 10]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-10, 10]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-10, 10]

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
       parameters["x1"] = np.linspace(-10, 10, 100)
       parameters["x2"] = np.linspace(-10, 10, 100)

       constants = None

       plot("CrossinTrayFunction", parameters, constants, plot_3d=False)

    .. [1] Test functions for optimization. In Wikipedia. Retrieved June 2013, from
       https://en.wikipedia.org/wiki/Test_functions_for_optimization.
    .. [2] https://www.sfu.ca/~ssurjano/crossit.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = -0.0001 * (np.abs(np.sin(self.p["x1"]) * np.sin(self.p["x2"]) *
                              np.exp(np.abs(100 - (np.sqrt(self.p["x1"]**2 + self.p["x2"]**2)) / np.pi))) + 1) **0.1

        y_out = y[:, np.newaxis]

        return y_out


class BohachevskyFunction1(AbstractModel):
    """
    2-dimensional first Bohachevsky function [1][2].
    The Bohachevsky functions all have the same similar bowl shape. The one shown above is the first function.

    .. math::
     y = x_1^2 + 2x_2^2 - 0.3\\cos(3\\pi x_1) - 0.4\\cos(4\\pi x_2) + 0.7

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-100, 100]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-100, 100]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-100, 100]

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
       parameters["x1"] = np.linspace(-100, 100 , 100)
       parameters["x2"] = np.linspace(-100, 100 , 100)

       constants = None

       plot("BohachevskyFunction1", parameters, constants, plot_3d=False)

    .. [1] Global Optimization x_1 **2 + 2x_2 **2 - 0.3 * cos(3pi * x_1) - 0.4 * cos(4pi * x_2) + 0.7 Test Problems.
       Retrieved June 2013, from http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.
    .. [2] https://www.sfu.ca/~ssurjano/boha.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = (self.p["x1"] **2) + (self.p["x2"] **2) - 0.3 * np.cos(3 * np.pi * self.p["x1"])\
            - 0.4 * np.cos(4 * np.pi * self.p["x2"]) + 0.7

        y_out = y[:, np.newaxis]

        return y_out


class PermFunction(AbstractModel):
    """
    d-dimensional Perm function 0, D, b (beta) [1][2]. The parameter b (beta) is often assumed to be b=10.

    .. math::
      y = \\sum_{i=1}^{d}\\left(\\sum_{j=1}^{d}(j + \\beta)\\left(x_j^i-\\frac{1}{j^i}\\right)\\right)^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-d, d]
    p["xi"]: float or ndarray of float [n_grid]
        i parameter defined in [-d, d]
    p["xj"]: float or ndarray of float [n_grid]
        j parameter defined in [-d, d]

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
       parameters["x1"] = np.linspace(-2, 2, 100)
       parameters["x2"] = np.linspace(-2, 2, 100)

       constants = OrderedDict()
       constants["b"] = 10

       plot("PermFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.
    .. [2] https://www.sfu.ca/~ssurjano/perm0db.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # set constants
        p = copy.deepcopy(self.p)
        b = self.p["b"]
        del p["b"]

        n = len(p.keys())

        # determine sum
        y = np.zeros(np.array(p[list(p.keys())[0]]).size)
        tmp = np.zeros(np.array(p[list(p.keys())[0]]).size)

        for i, i_key in enumerate(p.keys()):
            for j, j_key in enumerate(p.keys()):
                tmp += (j+1+b)*(p[j_key]**(i+1)-1/((j+1)**(i+1)))
            y += tmp**2
            tmp = np.zeros(np.array(p[list(p.keys())[0]]).size)

        y_out = y[:, np.newaxis]

        return y_out


class SixHumpCamelFunction(AbstractModel):
    """
    2-dimensional Six - Hump Camel function [1][2].
    The plot on the left shows the six-hump Camel function on its recommended input domain,
    and the plot on the right shows only a portion of this domain,
    to allow for easier viewing of the function's key characteristics.
    The function has six local minima, two of which are global.

    .. math::
      y = \\left(4 - 2.1x_1^2 + \\frac{x_1^4}{3}\\right)x_1^2 + x_1x_2 + (-4 + 4x_2^2)x_2^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-3, 3]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-2, 2]

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
       parameters["x1"] = np.linspace(-3, 3 , 100)
       parameters["x2"] = np.linspace(-2, 2 , 100)

       constants = None

       plot("SixHumpCamelFunction", parameters, constants, plot_3d=False)

    .. [1] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005). Retrieved June 2013, from
       http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    .. [2] https://www.sfu.ca/~ssurjano/camel6.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = (4 - 2.1*(self.p["x1"] **2) + (self.p["x1"] **4)/3) * (self.p["x1"] **2) + self.p["x1"] * self.p["x2"] + \
            (-4 + 4 * (self.p["x2"] **2)) * (self.p["x2"] **2)

        y_out = y[:, np.newaxis]

        return y_out


class RotatedHyperEllipsoid(AbstractModel):
    """
    d-dimensional Rotated-Hyper Ellipsoid Function [1][2].
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum Squares function.
    The plot shows its two-dimensional form.

    .. math::
       y = \\sum_{i=1}^{d}\\sum_{j=1}^{i}\\* x_j^2

    Parameters
    ----------
    p["xi"]: float or ndarray of float [n_grid]
        i parameter defined in [-65.536, 65.536]
    p["xj"]: float or ndarray of float [n_grid]
        j parameter defined in [-65.536, 65.536]

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
       parameters["x1"] = np.linspace(-65.536, 65.536, 100)
       parameters["x2"] = np.linspace(-65.536, 65.536, 100)

       constants = None

       plot("RotatedHyperEllipsoid", parameters, constants, plot_3d=False)

    .. [1] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    .. [2] https://www.sfu.ca/~ssurjano/rothyp.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # determine sum
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, i_key in enumerate(keys):
            for j, j_key in enumerate(keys[0:i+1]):
               y += self.p[j_key]**2

        y_out = y[:, np.newaxis]

        return y_out


class SumOfDifferentPowersFunction(AbstractModel):
    """
    d-dimensional Sum Of Different Powers Function [1][2].
    The Sum of Different Powers function is unimodal. It is shown here in its two-dimensional form.

    .. math::
         y = \\sum_{i=1}^{d}\\mid x_i \\mid ^{i+1}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-1, 1]
    p["xj"]: float or ndarray of float [n_grid]
        j-th parameter defined in [-1, 1]

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
       parameters["x1"] = np.linspace(-1, 1, 100)
       parameters["x2"] = np.linspace(-1, 1, 100)

       constants = None

       plot("SumOfDifferentPowersFunction", parameters, constants, plot_3d=False)

    .. [1] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    .. [2] https://www.sfu.ca/~ssurjano/sumpow.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # determine sum
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, i_key in enumerate(keys):
               y += np.abs(self.p[i_key] **(i+2))

        y_out = y[:, np.newaxis]

        return y_out


class ZakharovFunction(AbstractModel):
    """
    d-dimensional Zakharov Function [1][2].
    The Zakharov function has no local minima except the global one. It is shown here in its two-dimensional form.

    .. math::
       y = \\sum_{i=1}^{d} x_i^2+\\left(\\sum_{i+1}^{d}0.5ix_i\\right)^2+\\left(\\sum_{i+1}^d0.5ix_i\\right)^4

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-5, 10]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-5, 10]

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
       parameters["x1"] = np.linspace(-5, 10, 100)
       parameters["xi"] = np.linspace(-5, 10, 100)

       constants = None

       plot("ZakharovFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.

    .. [2] https://www.sfu.ca/~ssurjano/zakharov.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
               self.p[key] = self.p[key].flatten()

        # determine sum
        s1 = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        s2 = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, key in enumerate(keys):
            s1 += (self.p[key] ** 2)
            s2 += 0.5 * (i+1) * self.p[key]

        # determine output
        y = s1 + s2 ** 2 + s2 ** 4

        y_out = y[:, np.newaxis]

        return y_out


class DropWaveFunction(AbstractModel):
    """
    2-dimensional DropWaveFunction [1][2].
    The Drop-Wave function is multimodal and highly complex.
    The second plot above shows the function on a smaller input domain, to illustrate its characteristic features.

    .. math::
      y = -\\frac{1+\\cos\\left(12\\sqrt{x_1^2+x_2^2}\\right)}{0.5(x_1^2+x_2^2)+2}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-5.12, 5.12]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-5.12, 5.12]

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
       parameters["x1"] = np.linspace(-5.12, 5.12, 100)
       parameters["x2"] = np.linspace(-5.12, 5.12, 100)

       constants = None

       plot("DropWaveFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Functions Index.
       Retrieved June 2013, from http://infinity77.net/global_optimization/test_functions.html#test-functions-index.

    .. [2] https://www.sfu.ca/~ssurjano/drop.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = - (1 + np.cos(12 * np.sqrt(self.p["x1"] ** 2 + self.p["x2"] ** 2))) \
            / (0.5 * (self.p["x1"] ** 2 + self.p["x2"] ** 2) + 2)

        y_out = y[:, np.newaxis]

        return y_out


class DixonPriceFunction(AbstractModel):
    """
    d-dimensional Dixon-Price Function [1][2].

    .. math::
      y = (x_1-1)^2+\\sum_{i=d}^{2}(2x_{i}^{2}-x_{i-1})^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-10, 10]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-10, 10]

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
       parameters["x1"] = np.linspace(-10, 10, 100)
       parameters["x2"] = np.linspace(-10, 10, 100)

       constants = None

       plot("DixonPriceFunction ", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Functions Index.
       Retrieved June 2013, from http://infinity77.net/global_optimization/test_functions.html#test-functions-index.

    .. [2] https://www.sfu.ca/~ssurjano/dixonpr.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
               self.p[key] = self.p[key].flatten()

        s1 = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, key in enumerate(keys[1:]):
            s1 += (i+2) * (2 * self.p[key] ** 2 - self.p[keys[i]]) ** 2

        # determine output
        y = (self.p[keys[0]] - 1) ** 2 + s1

        y_out = y[:, np.newaxis]

        return y_out


class RosenbrockFunction(AbstractModel):
    """
    d-dimensional Rosenbrock Function [1][2][3][4][5].
    The Rosenbrock function, also referred to as the Valley or Banana function,
    is a popular test problem for gradient-based optimization algorithms.
    It is shown in the plot above in its two-dimensional form.
    The function is unimodal, and the global minimum lies in a narrow, parabolic valley.
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    .. math::
      y = \\sum_{i=1}^{d-1}[100(x_{i+1}-x_i^2)^2+(x_i-1)^2]

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-5, 10]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [-5, 10]

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
       parameters["x1"] = np.linspace(-5, 10, 100)
       parameters["x2"] = np.linspace(-5, 10, 100)

       constants = None

       plot("RosenbrockFunction ", parameters, constants, plot_3d=False)

    .. [1] Dixon, L. C. W., & Szego, G. P. (1978). The global optimization problem: an introduction.
       Towards global optimization, 2, 1-15.

    .. [2] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.

    .. [3] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

    .. [4] Picheny, V., Wagner, T., & Ginsbourger, D. (2012).
       A benchmark of kriging-based infill criteria for noisy optimization.

    .. [5] https://www.sfu.ca/~ssurjano/rosen.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
               self.p[key] = self.p[key].flatten()

        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, key in enumerate(keys[:-1]):
            y += 100 * (self.p[keys[i+1]] - self.p[key] ** 2) ** 2 + (self.p[key] - 1) ** 2

        y_out = y[:, np.newaxis]

        return y_out


class MichalewiczFunction(AbstractModel):
    """
    d-dimensional Michalewicz function [1][2][3][4].
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges; a larger m leads to a more difficult search.
    The recommended value of m is m = 10. The function's two-dimensional form is shown in the plot above.

    .. math::
       y=-\\sum_{i=1}^{d}\\sin(x_i)\\sin^{2m}\\left(\\frac{i x_i^2}{\\pi}\\right)

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, np.pi]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, np.pi]

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
       parameters["x1"] = np.linspace(0, np.pi, 100)
       parameters["xi"] = np.linspace(0, np.pi, 100)

       constants = OrderedDict()
       constants["m"] = 10.

       plot("MichalewiczFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Functions Index.
       Retrieved June 2013, from http://infinity77.net/global_optimization/test_functions.html#test-functions-index.

    .. [2]Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.

    .. [3] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.

    .. [4] https://www.sfu.ca/~ssurjano/michal.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
               self.p[key] = self.p[key].flatten()

        # set constants
        p = copy.deepcopy(self.p)
        m = self.p["m"]
        del p["m"]

        # determine sum
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, key in enumerate(keys):
            y += - np.sin(self.p[key]) * (np.sin(self.p[keys[i]] * (self.p[key] ** 2) / np.pi)) ** (2 * m)

        y_out = y[:, np.newaxis]

        return y_out


class DeJongFunctionFive(AbstractModel):
    """
    2-dimensional DeJong Function Number Five [1][2].
    The fifth function of De Jong is multimodal, with very sharp drops on a mainly flat surface.

    .. math::
      y = \\left(0.002+\\sum_{i+1}^{25}\\frac{1}{i+(x_1-a_{1i})^6+(x_2-a_{2i})^6}\\right)^{-1}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-65.536, 65.536]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-65.536, 65.536]

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
       parameters["x1"] = np.linspace(-65.536, 65.536, 100)
       parameters["x2"] = np.linspace(-65.536, 65.536, 100)

       constants = None

       plot("DeJongFunctionFive", parameters, constants, plot_3d=False)

    .. [1] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).
       Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    .. [2] https://www.sfu.ca/~ssurjano/dejong5.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
               self.p[key] = self.p[key].flatten()

        # set constants
        seq = [-32, -16, 0, 16, 32]
        a = np.array([seq * 5, np.hstack([[i] * 5 for i in seq])])
        s1 = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        # determine sum
        for i in range(25):
            s1 += 1 / (i+1 + (self.p[keys[0]] - a[0, i]) ** 6 + (self.p[keys[1]] - a[1, i]) ** 6)

        y = 1 / (0.002 + s1)

        y_out = y[:, np.newaxis]

        return y_out


class MatyasFunction(AbstractModel):
    """
    2-dimensional Matyas function [1][2].
    The Matyas function has no local minima except the global one.

    .. math::
      y = 0.26(x_1^2+x_2^2)-0.48x_1x_2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-10, 10]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-10, 10]

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
       parameters["x1"] = np.linspace(-10, 10, 100)
       parameters["x2"] = np.linspace(-10, 10, 100)

       constants = None

       plot("MatyasFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.

    .. [2] https://www.sfu.ca/~ssurjano/matya.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = 0.26 * (self.p["x1"] ** 2 + self.p["x2"] ** 2) - 0.48 * self.p["x1"] * self.p["x2"]

        y_out = y[:, np.newaxis]

        return y_out


class GramacyLeeFunction(AbstractModel):
    """
    1-dimensional Gramacy and Lee function[1][2][3].
    This is a simple one-dimensional test function.

    .. math::
      y = \\frac{\\sin(10 \\pi x)}{2x}+(x-1)^4

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0.5, 2.5]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [0.5, 2.5]

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
       parameters["x1"] = np.linspace(0.5, 2.5, 100)
       parameters["x2"] = np.linspace(0.5, 2.5, 100)

       constants = None

       plot("GramacyLeeFunction", parameters, constants, plot_3d=False)

    .. [1] Gramacy, R. B., & Lee, H. K. (2012). Cases for the nugget in modeling computer experiments.
       Statistics and Computing, 22(3), 713-722.

    .. [2] Ranjan, P. (2013). Comment: EI Criteria for Noisy Computer Simulators. Technometrics, 55(1), 24-28.

    .. [3] https://www.sfu.ca/~ssurjano/grlee12.html

    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = (np.sin(10 * np.pi * self.p["x1"]) / 2 * self.p["x1"]) + (self.p["x1"] - 1) ** 4

        y_out = y[:, np.newaxis]

        return y_out


class SchafferFunction4(AbstractModel):
    """
    2-dimensional Schaffer function No. 4. [1][2].

    .. math::
      y = 0.5+\\frac{\\cos{(\\sin{(\mid x_1^2-x_2^2 \mid )})}-0.5}{(1+0.001(x_1^2+x_2^2))^2}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-100, 100]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-100, 100]

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
       parameters["x1"] = np.linspace(-100, 100, 100)
       parameters["x2"] = np.linspace(-100, 100, 100)

       constants = None

       plot("SchafferFunction4", parameters, constants, plot_3d=False)

    .. [1] Test functions for optimization. In Wikipedia.
       Retrieved June 2013, from https://en.wikipedia.org/wiki/Test_functions_for_optimization.

    .. [2] https://www.sfu.ca/~ssurjano/schaffer4.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = 0.5 + (np.cos(np.sin(np.abs(self.p["x1"] ** 2 - self.p["x2"] ** 2))) - 0.5) /\
            (1 + 0.001 * (self.p["x1"] ** 2 + self.p["x2"] ** 2)) ** 2

        y_out = y[:, np.newaxis]

        return y_out


class SphereFunction(AbstractModel):
    """
    d-dimensional Sphere Function [1][2][3][4].
    The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal.
    The plot shows its two-dimensional form.

    .. math::
         y = \\sum_{i=1}^{d}x_i^2

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-5.12, 5.12]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-5.12, 5.12]

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
       parameters["x1"] = np.linspace(-5.12, 5.12, 100)
       parameters["x2"] = np.linspace(-5.12, 5.12, 100)

       constants = None

       plot("SphereFunction", parameters, constants, plot_3d=False)

    .. [1]Dixon, L. C. W., & Szego, G. P. (1978). The global optimization problem: an introduction.
       Towards global optimization, 2, 1-15.
    .. [2] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005). Retrieved June 2013,
       from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
    .. [3]Picheny, V., Wagner, T., & Ginsbourger, D. (2012).
       A benchmark of kriging-based infill criteria for noisy optimization.
    .. [4]https://www.sfu.ca/~ssurjano/spheref.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # determine sum
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, i_key in enumerate(keys):
               y += self.p[i_key] ** 2

        y_out = y[:, np.newaxis]

        return y_out


class Lin2Coupled(AbstractModel):
    """
    d-dimensional Lin2Coupled Function.
    The Lin2Coupled function is continuous, convex and unimodal.
    The plot shows its two-dimensional form.

    .. math::
         y = \\sum_{i=1}^{d}x_i x_{i+1}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-1, 1]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-1, 1]

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
       parameters["x1"] = np.linspace(-1, 1, 100)
       parameters["x2"] = np.linspace(-1, 1, 100)

       constants = None

       plot("Lin2Coupled", parameters, constants, plot_3d=False)

    .. [1] Alemazkoor, N., & Meidani, H. (2018). A near-optimal sampling strategy for sparse recovery of
       polynomial chaos expansions. Journal of Computational Physics, 371, 137-151.
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for i, key in enumerate(self.p.keys()):
            if type(self.p[key]) is np.ndarray:
                self.p[key] = self.p[key].flatten()

        # determine sum
        y = np.zeros(np.array(self.p[list(self.p.keys())[0]]).size)
        keys = list(self.p.keys())

        for i, i_key in enumerate(keys):
            if i < (len(keys)-1):
                y += self.p[keys[i]] * self.p[keys[i+1]]

        y_out = y[:, np.newaxis]

        return y_out


class McCormickFunction(AbstractModel):
    """
    2-dimensional McCormick Function [1][2].

    .. math::
      y = \\sin(x_1+x_2)+(x_1-x_2)^2-1.5x_1+2.5x_2+1

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-1.5, 4]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-3, 4]

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
       parameters["x1"] = np.linspace(-1.5, 4, 100)
       parameters["x2"] = np.linspace(-3, 4, 100)

       constants = None

       plot("McCormickFunction", parameters, constants, plot_3d=False)

    .. [1] Adorio, E. P., & Diliman, U. P. MVF - Multivariate Test Functions Library in C for Unconstrained
       Global Optimization (2005). Retrieved June 2013, from http://http://www.geocities.ws/eadorio/mvf.pdf.

    .. [2] https://www.sfu.ca/~ssurjano/mccorm.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = np.sin(self.p["x1"] + self.p["x2"]) + (self.p["x1"] - self.p["x2"]) ** 2 - 1.5 * self.p["x1"] +\
                   2.5 * self.p["x2"] + 1

        y_out = y[:, np.newaxis]

        return y_out


class BoothFunction(AbstractModel):
    """
    2-dimensional BoothFunction.[1][2].

    .. math::
      y = (x_1+2x_2-7)^2+(2x_1+x_2-5)^2
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [-10, 10]
    p["x2"]: float or ndarray of float [n_grid]
        second parameter defined in [-10, 10]

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
       parameters["x1"] = np.linspace(-10, 10, 100)
       parameters["x2"] = np.linspace(-10, 10, 100)

       constants = None

       plot("BoothFunction", parameters, constants, plot_3d=False)

    .. [1] Global Optimization Test Problems. Retrieved June 2013, from
       http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.
    .. [2] https://www.sfu.ca/~ssurjano/booth.html
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y = (self.p["x1"] + 2 * self.p["x2"] - 7) ** 2 + (2 * self.p["x1"] + self.p["x2"] - 5) ** 2

        y_out = y[:, np.newaxis]

        return y_out


class Peaks(AbstractModel):
    """
    Three-dimensional peaks function.

    .. math::
      y = 3(1-x_1)^2 e^{-(x_1^2)-(x_3+1)^2}-10(\\frac{x_1}{5}-x_1^3-x_3^5) e^{-x_1^2-x_3^2}-
      \\frac{1}{3} e^{-(x_1+1)^2 - x_3^2} + x_2

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
       plot("Peaks", parameters, constants, plot_3d=False)
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # # two output variables for testing
        # if y.size > 1:
        #     y_out = np.array([y, 2 * y]).transpose()
        #     additional_data = y.size * [additional_data]
        # else:
        #     y_out = np.array([y, 2 * y])
        if y.ndim == 1:
            y_out = y[:, np.newaxis]

        return y_out, additional_data


class Peaks_NaN(AbstractModel):
    """
    Three-dimensional peaks function returning NaN values for certain parameters (for testing).

    .. math::
      y = 3(1-x_1)^2 e^{-(x_1^2)-(x_3+1)^2}-10(\\frac{x_1}{5}-x_1^3-x_3^5) e^{-x_1^2-x_3^2}-
      \\frac{1}{3} e^{-(x_1+1)^2 - x_3^2} + x_2

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
       plot("Peaks", parameters, constants, plot_3d=False)
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # add some NaN values for testing
        mask_nan = self.p["x1"] < 1.5

        y[mask_nan] = np.nan

        additional_data = {"additional_data/list_mult_int": [1, 2, 3],
                           "additional_data/list_single_float": [0.2],
                           "additional_data/list_single_str": ["test"],
                           "additional_data/list_mult_str": ["test1", "test2"],
                           "additional_data/single_float": 0.2,
                           "additional_data/single_int": 2,
                           "additional_data/single_str": "test"}

        # # two output variables for testing
        # if y.size > 1:
        #     y_out = np.array([y, 2 * y]).transpose()
        #     additional_data = y.size * [additional_data]
        # else:
        #     y_out = np.array([y, 2 * y])
        if y.ndim == 1:
            y_out = y[:, np.newaxis]

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        y_1 = DiscontinuousRidgeManufactureDecay().set_parameters(self.p).simulate()
        y_2 = GenzDiscontinuous().set_parameters(self.p).simulate()

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

       plot("SurfaceCoverageSpecies", parameters, constants, plot_3d=False)

    .. [1] Le Maitre, O.P., Najm, H.N., Ghanem, R.G., Knio, O.M., (2004).
       Multi-resolution analysis of Wiener-type uncertainty propagation schemes.
       Journal of Computational Physics, 197, 502-531.
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # y_out = np.hstack((y_out, 2.*y_out))

        return y_out


class SurfaceCoverageSpecies_NaN(AbstractModel):
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

       plot("SurfaceCoverageSpecies", parameters, constants, plot_3d=False)

    .. [1] Le Maitre, O.P., Najm, H.N., Ghanem, R.G., Knio, O.M., (2004).
       Multi-resolution analysis of Wiener-type uncertainty propagation schemes.
       Journal of Computational Physics, 197, 502-531.
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # add some NaN values for testing
        mask_nan = self.p["beta"] > 18.5

        y_out[mask_nan, 0] = np.nan

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    By default u_i = 0.5 and a_i = 5.

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    .. math:: y = \\cos \\left( 2 \\pi u_1 + \\sum_{i=1}^{N}a_i x_i \\right)

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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


class GenzOscillatory_NaN(AbstractModel):
    """
    N-dimensional "Oscillatory" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].

    .. math:: y = \\cos \\left( 2 \\pi u_1 + \\sum_{i=1}^{N}a_i x_i \\right)

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # insert some NaN values for testing
        mask = self.p[list(self.p.keys())[0]] > 0.8
        y_out[mask, 0] = np.nan

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        y = 9 + 2.5 * self.p["x1"] - 17.5 * self.p["x2"] + 2.5 * self.p["x1"] * self.p["x2"] + 19 * self.p["x2"] ** 2 -\
            7.5 * self.p["x1"] ** 3 - 2.5 * self.p["x1"] * self.p["x2"] ** 2 - 5.5 * self.p["x2"] ** 4 +\
            (self.p["x1"] ** 3) * (self.p["x2"] ** 2)

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

       plot("Ishigami", parameters, constants, plot_3d=False)

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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


class Ishigami_NaN(AbstractModel):
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

       plot("Ishigami", parameters, constants, plot_3d=False)

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

        # insert some NaN values for testing
        mask = (self.p["x1"] > 2).flatten()
        y_out[mask, 0] = np.nan

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

       plot("GFunction", parameters, constants, plot_3d=False)

    .. [1] Saltelli, Andrea; Sobol, I. M. (1995): Sensitivity analysis for nonlinear
       mathematical models: numerical experience. In: Mathematical models and
       computer experiment 7 (11), pp. 16-28.

    .. [2] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
       Polynomial chaos expansion for uncertainties quantification and sensitivity analysis [PowerPoint slides].
       Retrieved from SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        x = np.vstack([self.p[key].squeeze() for key in self.p.keys()])

        y = np.ones(x.shape[1])
        y[np.linalg.norm(x-0.5, axis=0) <= 0.25] = 2.

        y_out = y[:, np.newaxis]

        return y_out


class Cluster3Simple(AbstractModel):
    """
    2-dimensional testfunction containing a spherical and a linear discontinuity.

    .. math::
       y = \\begin{cases}
       2, & \\text{if } \\sqrt{\\sum_{i=1}^{N}(x_i-0.5)^2} \\leq 0.25 \\\\
       1, & \\text{otherwise}
       \\end{cases}

    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter [0, 1]
    p["x2"]: float or ndarray of float [n_grid]
        2-nd parameter defined in [0, 1]

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

       plot("Cluster3Simple", parameters)
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        x = np.vstack([self.p[key].squeeze() for key in self.p.keys()])

        y = np.ones(x.shape[1])
        y[np.linalg.norm(x, axis=0) <= 0.25] = 2.
        y[np.sum(x, axis=0) >= 1.5] = 3.

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for key in self.p.keys():
            if self.p[key].ndim == 1:
                self.p[key] = self.p[key][:, np.newaxis]

        x = np.hstack([self.p[key] for key in self.p.keys()])

        y = np.zeros(x.shape[0])
        mask = (np.sum(x, axis=1) <= 1.).flatten()
        # mask = np.logical_and(mask, np.linalg.norm(x-0.85, axis=1) > .8)

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
        y_out = np.hstack((y_out, y_out))

        return y_out


class DiscontinuousRidgeManufactureDecay_NaN(AbstractModel):
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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        for key in self.p.keys():
            if self.p[key].ndim == 1:
                self.p[key] = self.p[key][:, np.newaxis]

        x = np.hstack([self.p[key] for key in self.p.keys()])

        y = np.zeros(x.shape[0])
        mask = (np.sum(x, axis=1) <= 1.).flatten()
        # mask = np.logical_and(mask, np.linalg.norm(x-0.85, axis=1) > .8)

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
        y_out = np.hstack((y_out, y_out))

        # insert some NaN values for testing
        mask = (self.p[list(self.p.keys())[0]] > 0.8).flatten()
        y_out[mask, 0] = np.nan

        return y_out


class OakleyOhagan2004(AbstractModel):
    # Todo: @Lucas: remove text files
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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):
        # load coefficients
        folder = os.path.split(os.path.dirname(__file__))[0]
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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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
    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())
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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

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


class ElectrodeModel(AbstractModel):
    """
    Modified version of Randles circuit.
    This circuit is used to model the impedance of an inert electrode in an electrolyte with a finite Warburg layer.
    Circuit: -Rs-(Q(Rct-(WRw)))-

    Parameters
    ----------
    p["n_Qdl"]: float or ndarray of float [n_grid]
        First parameter defined in [0, Inf]
    p["Qdl"]: float or ndarray of float [n_grid]
        Second parameter defined in [0, 1]
    p["n_Qd"]: float or ndarray of float [n_grid]
        Third parameter defined in [0, Inf]
    p["Qd"]: float or ndarray of float [n_grid]
        Fourth parameter defined in [0, 1]
    p["Rs"]: float or ndarray of float [n_grid]
        Fifth parameter defined in [0, Inf]
    p["Rct"]: float or ndarray of float [n_grid]
        Sixth parameter defined in [0, Inf]
    p["Rd"]: float or ndarray of float [n_grid]
        Seventh parameter defined in [0, Inf]
    p["w"]: float or ndarray of float [n_w]
         Frequency variable defined in [0, Inf]


    Returns
    -------
    Z: ndarray of float [n_grid x 1]
        Output
    """

    def __init__(self, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        # Z = self.p["Rs"] + 1 / (1 / (1 / (self.p["Qdl"] * (self.p["w"].T * 1j) ** self.p["n_Qdl"])) +
        #           1 / (self.p["Rct"] + 1 / (1 / (1 / (self.p["Qd"] * (self.p["w"].T * 1j) ** self.p["n_Qd"]))
        #                                     + 1 / self.p["Rd"])))

        Z = self.p["Rs"] + \
            ((self.p["Rct"] + (self.p["Rd"]*1/(self.p["Qd"]*(1j*self.p["w"].T)**(self.p["n_Qd"])))/(self.p["Rd"]+1/(self.p["Qd"]*(1j*self.p["w"].T)**(self.p["n_Qd"]))))*1/(self.p["Qdl"]*(1j*self.p["w"].T)**(self.p["n_Qdl"])))/\
            (self.p["Rct"] + (self.p["Rd"]*1/(self.p["Qd"]*(1j*self.p["w"].T)**(self.p["n_Qd"]))/(self.p["Rd"]+1/(self.p["Qd"]*(1j*self.p["w"].T)**(self.p["n_Qd"]))) + 1/(self.p["Qdl"]*(1j*self.p["w"].T)**(self.p["n_Qdl"]))))
        Z = np.concatenate((np.real(Z.T), np.imag(Z.T)), axis=1)

        return Z


class Lorenz_System(AbstractModel):
    """
    Model for the Lorenz System of differential equations. It is nonlinear and shows chaotic behaviour specifically for
    the three parameter values sigma = 10.0, beta = 28.0 and rho = 8.0/3.0. The lorenz attractor can then be observed in
    any three dimensional trajectory.

    Parameters
    ----------
    p["sigma"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["beta"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["rho"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["y1_0"] : float
        initial value of first variable
    p["y2_0"] : float
        initial value of second variable
    p["y3_0"] : float
        initial value of third variable
    p["t_end"] : float
        the end of the timespan for which the system will be evaluated
    p["step_size"] : float
        the step size for the time increments during which the system will be evaluated

    Returns
    -------
    x : ndarray of float [n_grid x n_time_steps]
        Results of x coordinate of the system in time steps, the gPC is conducted for the three parameters sigma, beta
        and rho
    """

    def __init__(self):
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        def lorenz(t, state, sigma, beta, rho):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z

            return [dx, dy, dz]

        x_out_shape = self.p["sigma"].shape[0]
        t_span = (0.0, self.p["t_end"])
        t = np.arange(0.0, self.p["t_end"][0], self.p["step_size"][0])
        sols = np.zeros((x_out_shape, t.shape[0]))
        for i in range(x_out_shape):
            p = (self.p["sigma"][i], self.p["beta"][i], self.p["rho"][i])
            y0 = [self.p["x_0"][i], self.p["y_0"][i], self.p["z_0"][i]]
            # only save x-coordinate (index 0)
            sols[i, :] = odeint(lorenz, y0, t, p, tfirst=True)[:, 0]
        x_out = sols[:, 1:]   # skip first timestep

        return x_out


class Lorenz_System_julia(AbstractModel):
    """
    Model for the Lorenz System of differential equations in julia. It is nonlinear and shows chaotic behaviour
    specifically for the three parameter values sigma = 10.0, beta = 28.0 and rho = 8.0/3.0. The lorenz attractor can
    then be observed in any three dimensional trajectory.

    Parameters
    ----------
    p["sigma"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["beta"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["rho"] : float or ndarray of float [n_grid]
        Parameter of the system
    p["y1_0"] : float
        initial value of first variable
    p["y2_0"] : float
        initial value of second variable
    p["y3_0"] : float
        initial value of third variable
    p["t_end"] : float
        the end of the timespan for which the system will be evaluated
    p["step_size"] : float
        the step size for the time increments during which the system will be evaluated

    Returns
    -------
    x : ndarray of float [n_grid x n_time_steps]
        Results of x coordinate of the system in time steps, the gPC is conducted for the three parameters sigma, beta
        and rho
    """

    def __init__(self, fname_julia=None):
        if fname_julia is not None:
            self.fname_julia = fname_julia
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        from julia import Main
        # the package DifferentialEquations.jl needs to be installed in the julia environment
        # you can add dependencies of your model by using a dedicated environment, for this example
        # the folder "julia_env" is located in the same folder as the julia file
        # if you installed the DifferentialEquations package by yourself in julia, you do not have to use a
        # dedicated environment (if julia updates, the environments also have to be updated from time to time
        # to keep working)

        #fname_folder = os.path.split(self.fname_julia)[0]
        #Main.fname_environment = os.path.join(fname_folder, 'julia_env')
        #Main.eval('import Pkg; Pkg.activate(fname_environment)')

        # access .jl file
        Main.fname_julia = self.fname_julia
        Main.include(Main.fname_julia)

        x_out_shape = self.p["sigma"].shape[0]
        t = np.arange(0.0, self.p["t_end"][0], self.p["step_size"][0])
        sols = np.zeros((x_out_shape, t.shape[0]))
        for i in range(x_out_shape):
            p = [self.p["sigma"][i], self.p["beta"][i], self.p["rho"][i]]
            y0 = [self.p["x_0"][i], self.p["y_0"][i], self.p["z_0"][i]]
            # only save x-coordinate (index 0)
            sols[i, :] = Main.Julia_Lorenz(p, y0, t)[0]
        x_out = sols[:, 1:]     # skip first timestep

        return x_out

class MUAP_gpc(AbstractModel):
    """
    MUAP_gpc runs a spinal periferal model that reproduces TMS invoked MEPs.
    The parameters of the model (constants parameters and the parameter distributions tested in the gPC)
    are stored in the dictionary p. Their type is defined during the problem definition (in main.py).

    Parameters
    ----------
    data_dir : str, optional, default: None
        Folder containing the model data (S1.h5, AP.h5, mu_spiketimes_S1.h5), e.g. as downloaded with pooch
        from Zenodo. If None, the folder "data" next to this file is used.

    Returns
    -------
    R2 : ndarray of float [n_grid x n_out]
        Goodness of fit of the MEPs in the simulations the gPC is conducted for
    """

    def __init__(self, data_dir=None, fname_matlab=None, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())
        self.fname_matlab = fname_matlab
        self.data_dir = data_dir

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        # fixed settings
        SUBJ       = 1  # To pass as a static parameter -> self.p["SUBJ"]
        WITHRC     = 1
        AMPAWEIGHT = None

        if self.data_dir is not None:
            DATA_DIR = self.data_dir
        else:
            DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        SPIKE_FILE = os.path.join(DATA_DIR, f'mu_spiketimes_S{SUBJ}.h5')

        # pygpc parameters
        b   = self.p["b"]      # pygpc parameter [45, 425] uniform
        lam = self.p["lam"]    # pygpc parameter, normal distribution mu=3.619, std=0.774

        mep_model = MEPmodel(data_dir=DATA_DIR)

        R2 = np.zeros(b.shape[0])

        for n in range(b.shape[0]):
            # Output metric
            R2[n] = mep_model.run_model(b[n], lam[n], SUBJ, WITHRC, AMPAWEIGHT, SPIKE_FILE)

        R2 = R2[:, np.newaxis]

        return R2

import os
import h5py
import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class MEPmodel:
    # Target MEP (ref) + saved MN firing times, shared by all instances
    # (equivalent to the module-level _CACHE of pygpc_main.py)
    _CACHE = {}

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        self.data_dir = data_dir

    # ======================================================================
    # pygpc interface
    # ======================================================================
    def get_setup(self, subj, withRC, AMPAweight, spike_file):
        """
        Target MEP (ref) + previously saved MN firing times.  Built on the first
        call, cached afterwards.
        """
        spike_file = spike_file or os.path.join(self.data_dir, f'mu_spiketimes_S{subj}.h5')
        key = (subj, withRC, spike_file)

        if key not in MEPmodel._CACHE:
            ref = self.config_model_bio(subj, withRC, AMPAweight)

            if not os.path.isfile(spike_file):
                raise FileNotFoundError(
                    f'Saved spike times not found: {spike_file}\n'
                    'Run the fitted model once (MEPmodel_bio_core) to generate it.'
                )
            with h5py.File(spike_file, 'r') as f:
                spike_times = np.asarray(self.load_h5_to_dict(f)['spike_times'], dtype=float)

            nIntensities = ref['y0'].shape[1]
            if spike_times.shape[2] != nIntensities:
                raise ValueError(
                    f'{spike_file} holds {spike_times.shape[2]} TMS intensities, '
                    f'but subject {subj} has {nIntensities}.'
                )

            MEPmodel._CACHE[key] = (ref, spike_times)

        return MEPmodel._CACHE[key]

    # ----------------------------------------------------------------------
    def run_model(self, b, lam, subj, withRC, AMPAweight, spike_file):
        """
        Call to the biological spinal periferal model.

        b : float
            MUAP amplitude distribution parameter, A = a*b^x. a is hard coded to 6.28
        lam  : float
            MUAP shape parameter (with zero crossing = 2.35 * lam - 0.18)
        subj : int
            the subject to fit
        withRC: int
            0 for the model without Renshaw cells, 1 for the model with Renshaw cells
        AMPAweight: list of floats
            [] to select the best value
        spike_file: str
            the name of the file containing the MN spike times

        Returns
        -------
        R2 : float
            Goodness of fit of the simulated vs. measured MEPs
        """
        ref, spike_times = self.get_setup(subj, withRC, AMPAweight, spike_file)

        # ----- generate MUAPs for this sample -----
        delay = np.ones(100) * (2.35 * lam - 0.18)  # fixed parameter
        muaps, tmuap = self.gen_muaps(n_neurons=100,
                                      amplitude=[6.28, b],  # a fixed to 6.28
                                      axonalDelay=delay,
                                      lam=lam)

        ref['model']['muaps'] = muaps
        ref['model']['tmuap'] = tmuap

        # ----- rebuild MEP from the saved spike times (no network simulation) -----
        sim = self.MEPmodel_bio_core(ref['model'], spike_times=spike_times)

        # ----- output metric -----
        ref = self.cal_error(ref, sim)

        return ref['R2']

    # ======================================================================
    # Model configuration (config_model_bio.py)
    # ======================================================================
    def config_model_bio(self, subj, withRC, AMPAweight=[]):

        model = {}
        model["AMPAweight"] = AMPAweight
        model["withRC"] = withRC
        model["maxES"] = 5  # take first 5 spikes

        # -----subject MEP (target) -----
        ref = {}
        ref["subj"] = subj
        ref["tcrop"] = [20, 50]
        if ref["subj"] == 10:
            intensity_idx = np.arange(6)  # remove 50% intensity as no MEP is elicited
        else:
            intensity_idx = None
        y0, t0, _, intensities, _, _ = self.load_MEP(subj, intensity_idx, ref["tcrop"])
        ref["intensities"] = intensities
        intensity_idx = np.arange(len(intensities))
        ref["intensity_idx"] = intensity_idx
        ref["t0"] = t0
        ref["y0"] = y0.T

        # -----subject RMT ---------
        # (by eyeballing the IO curve)
        if ref["subj"] == 1 or ref["subj"] == 8:
            ref["RMT"] = 32
        elif ref["subj"] == 2 or ref["subj"] == 10:
            ref["RMT"] = 35
        elif ref["subj"] == 3 or ref["subj"] == 9:
            ref["RMT"] = 38
        elif ref["subj"] == 4:
            ref["RMT"] = 41
        elif ref["subj"] == 5 or ref["subj"] == 6:
            ref["RMT"] = 47
        else:
            ref["RMT"] = 44

        # ----- simulated DI wave -----
        tlength = t0[-1]  # ms
        dt = 0.1  # ms
        t = np.arange(0, tlength, dt)
        model["DIwave0"] = np.zeros((len(ref["intensities"]), len(t)))
        for i in range(len(ref["intensities"])):
            model["DIwave0"][i, :] = self.gen_DIwave(t, ref["intensities"][i] / ref["RMT"])
        model["DIwave"] = self.deconv_DIwave(t, model["DIwave0"], ref)

        # ----- AMPA, NMDA kernels -----
        model["AMPA"], model["NMDA"] = self.gen_kernels(dt, tlength)

        # ----- AchR, GlyR kernels -----
        if "kernel" not in model.keys():
            model["kernel"] = {}
        model["kernel"]["tau"] = [[0.5, 3.6], [1.8, 20.2], [1, 6]]
        model["kernel"]["h"] = [[1.5977], [1.3908], [1.7175]]

        # ----- Motor neuron -----
        model["tauLIF"] = 10  # ms
        model["Eexc"] = 0  # mV
        model["Einh"] = -75  # mV
        model["V_rest"] = -65  # mV
        model["V_thr"] = -55  # mV
        model["T_ref"] = 2  # ms, refractory period
        model["fastAChRweight"] = 0.5  # AchR (MN->RC)

        # ----- tDIwave (input) -----
        model["DIwaveConv"] = model["DIwave"]
        model["DIwaveConv_AMPA"] = np.zeros((len(ref["intensities"]), len(t)))
        model["DIwaveConv_NMDA"] = np.zeros((len(ref["intensities"]), len(t)))
        fr = 1
        for i in range(len(ref["intensities"])):
            g = np.convolve(fr * model["DIwave"][i, :], model["AMPA"])
            model["DIwaveConv_AMPA"][i, :] = g[:len(t)]
            g = np.convolve(fr * model["DIwave"][i, :], model["NMDA"])
            model["DIwaveConv_NMDA"][i, :] = g[:len(t)]
        model["t"] = t
        model["dt"] = dt

        # ----- search boundary-----
        if withRC == 0:
            nParams = 7
            boundary = np.zeros((nParams, 2))  # [lower, upper]
            boundary[0, :] = np.array([0, 10])  # R1
            boundary[1, :] = np.array([0, 10])  # R2
            boundary[2, :] = np.array([0, 10])  # R3
            boundary[3, :] = np.array([0, 10])  # R4
            boundary[4, :] = np.array([0, 10])  # R5
            boundary[5, :] = np.array([5, 10])  # Tmuap
            boundary[6, :] = np.array([0, 1])  # AMPAweight
            if AMPAweight is not None and len(np.atleast_1d(AMPAweight)) > 0:
                boundary[6, :] = np.array([1, 1]) * np.atleast_1d(AMPAweight)  # fixed value
            model["boundarytext"] = ['R1', 'R2', 'R3', 'R4', 'R5', 'MU.T', 'AMPAw']
        else:
            nParams = 12
            boundary = np.zeros((nParams, 2))  # [lower, upper]
            boundary[0, :] = np.array([0, 10])  # R1
            boundary[1, :] = np.array([0, 10])  # R2
            boundary[2, :] = np.array([0, 10])  # R3
            boundary[3, :] = np.array([0, 10])  # R4
            boundary[4, :] = np.array([0, 10])  # R5
            boundary[5, :] = np.array([0, 20])  # E1 (MN1 -> RC)
            boundary[6, :] = np.array([0, 20])  # E2 (MN100 -> RC)
            boundary[7, :] = np.array([0, 10])  # I1 (RC -> MN1)
            boundary[8, :] = np.array([0, 10])  # I2 (RC -> MN100)
            boundary[9, :] = np.array([1, 10])  # RC.th
            boundary[10, :] = np.array([5, 10])  # Tmuap
            boundary[11, :] = np.array([0, 1])  # AMPAweight
            if AMPAweight is not None and len(np.atleast_1d(AMPAweight)) > 0:
                boundary[11, :] = np.array([1, 1]) * np.atleast_1d(AMPAweight)  # fixed value
            model["boundarytext"] = ['R1', 'R2', 'R3', 'R4', 'R5', 'E1', 'E2', 'I1', 'I2', 'RC.th', 'MU.T', 'AMPAw']

        model["boundary"] = boundary
        ref["model"] = model

        if withRC == 1:
            if AMPAweight is None or len(np.atleast_1d(AMPAweight)) == 0:
                # Free fit — no fixed AMPAweight
                ref["resultname"] = os.path.join(self.data_dir, f"result_bio_s{subj}.h5")
            else:
                # Fixed AMPAweight provided — encode each value in the filename
                ampa_tag = '_'.join(f'{v:g}' for v in np.atleast_1d(AMPAweight))
                ref["resultname"] = os.path.join(self.data_dir, f"result_bio_s{subj}[{ampa_tag}].h5")
        else:
            ref["resultname"] = os.path.join(self.data_dir, f"result_bioNoRC_s{subj}.h5")
        ref["figname"] = ref["resultname"][:-3] + ".svg"

        return ref

    # ======================================================================
    # Data loading (load_MEP.py, h5_helpers.py)
    # ======================================================================
    def load_MEP(self, subj, iidx=None, tcrop=[20, 50]):
        TSTIM = 100  # TMS stimulus was at t=100 ms

        # ----- load data -----
        mat_path = os.path.join(self.data_dir, f"S{subj}.h5")

        with h5py.File(mat_path, 'r') as f:
            tmp = self.load_h5_to_dict(f)

        mep = tmp["mep"]  # shape: [N x time x trials]
        intensities = tmp["intensities"].flatten()
        times = tmp["t"].flatten() - TSTIM

        # ----- select intensities -----
        if iidx is None:
            iidx = np.arange(len(intensities))

        mep = mep[iidx, :, :]
        intensities = intensities[iidx]

        # ----- crop -----
        tidx = np.where((times >= tcrop[0]) & (times < tcrop[1]))[0]
        t = times[tidx]

        yall = mep[:, tidx, :]
        y = np.mean(yall, axis=2)

        # ----- remove baseline [-20, -10] ms -----
        baseline_idx = np.where((times >= -20) & (times < -10))[0]

        baseline = np.mean(
            np.mean(mep[:, baseline_idx, :], axis=2),
            axis=1
        )  # shape: [n]

        # subtract baseline
        y = y - baseline[:, np.newaxis]
        mep = mep - baseline[:, np.newaxis, np.newaxis]

        return y, t, mep, intensities, times, yall

    # ----------------------------------------------------------------------
    @staticmethod
    def load_h5_to_dict(group):
        """Recursively loads an H5 group back into a dictionary."""
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data[key] = MEPmodel.load_h5_to_dict(item)
            else:
                value = item[()]  # [()] extracts the data as a numpy array/scalar
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                data[key] = value
        return data

    # ======================================================================
    # DI-wave generation (gen_DIwave.py)
    # ======================================================================
    def gen_DIwave(self, t, intensity):
        t0 = 5
        T = 1.5
        width = 0.25

        DIwave = np.zeros(np.size(t))

        # D, I1, I2, I3, I4
        x0 = [1.36192637, 1.04127548, 1.16603639, 1.03733872, 1.45405986]
        r = [18.50774852, 9.26210842, 5.91559859, 17.7805388, 425.51252596]
        a = [0.34532065, 1.0, 0.80577286, 0.46054753, 0.27828232]

        for i in range(5):
            DIwave = DIwave + np.exp(-(t - t0 - i * T) ** 2 / 2 / width ** 2) * self.sigmoid(intensity, x0[i], r[i],
                                                                                             a[i])

        return DIwave

    # ======================================================================
    # DI-wave deconvolution (deconv_DIwave.py, generate_EP.py)
    # ======================================================================
    @staticmethod
    def deconvreg(signal: np.ndarray, psf: np.ndarray, lam: float = 100.0) -> np.ndarray:
        """
        1-D regularised deconvolution (Python equivalent of MATLAB deconvreg).

        Parameters
        ----------
        signal : array_like, shape (N,)
            Observed / blurred signal (DIwave2 in the original code).
        psf : array_like, shape (M,)
            Point-spread function / kernel (EP).  If M < N it is zero-padded
            to length N before the FFT; if M > N it is truncated.
        lam : float, optional
            Regularisation parameter (lambda in MATLAB).  Default is 100.

        Returns
        -------
        rate : ndarray, shape (N,)
            Deconvolved (restored) signal, real-valued.

        Notes
        -----
        The formula applied is Tikhonov regularisation with a second-difference
        (Laplacian) constraint matrix C:

            F = conj(H) * G / (|H|² + lambda * C²)

        where G = FFT(signal), H = FFT(psf_padded), and
        C(k) = 2 − 2·cos(2π k / N)  (eigenvalues of the 1-D discrete Laplacian).

        This matches MATLAB's default deconvreg behaviour to floating-point
        precision (relative error < 1e-12 on test data).
        """
        signal = np.asarray(signal, dtype=float)
        psf = np.asarray(psf, dtype=float)

        N = len(signal)

        # Zero-pad (or truncate) PSF to the length of the signal
        psf_padded = np.zeros(N)
        m = min(len(psf), N)
        psf_padded[:m] = psf[:m]

        # Optionally apply psf2otf-style centering: pad FIRST, then roll.
        # This replicates MATLAB's psf2otf(), which shifts the already-padded
        # array so that the PSF centre lands at index 0.
        if True:
            center = len(psf) // 2
            psf_padded = np.roll(psf_padded, -center)

        # Frequency-domain representations
        G = fft(signal)
        H = fft(psf_padded)

        # Eigenvalues of the 1-D discrete Laplacian (constraint matrix)
        k = np.arange(N)
        C = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)

        # Tikhonov regularised inverse filter
        F = (np.conj(H) * G) / (np.abs(H) ** 2 + lam * C ** 2)

        # Back to time domain -- result should be real
        rate = np.real(ifft(F))
        return rate

    # ----------------------------------------------------------------------
    def deconv_DIwave(self, times: np.ndarray, DIwave: np.ndarray, ref: dict) -> np.ndarray:
        """
        Python translation of the MATLAB function deconv_DIwave.

        Parameters
        ----------
        times   : 1-D array, shape (501,)      — time axis in ms (0–50 ms).
        DIwave  : 2-D array, shape (10, 501)   — input waveforms.
        ref     : dict with optional keys:
                    'EP' — pre-computed EP waveform (1-D array)
                    'dt' — time step of EP (float, ms)

        Returns
        -------
        DIwave_deconv : 2-D array, shape (10, 501) — deconvolved, normalised waveforms.
        """

        DIwave_deconv = np.zeros(DIwave.shape)

        # --------- EP ------------
        if "EP" not in ref:
            d = 0.1
            EP, t, _ = self.generate_EP(d)
            dt = t[1] - t[0]
            EP = -EP
        else:
            EP = ref["EP"]
            dt = ref["dt"]

        # -------- DIwave -------------
        times2 = np.arange(times[0], times[-1], dt)
        DIwave3 = np.zeros((10, len(times2)))
        rate1 = np.zeros((10, len(times2)))

        for i in range(DIwave.shape[0]):  # number of intensities

            # Resample DIwave[i] onto a uniform grid with step dt.
            # np.arange mirrors MATLAB's times(1):dt:times(end).
            times2 = np.arange(times[0], times[-1], dt)  # ms

            valid = ~np.isnan(DIwave[i])
            f_interp = interp1d(times[valid], DIwave[i][valid],  # ← no NaN knots
                                kind='linear',
                                bounds_error=False,
                                fill_value=np.nan)
            DIwave2 = f_interp(times2)
            DIwave3[i, :] = DIwave2

            # -------- deconv(DIwave, EP) -----------
            lambda_ = 1e9
            # MATLAB: deconvreg(DIwave2', EP, lambda)
            rate = self.deconvreg(DIwave2.T, EP, lambda_)
            rate1[i, :] = rate

            # Interpolate rate back onto the original time axis.
            # NaN outside bounds, matching MATLAB interp1 default.

            f_back = interp1d(
                times2, rate,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            DIwave_deconv[i, :] = f_back(times)

        # Normalise by the global maximum (matches MATLAB's max(DIwave_deconv(:)))
        DIwave_deconv = DIwave_deconv / np.nanmax(DIwave_deconv)
        return DIwave_deconv

    # ----------------------------------------------------------------------
    def generate_EP(self, d=0.01, Axontype=1):
        # ----------load AP (by Arancibia-Carcamo model)
        if Axontype == 1:
            data_path = os.path.join(self.data_dir, 'AP.h5')
        else:
            data_path = os.path.join(self.data_dir, 'AP2.h5')

        # Load the file
        with h5py.File(data_path, 'r') as f:
            mat_contents = self.load_h5_to_dict(f)
        times = mat_contents['TIME_VECTOR']
        times = times.reshape((len(times), 1))
        MEMBRANE_POTENTIAL = mat_contents['MEMBRANE_POTENTIAL']

        # ----------------------------------------------------
        idx = 20
        padding = 1000
        dt = times[1] - times[0]

        # Append padding to times
        padding_times = times[-1] + (np.arange(1, padding + 1) * dt)
        padding_times = padding_times.reshape((len(padding_times), 1))
        times = np.concatenate((times, padding_times))
        # Extract specific AP and add padding
        v_segment = MEMBRANE_POTENTIAL[:, idx]
        v_segment = v_segment.reshape(-1, 1)
        v_padding = np.ones(padding) * MEMBRANE_POTENTIAL[-1, idx]
        v_padding = v_padding.reshape(-1, 1)
        v = np.concatenate((v_segment, v_padding))

        # Calculate derivatives
        if isinstance(idx, int):
            s = 1
        else:
            s = len(idx)
        dv = np.vstack((np.diff(v, axis=0), np.zeros((1, s))))
        ddv = np.vstack((np.diff(dv, axis=0), np.zeros((1, s))))

        # ---------calculate extracellular potential------------
        EP = np.zeros(ddv.shape)
        for i in range(v.shape[1]):
            for t_idx in range(len(times)):
                mask = 1.0 / np.sqrt((times - times[t_idx]) ** 2 + d ** 2).T
                EP[t_idx, i] = np.sum(ddv[:, i] * mask)

        # ---------------prepare output data---------------
        dt2 = 1e-3
        times2 = np.arange(-0.5, 0.5 + dt2, dt2)

        # Process AP2
        AP2_raw = v.flatten() - np.min(v)
        max_val_ap = np.max(AP2_raw)
        i_max_ap = np.argmax(AP2_raw)
        AP2_norm = AP2_raw / max_val_ap

        f_ap = interp1d((times - times[i_max_ap]).flatten(), AP2_norm, kind='linear')
        AP2 = f_ap(times2).T

        # Process EP2
        min_val_ep = np.min(EP)
        i_min_ep = np.argmin(EP)
        EP2_norm = EP.flatten() / np.abs(min_val_ep)

        f_ep = interp1d((times - times[i_min_ep]).flatten(), EP2_norm, kind='linear')
        EP2 = f_ep(times2).T

        return EP2, times2, AP2

    # ======================================================================
    # Synaptic kernels (gen_kernels.py)
    # ======================================================================
    @staticmethod
    def gen_kernels(dt, tlength):
        """
        Generate AMPA and NMDA kernels (in msec) for convolution.
        Parameters and refs are based on Roese's report.
        """

        t = np.arange(0, tlength, dt)
        kernels = np.zeros((5, t.size))

        # tau = [rise, decay]
        tau = np.array([
            [1.0, 5.0],  # Index 0: AMPA (In->MN)
            [3.0, 50.0],  # Index 1: NMDA (In->MN)
            [0.5, 3.6],  # Index 2: fast AChR
            [1.8, 20.2],  # Index 3: slow AChR
            [1.0, 6.0]  # Index 4: GlyR (RC->MN)
        ])

        # h = normalizing terms
        h = np.array([1.8692, 1.2731, 1.5977, 1.3908, 1.7175]) * 1e-2

        for i in range(5):
            kernels[i, :] = h[i] * (np.exp(-t / tau[i, 1]) - np.exp(-t / tau[i, 0]))

        AMPA = kernels[0, :]
        NMDA = kernels[1, :]

        return AMPA, NMDA

    # ======================================================================
    # Synthetic MUAPs (gen_muaps.py, helper_f_genmuaps.py)
    # ======================================================================
    @staticmethod
    def time_vectors():
        """
        Generates and returns the time vectors of the MUAPs (tmuap) and of the
        motor neurons (t_mn)
        """
        tmuap = np.arange(0, 20, 0.1)
        t_mn = np.linspace(0, 20, 200)
        return tmuap, t_mn

    # ----------------------------------------------------------------------
    def gen_muaps(self, n_neurons, amplitude, axonalDelay, lam, zero_muaps=None):
        """
        Generate MUAP waveforms using the first-order Hermite-Rodriguez
        function described in the paper (Eqs. 4-5)
        (https://journals.physiology.org/doi/full/10.1152/jn.00626.2017).

        Parameters
        ----------
        n_neurons    : int
            Number of motor neurons
        amplitude    : ndarray
            Either amplitude of each MUAP (n_neurons,)
            Or scaling (a) and base (b) of exponent of the amplitude distribution a*b^x (2,)
        axonal delay : ndarray, shape (n_neurons,)
            Zero-crossing time of each anatomical MUAP, or a mean value.
        lam          : float or ndarray, shape (n_neurons,)
            Shape parameter of the Hermite-Rodriguez function [ms]. One value per
            motor unit as returned by fit_lam, or a mean value.
        zero_muaps   : ndarray of bool, shape (n_neurons,), optional
            Motor units whose anatomical MUAP is all zero. The corresponding
            synthetic MUAPs are set to all zero as well. If None, every MUAP is
            generated from the Hermite-Rodriguez function.

        Returns
        -------
        muaps : ndarray, shape (200, n_neurons)
        tmuap : ndarray, shape (200,)
        """
        # Paper parameters
        if isinstance(lam, np.ndarray):
            lam = np.atleast_1d(np.asarray(lam, dtype=float))[None, :]

        # Time vectors
        tmuap, t_mn = self.time_vectors()

        # Amplitude A_i: exponential (log-spaced) distribution over a n_neurons-fold
        # range, from A_1 = a (first-recruited MU) to A_M = b*a
        # (last-recruited MU). Motor units are assumed ordered by recruitment
        # order (column index 0 = first recruited).
        if len(amplitude) == 2:
            a = amplitude[0]
            b = amplitude[1]
            if n_neurons > 1:
                frac = np.arange(n_neurons) / (n_neurons - 1)
            else:
                frac = np.zeros(1)
            A = a * b ** frac
        else:
            A = amplitude

        t_M = t_mn[:, None]  # (200, 1)
        t_D = axonalDelay  # shape (N,)
        t_D = t_D[None, :]  # (1, N)
        t_MUAP = tmuap[:, None]  # (1, N)
        A = A[None, :]  # (1, N)

        z = t_D - t_M - t_MUAP  # (200, N)

        # Eq. 4: H_i(t) = A_i * (tau_i - t) * exp(-((tau_i - t)/lambda)^2) * u(tau_i - t)
        normalization_factor = np.max(z * np.exp(-(z / lam) ** 2), axis=0, keepdims=True)  # To ensure that Am = b * A1
        muaps = A * (z * np.exp(-(z / lam) ** 2)) / normalization_factor

        # An all-zero anatomical MUAP gives an all-zero synthetic MUAP. Its
        # amplitude and zero-crossing time are undefined (0 and NaN), so the
        # column is overwritten instead of being computed
        if zero_muaps is not None:
            muaps[:, np.asarray(zero_muaps, dtype=bool)] = 0.0

        return muaps, tmuap

    # ======================================================================
    # Spinal model core (MEPmodel_bio_core.py, sigmoid.py)
    # ======================================================================
    @staticmethod
    def sigmoid(x, x0, r, a):
        return a / (1 + np.exp(r * (x0 - x)))

    # ----------------------------------------------------------------------
    def MEPmodel_bio_core(self, model, spike_times=None):
        """
        Spinal model: LIF motor-neuron pool (+ optional Renshaw cells) driven by the
        DI-wave input, followed by superposition of the MUAPs at the MN firing times.

        Parameters
        ----------
        model : dict
            Model configuration (see config_model_bio / update_model_bio).
        spike_times : np.ndarray, optional
            [nMU x maxES x nIntensities] array of previously computed MN firing
            times (NaN-padded).  When supplied, the network simulation is skipped
            entirely and the MEP is rebuilt from these firing times together with
            the MUAPs in `model`.  Only model['t'], model['muaps'] and
            model['tmuap'] are then required.

        Returns
        -------
        sim : dict
        """
        if spike_times is not None:
            return self._core_from_spike_times(model, spike_times)

        # Mapping dictionary keys from the model structure
        DIwave = model['DIwaveConv']
        t = model['t']
        muaps = model['muaps']
        tmuap = model['tmuap']
        R = model['R']  # membrane resistance of MNs
        Wexc = model['Wexc']
        RWinh = model['RWinh']  # R * Winh
        dt = model['dt']
        tauLIF = model['tauLIF']
        Eexc = model['Eexc']
        Einh = model['Einh']
        V_rest = model['V_rest']
        V_thr = model['V_thr']
        T_ref = model['T_ref']
        Tmu = model['Tmu']
        withRC = model['withRC']
        maxES = model['maxES']
        fastAChRweight = model['fastAChRweight']
        rc = model['rc']
        tau = np.array(model['kernel']['tau'])  # Nested structure handled as dict
        h = np.array(model['kernel']['h']).flatten()

        # ------------record for all TMS intensities--------------
        nIntensities = DIwave.shape[0]
        simMEP = np.zeros((nIntensities, len(t)))
        spike_times = np.full((100, maxES, nIntensities), np.nan)
        spike_times2 = [None] * nIntensities  # all spike times (list of objects)

        gexc_all = np.zeros((nIntensities, len(t)))
        ginh_all = np.zeros((nIntensities, len(t)))
        Rm_all = np.zeros((nIntensities, len(t)))
        Vm_all = np.zeros((nIntensities, len(t)))
        mRC_all = np.zeros((nIntensities, len(t)))
        mMN_all = np.zeros((nIntensities, len(t)))
        vRC_all = np.zeros((nIntensities, len(t)))
        Iexc_all = np.zeros((nIntensities, len(t)))
        Iinh_all = np.zeros((nIntensities, len(t)))

        for i in range(nIntensities):

            # ----- Initialization -----
            gexc = DIwave[i, :]  # synaptic conductances on MNs
            ginh = np.zeros(len(t))  # synaptic conductances on MNs
            Iexc = np.zeros((100, len(t)))  # synaptic current to MNs
            Iinh = np.zeros((100, len(t)))  # synaptic current to MNs
            Vm = np.ones((100, len(t))) * V_rest
            Vm_lag = np.ones((100, len(t))) * V_rest
            sMN = np.zeros((100, len(t)))  # MN spikes (0/1)
            TR = np.zeros(100)  # Timer of refractory period
            mMN = np.zeros(len(t))  # MN firing rate
            mRC = np.zeros(len(t))  # RC firing rate
            v = np.zeros((3, len(t)))  # average postsynaptic potential
            v2 = np.zeros((3, len(t)))  # derivative helper

            # ----- simulation -----
            for tt in range(len(t) - 1):

                # -----MN rate-----
                mMN[tt] = np.mean(Wexc * sMN[:, tt])

                # -----RC rate-----
                if withRC:
                    rescale = 100
                    mRC[tt] = self.sigmoid(v[0, tt] * fastAChRweight + v[1, tt] * (1 - fastAChRweight),
                                           rc['v_thr'] / rescale, rc['r'] * rescale, rc['fmax'])

                # -----PSPs (biexponential kernel)-----
                input_vec = np.array([mMN[tt], mMN[tt], mRC[tt]])

                dv = v2[:, tt]

                dv2 = (h * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (
                            tau[:, 0] * tau[:, 1]))

                v[:, tt + 1] = v[:, tt] + dv * dt
                v2[:, tt + 1] = v2[:, tt] + dv2 * dt

                if withRC:
                    ginh[tt] = v[2, tt]
                    Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]
                    Iinh[:, tt] = (Einh - Vm[:, tt]) * ginh[tt]
                    Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] +
                                                                 R * Iexc[:, tt] + RWinh * Iinh[:, tt])
                else:
                    Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]
                    Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] +
                                                                 R * Iexc[:, tt])

                # -----refractory period-----
                idx_tr = TR > 0
                Vm[idx_tr, tt + 1] = V_rest
                TR[idx_tr] -= 1

                # -----spike of 100 motor neurons-----
                idx_spike = Vm[:, tt + 1] > V_thr
                Vm_lag[idx_spike, tt + 1] = Vm[idx_spike, tt + 1]
                sMN[idx_spike, tt + 1] = 1
                Vm[idx_spike, tt + 1] = V_rest
                TR[idx_spike] = round(T_ref / dt)

            # ----- spike times -----
            T_idx = Tmu / dt

            for n in range(100):
                # Find all spikes for neuron n
                idx = np.where(Vm_lag[n, :] > V_thr)[0]

                # -----collect effective spikes-----
                idxES = []
                refractory_idx = 0
                counter = 0

                while len(idx) > 0 and counter < maxES:
                    if idx[0] > refractory_idx:
                        idxES.append(idx[0])
                        refractory_idx = idx[0] + T_idx
                        counter += 1
                    idx = idx[1:]

                # ---------------------------------
                for k in range(len(idxES)):
                    curr_idx = idxES[k]
                    spike_t = (dt / (Vm_lag[n, curr_idx] - Vm[n, curr_idx - 1]) * (V_thr - Vm[n, curr_idx - 1]) + t[
                        curr_idx - 1])

                    spike_times[n, k, i] = spike_t

            # ----- MEP = superposition of MUAPs at the MN firing times -----
            simMEP[i, :] = self.mep_from_spike_times(spike_times[:, :, i], muaps, tmuap, t)

            # ---------for record--------
            gexc_all[i, :] = R[0] * gexc
            Rm_all[i, :] = sMN[0, :]
            Vm_all[i, :] = Vm[0, :]

            spike_rows, spike_cols = np.where(sMN == 1)
            spike_times2[i] = np.column_stack((spike_rows + 1, t[spike_cols]))

            mMN_all[i, :] = mMN
            vRC_all[i, :] = v[0, :] * fastAChRweight + v[1, :] * (1 - fastAChRweight)
            Iexc_all[i, :] = Iexc[0, :]
            Iinh_all[i, :] = Iinh[0, :]

            if withRC:
                mRC_all[i, :] = mRC
                ginh_all[i, :] = RWinh[0] * ginh

        # Construct output dictionary
        sim = {
            't': t,
            'simMEP': simMEP,
            'spike_times': spike_times,
            'gexc_all': gexc_all,
            'ginh_all': ginh_all,
            'Rm_all': Rm_all,
            'Vm_all': Vm_all,
            'mRC_all': mRC_all,
            'mMN_all': mMN_all,
            'vRC_all': vRC_all,
            'spike_times2': spike_times2,
            'Iexc_all': Iexc_all,
            'Iinh_all': Iinh_all
        }

        return sim

    # ----------------------------------------------------------------------
    @staticmethod
    def mep_from_spike_times(spike_times_i, muaps, tmuap, t):
        """
        Superpose the MUAPs at the firing times of one TMS intensity.

        spike_times_i : [nMU x maxES] firing times (ms), NaN where there is no spike
        muaps         : [nSamples x nMU] MUAP waveforms
        tmuap         : [nSamples,] time base of the MUAPs (ms, relative to firing)
        t             : [nTime,] simulation time base (ms)
        """
        muaps = np.asarray(muaps)
        tmuap = np.asarray(tmuap).ravel()
        t = np.asarray(t).ravel()
        mep = np.zeros(len(t))

        for n in range(spike_times_i.shape[0]):
            st = spike_times_i[n, :]
            st = st[~np.isnan(st)]
            if st.size == 0:
                continue
            muap_n = muaps[:, n].ravel()
            for spike_t in st:
                # equivalent to interp1d(..., bounds_error=False, fill_value=0), but faster
                mep += np.interp(t, tmuap + spike_t, muap_n, left=0.0, right=0.0)

        return mep

    # ----------------------------------------------------------------------
    def _core_from_spike_times(self, model, spike_times):
        """
        Rebuild the MEP from previously saved MN firing times — no network
        simulation.  Only the MUAP-dependent part of the model is recomputed, so
        this is the fast path for parameter studies on the MUAPs (e.g. pygpc).
        """
        t = np.asarray(model['t']).ravel()
        muaps = np.asarray(model['muaps'])
        tmuap = np.asarray(model['tmuap'])
        spike_times = np.asarray(spike_times, dtype=float)

        if spike_times.ndim != 3:
            raise ValueError(
                f'spike_times must be [nMU x maxES x nIntensities], got shape {spike_times.shape}'
            )
        nMU, _, nIntensities = spike_times.shape
        if muaps.shape[1] != nMU:
            raise ValueError(
                f'muaps has {muaps.shape[1]} motor units but spike_times has {nMU}'
            )

        simMEP = np.zeros((nIntensities, len(t)))
        for i in range(nIntensities):
            simMEP[i, :] = self.mep_from_spike_times(spike_times[:, :, i], muaps, tmuap, t)

        return {'t': t, 'simMEP': simMEP, 'spike_times': spike_times}

    # ======================================================================
    # Error / goodness of fit (MEPmodel_bio.py)
    # ======================================================================
    @staticmethod
    def _shift_and_scale(t, simMEP, t0, y0, axonalDelay):
        """
        Shift the simulated MEPs by `axonalDelay` ms, resample them on t0 (zero outside
        the simulated range) and scale them to the maximum of the measured MEPs.
        """
        f_interp = interp1d(
            t + axonalDelay,
            simMEP,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=0
        )
        simMEP = f_interp(t0)

        # normalization
        if np.max(simMEP) > 0:
            simMEP2 = simMEP / np.max(simMEP) * np.max(y0)
        else:
            simMEP2 = simMEP

        return simMEP2

    # ----------------------------------------------------------------------
    def fit_axonal_delay(self, t, simMEP, t0, y0, bounds=(0.0, 30.0), grid_step=None):
        """
        Axonal delay [ms] that minimises the sum of squared errors between the
        shifted, normalised simulated MEPs and the measured MEPs, over all TMS
        intensities at once. This is equivalent to maximising R2.

        The error is multimodal in the shift (one local minimum per MEP lobe), so
        a coarse grid over the fixed `bounds` first locates the global minimum,
        and a bounded Brent search then refines it within +/- one grid step.
        `bounds` must not depend on the model parameters, otherwise the result is
        no longer a continuous function of them.
        """
        if grid_step is None:
            grid_step = t[1] - t[0]

        def sse(shift):
            return np.sum((y0 - self._shift_and_scale(t, simMEP, t0, y0, shift)) ** 2)

        grid = np.arange(bounds[0], bounds[1] + 0.5 * grid_step, grid_step)
        i_best = int(np.argmin([sse(s) for s in grid]))

        lo = max(bounds[0], grid[i_best] - grid_step)
        hi = min(bounds[1], grid[i_best] + grid_step)
        res = minimize_scalar(sse, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        return res.x

    # ----------------------------------------------------------------------
    def cal_error(self, ref, sim, delay_bounds=(0.0, 30.0)):

        y0 = ref["y0"]
        t0 = ref["t0"]

        simMEP = sim["simMEP"].T
        t = sim["t"]

        # ----- axonal delay: shift that minimises the error over all intensities -----
        axonalDelay = self.fit_axonal_delay(t, simMEP, t0, y0, bounds=delay_bounds)

        # ----- shifted, interpolated and normalised simulated MEPs -----
        simMEP2 = self._shift_and_scale(t, simMEP, t0, y0, axonalDelay)

        error = np.ravel(simMEP2, order="F") - np.ravel(y0, order="F")

        NRMSD = (
                np.linalg.norm(y0 - simMEP2)
                / np.sqrt(len(t0))
                / (np.max(y0) - np.min(y0))
        )

        R2 = 1 - np.sum((y0 - simMEP2) ** 2) / np.sum((y0 - np.mean(y0)) ** 2)

        ref["NRMSD"] = NRMSD
        sim["simMEP2"] = simMEP2
        ref["model"]["axonalDelay"] = axonalDelay
        ref["R2"] = R2
        ref["error"] = error
        ref["sim"] = sim

        return ref


