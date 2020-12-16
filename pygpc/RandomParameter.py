import scipy.special
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from .BasisFunction import *


class RandomParameter(object):
    """
    RandomParameter class

    Parameters
    ----------
    pdf_type : str
        Distribution type of random variable ('beta', 'norm', 'gamma')
    pdf_shape : list of float [2]
        Shape parameters
    pdf_limits : list of float [2]
        Lower and upper bounds of random variable (if applicable)

    Attributes
    ----------
    pdf_type : str
        Distribution type of random variable ('beta', 'norm', 'gamma')
    pdf_shape : list of float [2]
        Shape parameters of beta distributed random variable [p, q]
    pdf_limits : list of float [2]
        Lower and upper bounds of random variable [min, max]
    mean : float
        Mean value
    std : float
        Standard deviation
    var : float
        Variance
    """
    def __init__(self, pdf_type=None, pdf_shape=None, pdf_limits=None):
        """
        Constructor; Initializes random parameter;
        """

        self.pdf_type = pdf_type
        self.pdf_shape = np.array(pdf_shape).astype(float)
        self.pdf_limits = np.array(pdf_limits).astype(float)
        self.pdf_limits_norm = None
        self.mean = None
        self.std = None
        self.var = None
        self.p_perc = None

    def pdf(self, x=None):
        pass

    def pdf_norm(self, x=None):
        pass

    def icdf(self, p):
        pass

    def plot_pdf(self, legend_str=None, norm=False):
        """
        Plots probability density function

        Parameters
        ----------
        legend_str : str, optional, default: None
            Legend string
        norm : boolean, optional, default: False
            Plot pdfs in normalized space [-1, 1]
        """

        if not norm:
            # delta = self.pdf_limits[1] - self.pdf_limits[0]
            # x = np.linspace(self.pdf_limits[0] - 0.0 * delta,
            #                 self.pdf_limits[1] + 0.0 * delta, 200)
            x, y = self.pdf()
        else:
            delta = 2.
            # x = np.linspace(-1. - 0.0 * delta,
            #                 +1. + 0.0 * delta, 200)
            x, y = self.pdf_norm()

        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("p(x)")
        plt.grid(True)

        if legend_str is not None:

            if type(legend_str) is not list:
                legend_str = [legend_str]

            plt.legend(legend_str)

        return plt


class Beta(RandomParameter):
    """
    Beta distributed random variable sub-class

    Parameters
    ----------
    pdf_shape: list of float [2]
        Shape parameters of beta distributed random variable [p, q]
    pdf_limits: list of float [2]
        Lower and upper bounds of random variable [min, max]

    Notes
    -----
    Probability density function:

    .. math::

       p(x) = \\left(\\frac{\\Gamma(p)\\Gamma(q)}{\\Gamma(p+q)}(b-a)^{(p+q-1)}\\right)^{-1} (x-a)^{(p-1)} (b-x)^{(q-1)}

    Examples
    --------
    >>> import pygpc
    >>> pygpc.RandomParameter.Beta(pdf_shape=[5, 2], pdf_limits=[1.2, 2])
    """

    def __init__(self, pdf_shape, pdf_limits):
        """
        Constructor; Initializes beta distributed random variable
        """

        super(Beta, self).__init__(pdf_type='beta', pdf_shape=pdf_shape, pdf_limits=pdf_limits)

        self.mean = float(self.pdf_shape[0]) / (self.pdf_shape[0] + self.pdf_shape[1]) * \
                    (self.pdf_limits[1] - self.pdf_limits[0]) + self.pdf_limits[0]

        self.std = np.sqrt(self.pdf_shape[0] * self.pdf_shape[1] / \
                           ((self.pdf_shape[0] + self.pdf_shape[1] + 1) *
                            (self.pdf_shape[0] + self.pdf_shape[1])**2)) * \
                   (self.pdf_limits[1] - self.pdf_limits[0])

        self.var = self.std**2
        self.rv = scipy.stats.beta(a=self.pdf_shape[0], b=self.pdf_shape[1])

        self.pdf_limits_norm = [-1, 1]

    def sample(self, n_samples, normalized=True):
        """
        Samples random variable

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        normalized : bool, optional, default: True
            Return normalized value

        Returns
        -------
        sample : ndarray of float [n_sample]
            Sampling points
        """
        samples = 2 * self.rv.rvs(size=n_samples) - 1

        if not normalized:
            samples = (samples + 1) / 2 * (self.pdf_limits[1] - self.pdf_limits[0]) + self.pdf_limits[0]

        return samples

    def init_basis_function(self, order):
        """
        Initializes Jacobi BasisFunction of Beta RandomParameter

        Parameters
        ----------
        order: int
            Order of basis function
        """
        return Jacobi({"i": order, "p": self.pdf_shape[0], "q": self.pdf_shape[1]})

    def pdf(self, x=None, a=None, b=None):
        """
        Calculate the probability density function of the beta distributed random variable.

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable
        a: float
            Lower limit of beta distribution
        b: float
            Upper limit of beta distribution

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density at values x
        """

        p = self.pdf_shape[0]
        q = self.pdf_shape[1]

        if a is None:
            a = self.pdf_limits[0]

        if b is None:
            b = self.pdf_limits[1]

        if x is None:
            x = np.linspace(a, b, 200)

        y = np.zeros(x.shape)

        mask = np.logical_and(a < x, x < b)

        y[mask] = (scipy.special.gamma(p) * scipy.special.gamma(q) / scipy.special.gamma(p + q)
                   * (b - a) ** (p + q - 1)) ** (-1) * (x[mask] - a) ** (p - 1) * (b - x[mask]) ** (q - 1)

        return x, y

    def pdf_norm(self, x=None):
        """
        Calculate the probability density function of the normalized beta distributed random variable in interval
        [-1, 1].

        pdf = Beta.pdf_norm(x)

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density at values x
        """

        if x is None:
            x = np.linspace(-1, 1, 200)

        x, y = self.pdf(x, a=-1., b=1.)

        return x, y

    def icdf(self, p):
        """
        Inverse cumulative density function [0, 1]

        Parameters
        ----------
        p: ndarray of float [n_p]
            Cumulative probability

        Returns
        -------
        x: ndarray of float [n_p]
            Sample value of the random variable such that the probability of the variable being less than or equal
            to that value equals the given probability.
        """
        x = 2 * self.rv.ppf(p.flatten()) - 1

        return x

    def cdf_norm(self, x):
        """
        Cumulative density function defined in normalized parameter space [-1, 1].

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable (normalized) [-1, 1]

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        c = self.rv.cdf((x.flatten() + 1) / 2)

        return c

    def cdf(self, x):
        """
        Cumulative density function.

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        c = self.rv.cdf((x.flatten() - self.pdf_limits[0]) / (self.pdf_limits[1] - self.pdf_limits[0]))

        return c


class Norm(RandomParameter):
    """
    Normal distributed random variable sub-class

    Parameters
    ----------
    pdf_shape: list of float [2]
        Shape parameters of normal distributed random variable [mean, std]
    p_perc: float, optional, default=0.9973
        Probability of percentile, where infinite distributions are cut off
        (default value corresponds to 6 sigma)

    Notes
    -----
    Probability density function

    .. math::

       p(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp\\left({-\\frac{(x-\\mu)^2}{2\\sigma^2}}\\right)

    Examples
    --------
    >>> import pygpc
    >>> pygpc.RandomParameter.Norm(pdf_shape=[0.1, 0.15])
    """

    def __init__(self, pdf_shape, p_perc=0.9973):
        """
        Constructor; Initializes normal distributed random variable
        """

        self.x_perc = [None, None]
        self.x_perc[0] = scipy.stats.norm().ppf(0.5*(1-p_perc)) * pdf_shape[1] + pdf_shape[0]
        self.x_perc[1] = scipy.stats.norm().ppf(0.5*(1+p_perc)) * pdf_shape[1] + pdf_shape[0]

        self.x_perc_norm = [None, None]
        self.x_perc_norm[0] = scipy.stats.norm().ppf(0.5 * (1 - p_perc))
        self.x_perc_norm[1] = scipy.stats.norm().ppf(0.5 * (1 + p_perc))

        super(Norm, self).__init__(pdf_type='norm',
                                   pdf_shape=pdf_shape,
                                   pdf_limits=[self.x_perc[0], self.x_perc[1]])

        self.p_perc = p_perc
        self.mean = self.pdf_shape[0]
        self.std = self.pdf_shape[1]
        self.var = self.std ** 2
        self.pdf_limits_norm = [self.x_perc_norm[0], self.x_perc_norm[1]]
        self.rv = scipy.stats.norm()

    def sample(self, n_samples, normalized=True):
        """
        Samples random variable

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        normalized : bool, optional, default: True
            Return normalized value

        Returns
        -------
        sample : ndarray of float [n_sample]
            Sampling points
        """
        samples = self.rv.rvs(size=n_samples)

        if not normalized:
            samples = samples * self.pdf_shape[1] + self.pdf_shape[0]

        return samples

    @staticmethod
    def init_basis_function(order):
        """
        Initializes Hermite BasisFunction of Norm RandomParameter

        Parameters
        ----------
        order: int
            Order of basis function
        """
        return Hermite({"i": order})

    def pdf(self, x=None):
        """
        Calculate the probability density function of the normal distributed random variable.

        pdf = Norm.pdf(x)

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density
        """
        if x is None:
            x = np.linspace(self.x_perc[0], self.x_perc[1], 200)

        y = scipy.stats.norm.pdf(x, loc=self.pdf_shape[0], scale=self.pdf_shape[1])

        return x, y

    def pdf_norm(self, x=None):
        """
        Calculate the probability density function of the normalized normal distributed random variable
        (zero mean, std 1).

        pdf = Norm.pdf_norm(x)

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable in interval [-1, 1]

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density
        """
        if x is None:
            x = np.linspace(self.x_perc_norm[0], self.x_perc_norm[1], 200)

        y = scipy.stats.norm.pdf(x, loc=0, scale=1.)

        return x, y

    def icdf(self, p):
        """
        Inverse cumulative density function [0, 1]

        Parameters
        ----------
        p: ndarray of float [n_p]
            Cumulative probability

        Returns
        -------
        x: ndarray of float [n_p]
            Sample value of the random variable such that the probability of the variable being less than or equal
            to that value equals the given probability.
        """
        # transform probabilities to perc constraint
        p = self.p_perc * p + (1 - self.p_perc)/2

        # icdf
        x = self.rv.ppf(p.flatten())

        return x

    def cdf_norm(self, x):
        """
        Cumulative density function defined in normalized parameter space (mean=0, std=1).

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable (normalized, mean=0, std=1)

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        c = self.rv.cdf(x.flatten())

        return c

    def cdf(self, x):
        """
        Cumulative density function.

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        n = scipy.stats.norm(loc=self.pdf_shape[0], scale=self.pdf_shape[1])
        c = n.cdf(x.flatten())

        return c


class Gamma(RandomParameter):
    """
    Gamma distributed random variable sub-class

    Parameters
    ----------
    pdf_shape: list of float [3]
        Shape parameters of gamma distributed random variable [shape, rate, loc] (=[alpha, beta, location])
    p_perc: float, optional, default=0.9973
        Probability of percentile, where infinite distributions are cut off
        (default value corresponds to 6 sigma from normal distribution)

    Notes
    -----
    Probability density function:

    .. math::

       p(x) = \\frac{\\beta^{\\alpha}}{\\Gamma(\\alpha)}x^{\\alpha-1}e^{\\beta x}

    Examples
    --------
    >>> import pygpc
    >>> pygpc.RandomParameter.Gamma(pdf_shape=[5, 2, 1.2])
    """
    def __init__(self, pdf_shape, p_perc=0.9973):
        """
        Constructor; Initializes gamma distributed random variable
        """

        self.x_perc = scipy.stats.gamma.ppf(p_perc,
                                            a=pdf_shape[0],
                                            loc=pdf_shape[2],
                                            scale=1 / pdf_shape[1])

        self.x_perc_norm = scipy.stats.gamma.ppf(p_perc,
                                                 a=pdf_shape[0],
                                                 loc=0.,
                                                 scale=1.)

        super(Gamma, self).__init__(pdf_type='gamma',
                                    pdf_shape=pdf_shape,
                                    pdf_limits=[pdf_shape[2], self.x_perc])

        self.p_perc = p_perc

        self.mean = self.pdf_shape[0] / self.pdf_shape[1] + self.pdf_shape[2]

        self.std = np.sqrt(self.pdf_shape[0] / self.pdf_shape[1]**2)

        self.var = self.std**2

        self.pdf_limits_norm = [0, self.x_perc_norm]

    def sample(self, n_samples, normalized=True):
        """
        Samples random variable

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        normalized : bool, optional, default: True
            Return normalized value

        Returns
        -------
        sample : ndarray of float [n_sample]
            Sampling points
        """

        samples = scipy.stats.gamma.rvs(size=n_samples,
                                        a=self.pdf_shape[0],
                                        loc=0.,
                                        scale=1.)

        if not normalized:
            samples = samples + self.pdf_shape[2]

        return samples

    def init_basis_function(self, order):
        """
        Initializes Jacobi BasisFunction of Beta RandomParameter

        Parameters
        ----------
        order: int
            Order of basis function
        """
        return Laguerre({"i": order, "alpha": self.pdf_shape[0]-1, "beta": self.pdf_shape[1]})

    def pdf(self, x=None):
        """
        Calculate the probability density function of the beta distributed random variable.

        pdf = Gamma.pdf(x)

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density at values x
        """

        a = self.pdf_shape[0]
        b = self.pdf_shape[1]
        loc = self.pdf_shape[2]

        if x is None:
            x = np.linspace(0, self.x_perc, 200)

        y = scipy.stats.gamma.pdf(x, a=a, loc=loc, scale=1/b)
        # y = b**a/scipy.special.gamma(a) * (x-loc)**(a-1) * np.exp(-b*(x-loc))

        return x, y

    def pdf_norm(self, x=None):
        """
        Calculate the probability density function of the normalized gamma distributed random variable in interval
        [-1, 1].

        pdf = Gamma.pdf_norm(x)

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable in interval [-1, 1]

        Returns
        -------
        pdf: ndarray of float [n_x]
            Probability density at values x
        """
        if x is None:
            x = np.linspace(0, self.x_perc_norm, 200)

        y = scipy.stats.gamma.pdf(x, a=self.pdf_shape[0], loc=0, scale=1.)

        return x, y

    def icdf(self, p):
        """
        Inverse cumulative density function [0, 1]

        Parameters
        ----------
        p: ndarray of float [n_p]
            Cumulative probability

        Returns
        -------
        x: ndarray of float [n_p]
            Sample value of the random variable such that the probability of the variable being less than or equal
            to that value equals the given probability.
        """

        # transform probabilities to perc constraint
        p = self.p_perc * p

        # icdf
        x = scipy.stats.gamma.ppf(p.flatten(),
                                  a=self.pdf_shape[0],
                                  loc=0.,
                                  scale=1.)

        return x

    def cdf_norm(self, x):
        """
        Cumulative density function defined in normalized parameter space [0, inf].

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable (normalized) [0, inf]

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        c = scipy.stats.gamma.cdf(x.flatten(),
                                  a=self.pdf_shape[0],
                                  loc=0.,
                                  scale=1.)

        return c

    def cdf(self, x):
        """
        Cumulative density function.

        Parameters
        ----------
        x: ndarray of float [n_x]
            Values of random variable

        Returns
        -------
        cdf: ndarray of float [n_x]
            Cumulative density at values x [0, 1]
        """
        c = scipy.stats.gamma.cdf(x.flatten(),
                                  a=self.pdf_shape[0],
                                  scale=1/self.pdf_shape[1],
                                  loc=self.pdf_shape[2])

        return c
