#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from .RandomParameter import *


class Problem:
    """
    Data wrapper for the gpc problem containing the model to investigate.

    Parameters
    ----------
    model: Model object
        Model object instance of model to investigate (derived from AbstractModel class, implemented by user)
    parameters: OrderedDict
        Dictionary containing the model parameters as keys:
        - constants: value as float or list of float
        - random variable: namedtuple("RandomParameter", "pdf_type pdf_shape pdf_limits")
          (e.g. RandomParameter("beta", [5, 5], [0.15, 0.45]))

    Notes
    -----
    Add Attributes:

    random_vars: [dim] list of str
        String labels of the random variables
    N_out: int
        Number of output coefficients
    dim: int
        Number of uncertain parameters to process
    pdf_type: [dim] list of str
        Type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        Shape parameters of pdfs
        beta-dist:   [[], ... [alpha, beta], ..., []]
        normal-dist: [[], ... [mean, std], ..., []]
    pdf_limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[], ... [min, max], ..., []]
        normal-dist: [[], ... [0, 0], ..., []] (not used)

    Examples
    --------
    Setup model and specify parameters of gPC problem

    >>> import pygpc
    >>> from collections import OrderedDict
    >>>
    >>> # Define model
    >>> model = pygpc.testfunctions.SphereModel
    >>>
    >>> # Define Problem
    >>> parameters = OrderedDict()  # we must use an ordered dict form the start, otherwise the order will be mixed
    >>> parameters["R"] = [80, 90, 100]                                                 # constant parameter
    >>> parameters["phi_electrode"] = 15                                                #       "
    >>> parameters["N_points"] = 201                                                    #       "
    >>> parameters["sigma_1"] = pygpc.RandomParameter.Beta(pdf_shape=[5, 5], pdf_limits=[0.15, 0.45])  # random variable
    >>> parameters["sigma_2"] = pygpc.RandomParameter.Beta(pdf_shape=[1, 3], pdf_limits=[0.01, 0.02])  #       "
    >>> parameters["sigma_3"] = pygpc.RandomParameter.Norm(pdf_shape=[2, 2])                           #       "
    >>> problem = pygpc.Problem(model, parameters)
    """
    def __init__(self, model, parameters):
        assert(isinstance(parameters, OrderedDict))

        self.model = model
        self.parameters = parameters
        self.parameters_random = []
        self.params_names = self.parameters.keys()
        self.dim = 0
        self.random_vars = []
        self.pdf_shape = []
        self.pdf_type = []
        self.pdf_limits = []

        for i_p, p in enumerate(self.params_names):
            if isinstance(parameters[p], RandomParameter):
                self.dim += 1
                self.parameters_random.append(parameters[p])
                self.random_vars.append(p)
                self.pdf_shape.append(parameters[p].pdf_shape)
                self.pdf_type.append(parameters[p].pdf_type)
                self.pdf_limits.append(parameters[p].pdf_limits)

        self.mean_random_vars = self.get_mean_random_vars_input()

    def get_mean_random_vars_input(self):
        """
        Determine the average values of the input random variables from their pdfs.

        Returns
        -------
        mean_random_vars: [N_random_vars] np.ndarray
            Average values of the input random variables.
        """
        mean_random_vars = np.zeros(self.dim)

        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == 'normal':
                mean_random_vars[i_dim] = self.pdf_shape[i_dim][0]

            if self.pdf_type[i_dim] == 'beta':
                mean_random_vars[i_dim] = (float(self.pdf_shape[i_dim][0]) /
                                           (self.pdf_shape[i_dim][0] + self.pdf_shape[i_dim][1])) * \
                                          (self.pdf_limits[i_dim][1] - self.pdf_limits[i_dim][0]) + \
                                          (self.pdf_limits[i_dim][0])

        return mean_random_vars
