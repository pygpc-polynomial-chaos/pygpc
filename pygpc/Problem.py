#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from collections import namedtuple
import numpy as np


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
    >>> parameters["sigma_1"] = pygpc.RandomParameter("beta", [5, 5], [0.15, 0.45])     # random variable
    >>> parameters["sigma_2"] = pygpc.RandomParameter("beta", [1, 3], [0.01, 0.02])     #       "
    >>> parameters["sigma_3"] = pygpc.RandomParameter("beta", [2, 2], [0.4, 0.6])       #       "
    >>> problem = pygpc.Problem(model, parameters)
    """
    def __init__(self, model, parameters):
        assert(isinstance(parameters, OrderedDict))

        self.model = model
        self.parameters = parameters

        self.params_names = self.parameters.keys()
        self.dim = 0
        self.random_vars = []
        self.pdf_shape = []
        self.pdf_type = []
        self.pdf_limits = []

        for p in self.params_names:
            if type(self.parameters[p]) is RandomParameter:
                self.dim += 1
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
