#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from .RandomParameter import *
from .ValidationSet import *
from .Grid import *
from .Computation import *


class Problem:
    """
    Data wrapper for the gpc problem containing the model to investigate and the associated parameters.

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
    def __init__(self, model, parameters, validation=None):
        """
        Constructor; Initializes Problem instance

        Parameters
        ----------
        model: Model object
            Model object instance of model to investigate (derived from AbstractModel class, implemented by user)
        parameters: OrderedDict
            Dictionary containing the model parameters as keys:
            - constants: value as float or list of float
            - random variable: namedtuple("RandomParameter", "pdf_type pdf_shape pdf_limits")
              (e.g. RandomParameter("beta", [5, 5], [0.15, 0.45]))
        validation: ValidationSet object (optional)
            Object containing a set of validation points and corresponding solutions. Can be used
            to validate gpc approximation setting options["error_type"]="nrmsd".
            - grid: Grid object containing the validation points (grid.coords, grid.coords_norm)
            - results: ndarray [n_grid x n_out] results
        """
        assert(isinstance(parameters, OrderedDict))

        self.model = model                              # Model class instance
        self.parameters = parameters                    # OrderedDict of parameters (constants and random)
        self.parameters_random = OrderedDict()          # OrderedDict of parameters (random)
        self.validation = validation                    # ValidationSet object

        # extract random parameters
        for p in self.parameters:
            if isinstance(self.parameters[p], RandomParameter):
                self.parameters_random[p] = self.parameters[p]

        self.dim = len(self.parameters_random)

        # test problem definition
        self.validate()

    def validate(self):
        """
        Verifies the problem, by testing if the parameters including the random variables are defined appropriate.
        In cases, the model may not run correctly for some parameter combinations, the user may change the definition
        of the random parameters or the constants in model.validate.

        calls model.validate

        overwrites parameters
        """

        # initialize temporal model object
        m = self.model(p=self.parameters, context=None)

        # call model/problem validation
        parameters_corrected = self.model.validate(m)

        # update parameters and parameters_random in self
        if parameters_corrected is not None:
            self.parameters = parameters_corrected
            self.parameters_random = OrderedDict()

            for p in self.parameters:
                if isinstance(self.parameters[p], RandomParameter):
                    self.parameters_random[p] = self.parameters[p]

    def create_validation_set(self, n_samples, n_cpu=1):
        """
        Creates a ValidationSet instance (calls the model)

        Parameters
        ----------
        n_samples: int
            Number of sampling points contained in the validation set
        n_cpu: int
            Number of parallel function evaluations to evaluate validation set (n_cpu=0 assumes that the
            model is capable to evaluate all grid points in parallel)
        """
        # create set of validation points
        n_samples = n_samples
        grid = RandomGrid(parameters_random=self.parameters_random,
                          options={"n_grid": n_samples, "seed": None})

        # Evaluate original model at grid points
        com = Computation(n_cpu=n_cpu)
        results = com.run(model=self.model, problem=self, coords=grid.coords)

        if results.ndim == 1:
            results = results[:, np.newaxis]

        self.validation = ValidationSet(grid=grid, results=results)
