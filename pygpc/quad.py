# -*- coding: utf-8 -*-
"""
Class that provides polynomial chaos quadratur methods
"""

from builtins import range


class Quad(gPC):
    """
    Quadratur gPC subclass

    Initialisation
    --------------
    Quad(pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None)

    Attributes
    ----------
    N_grid: int
        number of grid points
    dim: int
        number of uncertain parameters to process
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str
        string labels of the random variables

    Parameters
    ----------
    pdf_type: [dim] list of str
        type of pdf 'beta' or 'norm'
    pdf_shape: list of list of float
        shape parameters of pdfs
        beta-dist:   [[alpha], [beta]    ]
        normal-dist: [[mean],  [variance]]
    limits: list of list of float
        upper and lower bounds of random variables
        beta-dist:   [[a1 ...], [b1 ...]]
        normal-dist: [[0 ... ], [0 ... ]] (not used)
    order: [dim] list of int
        maximum individual expansion order
        generates individual polynomials also if maximum expansion order in order_max is exceeded
    order_max: int
        maximum expansion order (sum of all exponents)
        the maximum expansion order considers the sum of the orders of combined polynomials only
    interaction_order: int
        number of random variables, which can interact with each other
        all polynomials are ignored, which have an interaction order greater than the specified
    grid: grid object
        grid object generated in grid.py including grid.coords and grid.coords_norm
    random_vars: [dim] list of str, optional, default=None
        string labels of the random variables
    """

    def __init__(self, pdf_type, pdf_shape, limits, order, order_max, interaction_order, grid, random_vars=None):
        gPC.__init__(self)
        self.random_vars = random_vars
        self.pdf_type = pdf_type
        self.pdf_shape = pdf_shape
        self.limits = limits
        self.order = order
        self.order_max = order_max
        self.interaction_order = interaction_order
        self.dim = len(pdf_type)
        self.grid = grid
        self.N_grid = grid.coords.shape[0]

        # setup polynomial basis functions
        self.init_polynomial_basis()

        # construct gpc matrix [Ngrid x Npolybasis]
        self.init_gpc_matrix()

        # get mean values of input random variables
        self.mean_random_vars = self.get_mean_random_vars()

    def get_coeffs_expand(self, sim_results):
        """
        Determine the gPC coefficients by the quadrature method

        coeffs = get_coeffs_expand(self, sim_results)

        Parameters
        ----------
        sim_results: [N_grid x N_out] np.ndarray of float
            results from simulations with N_out output quantities

        Returns
        -------
        coeffs: [N_coeffs x N_out] np.ndarray of float
            gPC coefficients
        """

        vprint('Determine gPC coefficients ...', verbose=self.verbose)
        self.N_out = sim_results.shape[1]

        # check if quadrature rule (grid) fits to the probability density distribution (pdf)
        grid_pdf_fit = True
        for i_dim in range(self.dim):
            if self.pdf_type[i_dim] == 'beta':
                if not (self.grid.gridtype[i_dim] == 'jacobi'):
                    grid_pdf_fit = False
                    break
            elif (self.pdf_type[i_dim] == 'norm') or (self.pdf_type[i_dim] == 'normal'):
                if not (self.grid.gridtype[i_dim] == 'hermite'):
                    grid_pdf_fit = False
                    break

        # if not, calculate joint pdf
        if not grid_pdf_fit:
            joint_pdf = np.ones(self.grid.coords_norm.shape)

            for i_dim in range(self.dim):
                if self.pdf_type[i_dim] == 'beta':
                    joint_pdf[:, i_dim] = get_pdf_beta(self.grid.coords_norm[:, i_dim],
                                                       self.pdf_shape[0][i_dim],
                                                       self.pdf_shape[1][i_dim], -1, 1)

                if self.pdf_type[i_dim] == 'norm' or self.pdf_type[i_dim] == 'normal':
                    joint_pdf[:, i_dim] = scipy.stats.norm.pdf(self.grid.coords_norm[:, i_dim])

            joint_pdf = np.array([np.prod(joint_pdf, axis=1)]).transpose()

            # weight sim_results with the joint pdf
            sim_results = sim_results * joint_pdf * 2 ** self.dim

        # scale rows of gpc matrix with quadrature weights
        gpc_matrix_weighted = np.dot(np.diag(self.grid.weights), self.gpc_matrix)

        # determine gpc coefficients [N_coeffs x N_output]
        return np.dot(sim_results.transpose(), gpc_matrix_weighted).transpose()
