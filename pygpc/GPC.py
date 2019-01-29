# -*- coding: utf-8 -*-
import copy
from .Grid import *
from .misc import get_cartesian_product
import numpy as np
import fastmat as fm
import scipy.stats
import ctypes


class GPC(object):
    """
    General gPC base class

    Attributes
    ----------
    problem: Problem class instance
        GPC Problem under investigation
    basis: Basis class instance
        Basis of the gPC including BasisFunctions
    grid: Grid class instance
        Grid of the derived gPC approximation
    gpc_matrix: [N_samples x N_poly] np.ndarray
        generalized polynomial chaos matrix
    gpc_matrix_inv: [N_poly x N_samples] np.ndarray
        pseudo inverse of the generalized polynomial chaos matrix
    nan_elm: ndarray of int
        Indices of NaN elements of model output
    gpc_matrix_coords_id: list of UUID4()
        UUID4() IDs of grid points the gPC matrix derived with
    gpc_matrix_b_id: list of UUID4()
        UUID4() IDs of basis functions the gPC matrix derived with
    solver: str
        Default solver to determine the gPC coefficients (can be chosen during GPC.solve)
        - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
        - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
        - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
    gpu: bool
        Flag to execute the calculation on the gpu
    verbose: bool
        boolean value to determine if to print out the progress into the standard output
    fn_results : string, optional, default=None
        If provided, model evaluations are saved in fn_results.hdf5 file and gpc object in fn_results.pkl file
    n_cpu : int, optional, default=1
        Number of threads to use for parallel evaluation of the model function.
    """

    def __init__(self, problem, fn_results):

        # objects
        self.problem = problem
        self.basis = None
        self.grid = None

        # arrays
        self.gpc_matrix = None
        self.gpc_matrix_inv = None
        self.nan_elm = []
        self.gpc_matrix_coords_id = None
        self.gpc_matrix_b_id = None

        # options
        self.solver = None
        self.settings = None
        self.gpu = None
        self.verbose = True
        self.fn_results = fn_results
        self.n_cpu = 1

    def init_gpc_matrix(self):
        """
        Sets self.gpc_matrix with given self.basis and self.grid
        """

        self.gpc_matrix = self.calc_gpc_matrix(self.basis.b, self.grid.coords_norm)
        self.gpc_matrix_coords_id = copy.deepcopy(self.grid.coords_id)
        self.gpc_matrix_b_id = copy.deepcopy(self.basis.b_id)

    def calc_gpc_matrix(self, b, x, verbose=False):
        """
        Construct the gPC matrix.

        gpc_matrix = calc_gpc_matrix(b, x)

        Parameters
        ----------
        b: list of BasisFunction object instances [n_basis x n_dim]
            Parameter wise basis function objects used in gPC (Basis.b)
            Multiplying all elements in a row at location xi = (x1, x2, ..., x_dim) yields the global basis function.
        x: ndarray of float [n_x x n_dim]
            Coordinates of x = (x1, x2, ..., x_dim) where the rows of the gPC matrix are evaluated (normalized [-1, 1])
        verbose: bool
            boolean value to determine if to print out the progress into the standard output

        Returns
        -------
        gpc_matrix: ndarray of float [n_x x n_basis]
            GPC matrix where the columns correspond to the basis functions and the rows the to the sample coordinates
        """

        iprint('Constructing gPC matrix...', verbose=verbose, tab=0)
        gpc_matrix = np.ones([x.shape[0], len(b)])

        if not self.gpu:
            for i_basis in range(len(b)):
                for i_dim in range(self.problem.dim):
                    gpc_matrix[:, i_basis] *= b[i_basis][i_dim](x[:, i_dim])

        # TODO: @Lucas: Bitte GPU support implementieren
        # else:
        #     # get parameters
        #     number_of_variables = len(self.poly[0])
        #     highest_degree = len(self.poly)
        #
        #     # handle pointer
        #     polynomial_coeffs_pointer = self.poly_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     polynomial_index_pointer = self.poly_idx_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        #     xi_pointer = self.grid.coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     polynomial_matrix_pointer = gpc_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     number_of_psi_size_t = ctypes.c_size_t(self.N_poly)
        #     number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
        #     highest_degree_size_t = ctypes.c_size_t(highest_degree)
        #     number_of_xi_size_t = ctypes.c_size_t(self.grid.coords.shape[0])
        #
        #     # handle shared object
        #     dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'gpc.so'), mode=ctypes.RTLD_GLOBAL)
        #     cuda_pce = dll.polynomial_chaos_matrix
        #     cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
        #                         [ctypes.POINTER(ctypes.c_double)] * 2 + [ctypes.c_size_t] * 4
        #
        #     # evaluate CUDA implementation
        #     cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, polynomial_matrix_pointer,
        #              number_of_psi_size_t, number_of_variables_size_t, highest_degree_size_t, number_of_xi_size_t)

        return gpc_matrix

    def get_pdf(self, coeffs, n_samples, output_idx=None):
        """ Determine the estimated pdfs of the output quantities

        pdf_x, pdf_y = SGPC.get_pdf(coeffs, n_samples, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_coeffs x n_out]
            GPC coefficients
        n_samples: int
            Number of samples used to estimate output pdfs
        output_idx: ndarray, optional, default=None [1 x n_out]
            Index of output quantities to consider (if output_idx=None, all output quantities are considered)

        Returns
        -------
        pdf_x: ndarray of float [100 x n_out]
            x-coordinates of output pdfs of output quantities
        pdf_y: ndarray of float [100 x n_out]
            y-coordinates of output pdfs (probability density of output quantity)
        """

        # handle (N,) arrays
        if len(coeffs.shape) == 1:
            n_out = 1
        else:
            n_out = coeffs.shape[1]

        # if output index array is not provided, determine pdfs of all outputs
        if not output_idx:
            output_idx = np.linspace(0, n_out - 1, n_out)
            output_idx = output_idx[np.newaxis, :]

        # sample gPC expansion
        samples_in, samples_out = self.get_samples(n_samples=n_samples, coeffs=coeffs, output_idx=output_idx)

        if n_out == 1:
            samples_out = samples_out[:, np.newaxis]

        # determine kernel density estimates using Gaussian kernel
        pdf_x = np.zeros([100, n_out])
        pdf_y = np.zeros([100, n_out])

        for i_out in range(n_out):
            kde = scipy.stats.gaussian_kde(samples_out.transpose(), bw_method=0.1 / samples_out[:, i_out].std(ddof=1))
            pdf_x[:, i_out] = np.linspace(samples_out[:, i_out].min(), samples_out[:, i_out].max(), 100)
            pdf_y[:, i_out] = kde(pdf_x[:, i_out])

        return pdf_x, pdf_y

    def get_samples(self, coeffs, n_samples, output_idx=None):
        """
        Randomly sample gPC expansion.

        x, pce = SGPC.get_pdf_mc(n_samples, coeffs, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients
        n_samples: int
            Number of random samples drawn from the respective input pdfs.
        output_idx: ndarray of int [1 x n_out] optional, default=None
            Index of output quantities to consider.

        Returns
        -------
        x: ndarray of float [n_samples x dim]
            Generated samples in normalized coordinates [-1, 1].
        pce: ndarray of float [n_samples x n_out]
            GPC approximation at points x.
        """

        # seed the random numbers generator
        np.random.seed()

        # generate temporary grid with random samples for each random input variable [n_samples x dim]
        grid = RandomGrid(parameters_random=self.problem.parameters_random,
                          options={"n_grid": n_samples, "seed": None})

        # if output index list is not provided, sample all gpc outputs
        if output_idx is None:
            n_out = 1 if coeffs.ndim == 1 else coeffs.shape[1]
            output_idx = np.arange(n_out)
            # output_idx = output_idx[np.newaxis, :]

        pce = self.get_approximation(coeffs=coeffs, x=grid.coords_norm, output_idx=output_idx)

        return grid.coords_norm, pce

    def get_approximation(self, coeffs, x, output_idx=None):
        """
        Calculates the gPC approximation in points with output_idx and normalized parameters xi (interval: [-1, 1]).

        pce = GPC.get_approximation(coeffs, x, output_idx=None)

        Parameters
        ----------
        coeffs: ndarray of float [n_basis x n_out]
            GPC coefficients for each output variable
        x: ndarray of float [n_x x n_dim]
            Coordinates of x = (x1, x2, ..., x_dim) where the rows of the gPC matrix are evaluated (normalized [-1, 1])
        output_idx: ndarray of int, optional, default=None [n_out]
            Index of output quantities to consider (Default: all).

        Returns
        -------
        pce: ndarray of float [n_x x n_out]
            GPC approximation at normalized coordinates x.
        """

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        # crop coeffs array if output index is specified
        if output_idx is not None:
            output_idx = np.array(output_idx)
            if len(output_idx.shape):
                output_idx = output_idx[np.newaxis, :]
            coeffs = coeffs[:, output_idx]

        if not self.gpu:
            # determine gPC matrix at coordinates x
            gpc_matrix = self.calc_gpc_matrix(self.basis.b, x)

            # multiply with gPC coeffs
            pce = np.matmul(gpc_matrix, coeffs)

        # TODO: @Lucas: Bitte GPU support implementieren
        # else:
        #     # initialize matrices and parameters
        #     pce = np.zeros([xi.shape[0], coeffs.shape[1]])
        #     number_of_variables = len(s.poly[0])
        #     highest_degree = len(s.poly)
        #
        #     # handle pointer
        #     polynomial_coeffs_pointer = s.poly_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     polynomial_index_pointer = s.poly_idx_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        #     xi_pointer = xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     sim_result_pointer = pce.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     sim_coeffs_pointer = coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        #     number_of_xi_size_t = ctypes.c_size_t(xi.shape[0])
        #     number_of_variables_size_t = ctypes.c_size_t(number_of_variables)
        #     number_of_psi_size_t = ctypes.c_size_t(coeffs.shape[0])
        #     highest_degree_size_t = ctypes.c_size_t(highest_degree)
        #     number_of_result_vectors_size_t = ctypes.c_size_t(coeffs.shape[1])
        #
        #     # handle shared object
        #     dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'pckg', 'pce.so'), mode=ctypes.RTLD_GLOBAL)
        #     cuda_pce = dll.polynomial_chaos_matrix
        #     cuda_pce.argtypes = [ctypes.POINTER(ctypes.c_double)] + [ctypes.POINTER(ctypes.c_int)] + \
        #                         [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_size_t] * 5
        #
        #     # evaluate CUDA implementation
        #     cuda_pce(polynomial_coeffs_pointer, polynomial_index_pointer, xi_pointer, sim_result_pointer,
        #              sim_coeffs_pointer, number_of_psi_size_t, number_of_result_vectors_size_t,
        #              number_of_variables_size_t,
        #              highest_degree_size_t, number_of_xi_size_t)

        return pce

    # TODO: @Lucas: Bitte GPU support implementieren
    def replace_gpc_matrix_samples(self, idx, seed=None):
        """
        Replace distinct sample points from the gPC matrix with new ones.

        GPC.replace_gpc_matrix_samples(idx, seed=None)

        Parameters
        ----------
        idx: ndarray of int [n_samples]
            Array of grid indices of grid.coords[idx, :] which are going to be replaced
            (rows of gPC matrix will be replaced by new ones)
        seed: float, optional, default=None
            Random seeding point
        """

        # Generate new grid points
        new_grid_points = RandomGrid(parameters_random=self.problem.parameters_random,
                                     options={"n_grid": idx.size, "seed": seed})

        # replace old grid points
        self.grid.coords[idx, :] = new_grid_points.coords
        self.grid.coords_norm[idx, :] = new_grid_points.coords_norm

        # replace old IDs of grid points with new ones
        for i in idx:
            self.grid.coords_id[i] = uuid.uuid4()
            self.gpc_matrix_coords_id[i] = copy.deepcopy(self.grid.coords_id[i])

        # determine new rows of gpc matrix and overwrite rows of gpc matrix
        self.gpc_matrix[idx, :] = self.calc_gpc_matrix(self.basis.b, new_grid_points.coords_norm)

    # TODO: @Lucas: Bitte GPU support implementieren
    def update_gpc_matrix(self):
        """
        Update gPC matrix according to existing self.grid and self.basis.

        Call this method when self.gpc_matrix does not fit to self.grid and self.basis objects anymore
        The old gPC matrix with their self.gpc_matrix_b_id and self.gpc_matrix_coords_id is compared
        to self.basis.b_id and self.grid.coords_id. New rows and columns are computed when differences are found.
        """

        if not self.gpu:
            # initialize updated gpc matrix
            gpc_matrix_updated = np.zeros((len(self.grid.coords_id), len(self.basis.b_id)))

            # # determine indices of new basis functions and grid_points
            # idx_coords_new = [i for i, _id in enumerate(self.grid.coords_id) if _id not in self.gpc_matrix_coords_id]
            # idx_basis_new = [i for i, _id in enumerate(self.basis.b_id) if _id not in self.gpc_matrix_b_id]

            # determine indices of old grid points in updated gpc matrix
            idx_coords_old = np.empty(len(self.gpc_matrix_coords_id))*np.nan
            for i, coords_id_old in enumerate(self.gpc_matrix_coords_id):
                for j, coords_id_new in enumerate(self.grid.coords_id):
                    if coords_id_old == coords_id_new:
                        idx_coords_old[i] = j
                        break

            # determine indices of old basis functions in updated gpc matrix
            idx_b_old = np.empty(len(self.gpc_matrix_b_id))*np.nan
            for i, b_id_old in enumerate(self.gpc_matrix_b_id):
                for j, b_id_new in enumerate(self.basis.b_id):
                    if b_id_old == b_id_new:
                        idx_b_old[i] = j
                        break

            # filter out non-existent rows and columns
            self.gpc_matrix = self.gpc_matrix[~np.isnan(idx_coords_old), :]
            self.gpc_matrix = self.gpc_matrix[:, ~np.isnan(idx_b_old)]

            idx_coords_old = idx_coords_old[~np.isnan(idx_coords_old)].astype(int)
            idx_b_old = idx_b_old[~np.isnan(idx_b_old)].astype(int)

            # indices of new coords and basis in updated gpc matrix (values have to be computed there)
            idx_coords_new = np.array(list(set(np.arange(len(self.grid.coords_id))) - set(idx_coords_old))).astype(int)
            idx_b_new = np.array(list(set(np.arange(len(self.basis.b_id))) - set(idx_b_old))).astype(int)

            # write old results at correct location in updated gpc matrix
            idx = get_cartesian_product([idx_coords_old, idx_b_old]).astype(int)
            idx_row = np.reshape(idx[:, 0], self.gpc_matrix.shape).astype(int)
            idx_col = np.reshape(idx[:, 1], self.gpc_matrix.shape).astype(int)

            gpc_matrix_updated[idx_row, idx_col] = self.gpc_matrix

            # determine new columns (new basis functions) with old grid
            idx = get_cartesian_product([idx_coords_old, idx_b_new]).astype(int)
            if idx.any():
                iprint('Adding columns to gPC matrix...', tab=0, verbose=True)

                idx_row = np.reshape(idx[:, 0], (idx_coords_old.size, idx_b_new.size)).astype(int)
                idx_col = np.reshape(idx[:, 1], (idx_coords_old.size, idx_b_new.size)).astype(int)

                gpc_matrix_updated[idx_row, idx_col] = self.calc_gpc_matrix(b=[self.basis.b[i] for i in idx_b_new],
                                                                            x=self.grid.coords_norm[idx_coords_old, :],
                                                                            verbose=False)

            # determine new rows (new grid points) with all basis functions
            idx = get_cartesian_product([idx_coords_new, np.arange(len(self.basis.b))]).astype(int)
            if idx.any():
                iprint('Adding rows to gPC matrix...', tab=0, verbose=True)

                idx_row = np.reshape(idx[:, 0], (idx_coords_new.size, len(self.basis.b))).astype(int)
                idx_col = np.reshape(idx[:, 1], (idx_coords_new.size, len(self.basis.b))).astype(int)

                gpc_matrix_updated[idx_row, idx_col] = self.calc_gpc_matrix(b=self.basis.b,
                                                                            x=self.grid.coords_norm[idx_coords_new, :],
                                                                            verbose=False)

            # overwrite old attributes
            self.gpc_matrix = gpc_matrix_updated
            self.gpc_matrix_coords_id = copy.deepcopy(self.grid.coords_id)
            self.gpc_matrix_b_id = copy.deepcopy(self.basis.b_id)

    def solve(self, sim_results, solver=None, settings=None, gpc_matrix=None, verbose=False):
        """
        Determines gPC coefficients

        Parameters
        ----------
        sim_results : [N_grid x N_out] np.ndarray of float
            results from simulations with N_out output quantities
        solver : str
            Solver to determine the gPC coefficients
            - 'Moore-Penrose' ... Pseudoinverse of gPC matrix (SGPC.Reg, EGPC)
            - 'OMP' ... Orthogonal Matching Pursuit, sparse recovery approach (SGPC.Reg, EGPC)
            - 'NumInt' ... Numerical integration, spectral projection (SGPC.Quad)
        settings : dict
            Solver settings
            - 'Moore-Penrose' ... None
            - 'OMP' ... {"n_coeffs_sparse": int} Number of gPC coefficients != 0 or "sparsity": float 0...1
            - 'NumInt' ... None
        gpc_matrix : ndarray of float [n_grid x n_basis], optional, default: self.gpc_matrix
            GPC matrix to invert
        verbose : bool
            boolean value to determine if to print out the progress into the standard output

        Returns
        -------
        coeffs: ndarray of float [n_coeffs x n_out]
            gPC coefficients
        """
        if gpc_matrix is None:
            gpc_matrix = self.gpc_matrix

        # use default solver if not specified
        if solver is None:
            solver = self.solver

        # use default solver settings if not specified
        if solver is None:
            settings = self.settings

        iprint("Determine gPC coefficients using '{}' solver...".format(solver), tab=0, verbose=verbose)

        if solver == 'Moore-Penrose':
            # determine pseudoinverse of gPC matrix
            self.gpc_matrix_inv = np.linalg.pinv(gpc_matrix)

            try:
                coeffs = np.dot(self.gpc_matrix_inv, sim_results)
            except ValueError:
                raise AttributeError("Please check format of parameter sim_results: [n_grid x n_out] np.ndarray.")

        elif solver == 'OMP':
            # transform gPC matrix to fastmat format
            gpc_matrix_fm = fm.Matrix(gpc_matrix)

            if sim_results.ndim == 1:
                sim_results = sim_results[:, np.newaxis]

            # determine gPC-coefficients of extended basis using OMP
            if "n_coeffs_sparse" in settings.keys():
                n_coeffs_sparse = int(settings["n_coeffs_sparse"])
            elif "sparsity" in settings.keys():
                n_coeffs_sparse = int(np.ceil(gpc_matrix.shape[1]*settings["sparsity"]))
            else:
                raise AttributeError("Please specify 'n_coeffs_sparse' or 'sparsity' in solver settings dictionary!")

            coeffs = fm.algs.OMP(gpc_matrix_fm, sim_results, n_coeffs_sparse)

        elif solver == 'NumInt':
            # check if quadrature rule (grid) fits to the probability density distribution (pdf)
            grid_pdf_fit = True
            for i_p, p in enumerate(self.problem.parameters_random):
                if self.problem.parameters_random[p].pdf_type == 'beta':
                    if not (self.grid.grid_type[i_p] == 'jacobi'):
                        grid_pdf_fit = False
                        break
                elif self.problem.parameters_random[p].pdf_type in ['norm', 'normal']:
                    if not (self.grid.grid_type[i_p] == 'hermite'):
                        grid_pdf_fit = False
                        break

            # if not, calculate joint pdf
            if not grid_pdf_fit:
                joint_pdf = np.ones(self.grid.coords_norm.shape)

                for i_p, p in enumerate(self.problem.parameters_random):
                    joint_pdf[:, i_p] = \
                        self.problem.parameters_random[p].pdf_norm(x=self.grid.coords_norm[:, i_p])

                joint_pdf = np.array([np.prod(joint_pdf, axis=1)]).transpose()

                # weight sim_results with the joint pdf
                sim_results = sim_results * joint_pdf * 2 ** self.problem.dim

            # scale rows of gpc matrix with quadrature weights
            gpc_matrix_weighted = np.dot(np.diag(self.grid.weights), gpc_matrix)

            # determine gpc coefficients [n_coeffs x n_output]
            coeffs = np.dot(sim_results.transpose(), gpc_matrix_weighted).transpose()

        else:
            raise AttributeError("Unknown solver: '{}'!")

        return coeffs
