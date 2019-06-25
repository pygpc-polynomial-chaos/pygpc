import numpy as np
cimport numpy as np

cdef extern from "gpc.h":
    int gpc(double* polynomial_coeffs, double* x, double* gpc_matrix, size_t n_psi, size_t n_var, size_t n_x)

cdef int n_var = 0
cdef int n_xi = 0

def calc_gpc_matrix_cpu(np.ndarray[double, ndim=1, mode="c"] polynomial_coeffs, np.ndarray[double, ndim=2, mode="c"] x, np.ndarray[double, ndim=2, mode="c"] gpc_matrix, int n_psi):
    n_var = x.shape[1]
    n_x = x.shape[0]
    gpc(<double*> polynomial_coeffs.data, <double*> x.data, <double*> gpc_matrix.data, <size_t> n_psi, <size_t> n_var, <size_t> n_x)