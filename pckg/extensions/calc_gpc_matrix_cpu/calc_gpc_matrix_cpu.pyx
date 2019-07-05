import numpy as np
cimport numpy as np

# define variables for static typing
cdef int n_var = 0
cdef int n_x = 0
cdef double psi = 0
cdef int i_basis = 0
cdef int i_x = 0
cdef int i_var = 0
cdef list basis_list = []


def calc_gpc_matrix_cpu(list basis, np.ndarray[double, ndim=2, mode="c"] x, np.ndarray[double, ndim=2, mode="c"] gpc_matrix, int gradient):
    
    # get number of variables and uncetain vectors
    n_var = x.shape[1]
    n_x = x.shape[0]
    
    # loop over basis
    for i_basis, basis_list in enumerate(basis):
        # loop over uncertain vectors
        for i_x in range(n_x):
                psi = 1
                # loop over variables
                for i_var in range(n_var):
                    if gradient == i_var:
                        psi *= basis_list[i_var].fun.deriv()(x[i_x, i_var])
                    else:
                        psi *= basis_list[i_var].fun(x[i_x, i_var])
                # write result back
                gpc_matrix[i_x, i_basis] = psi