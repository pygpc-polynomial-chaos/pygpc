#include <numpy/arrayobject.h>


#include "pygpc_extensions_cuda/create_gpc_matrix.cuh"


int create_gpc_matrix_wrapper(double* ptr_arguments, double* ptr_coeffs,
    double* ptr_result, npy_intp n_arguments, npy_intp n_dim, npy_intp n_basis,
    npy_intp n_grad, npy_intp n_coeffs)
{
    create_gpc_matrix_t<double, npy_intp>(ptr_arguments, ptr_coeffs,
        ptr_result, n_arguments, n_dim, n_basis, n_grad, n_coeffs);

    return 0;
}
