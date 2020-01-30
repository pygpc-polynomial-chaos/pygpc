#include <numpy/arrayobject.h>


#include "pygpc_extensions_cuda/get_approximation.cuh"


int get_approximation_wrapper(double* ptr_arguments,
    double* ptr_poly_coeffs, double* ptr_gpc_coeffs, double* ptr_result,
    npy_intp n_arguments, npy_intp n_dim, npy_intp n_basis,
    npy_intp n_poly_coeffs, npy_intp n_gpc_coeffs)
{
    get_approximation_t<double, npy_intp>(ptr_arguments, ptr_poly_coeffs,
        ptr_gpc_coeffs, ptr_result, n_arguments, n_dim, n_basis, n_poly_coeffs,
        n_gpc_coeffs);

    return 0;
}
