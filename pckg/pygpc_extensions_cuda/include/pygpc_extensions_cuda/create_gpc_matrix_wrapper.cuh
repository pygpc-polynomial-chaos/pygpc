#ifndef PYGPC_EXTENSIONS_CUDA_CREATE_GPC_MATRIX_WRAPPER_H
#define PYGPC_EXTENSIONS_CUDA_CREATE_GPC_MATRIX_WRAPPER_H


#include <numpy/arrayobject.h>


int create_gpc_matrix_wrapper(double* ptr_arguments, double* ptr_coeffs,
    double* ptr_result, npy_intp n_arguments, npy_intp n_dim, npy_intp n_basis,
    npy_intp n_grad, npy_intp n_coeffs);


#endif
