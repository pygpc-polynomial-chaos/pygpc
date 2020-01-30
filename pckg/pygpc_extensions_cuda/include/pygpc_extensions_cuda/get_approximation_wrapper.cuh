#ifndef PYGPC_EXTENSIONS_GET_APPROXIMATION_WRAPPER_H
#define PYGPC_EXTENSIONS_GET_APPROXIMATION_WRAPPER_H


int get_approximation_wrapper(double* ptr_arguments,
    double* ptr_poly_coeffs, double* ptr_gpc_coeffs, double* ptr_result,
    npy_intp n_arguments, npy_intp n_dim, npy_intp n_basis,
    npy_intp n_poly_coeffs, npy_intp n_gpc_coeffs);


#endif
