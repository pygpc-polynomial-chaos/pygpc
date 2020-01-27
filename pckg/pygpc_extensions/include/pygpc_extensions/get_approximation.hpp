#ifndef PYGPC_EXTENSIONS_GET_APPROXIMATION_H
#define PYGPC_EXTENSIONS_GET_APPROXIMATION_H

#include <cstdio>

template<typename T>
int get_approximation_cpu_t(T* ptr_arguments, T* ptr_poly_coeffs,
    T* ptr_gpc_coeffs, T* ptr_result, long int n_arguments, long int n_dim,
    long int n_basis, long int n_gpc_coeffs) {
    
    for(long int i_arguments = 0; i_arguments < n_arguments; ++i_arguments) {
        long int i_basis = 0;
        T* local_ptr_poly_coeffs = ptr_poly_coeffs;
        while(i_basis != n_basis) {
            long int i_dim = 0;
            T accumulated_result = 1;
            while(i_dim != n_dim) {
                // get argument
                T argument = ptr_arguments[i_arguments * n_dim + i_dim];
                // get order of polynomial
                // then to to first (highest) coefficient 
                long int n_order = static_cast<long int>
                    (*local_ptr_poly_coeffs++);
                // initialize result variable with highest coefficient
                // then go to next coefficient
                T evaluation_result = *local_ptr_poly_coeffs++;
                // use horners method to evaluate the polynomial
                for(long int i_coeff = 0; i_coeff < n_order; ++i_coeff) {
                    evaluation_result = evaluation_result*argument +
                        *local_ptr_poly_coeffs++; 
                }
                // accumulate to overall result
                accumulated_result *= evaluation_result;
                // increment dimension counter
                i_dim++;
            }
            // multiply accumulated_result with row of gpc coefficient matrix
            for (long int i_gpc_coeffs = 0; i_gpc_coeffs < n_gpc_coeffs;
                ++i_gpc_coeffs) {
                ptr_result[i_arguments * n_gpc_coeffs + i_gpc_coeffs] += 
                    ptr_gpc_coeffs[i_basis * n_gpc_coeffs + i_gpc_coeffs] *
                    accumulated_result;
            }
            // increment basis counter
            i_basis++;
        }
    }
    return 0;
}

template<typename T>
int get_approximation_omp_t(T* ptr_arguments, T* ptr_poly_coeffs,
    T* ptr_gpc_coeffs, T* ptr_result, long int n_arguments, long int n_dim,
    long int n_basis, long int n_gpc_coeffs) {
    
    #pragma omp parallel for schedule(static)
    for(long int i_arguments = 0; i_arguments < n_arguments; ++i_arguments) {
        long int i_basis = 0;
        T* local_ptr_poly_coeffs = ptr_poly_coeffs;
        while(i_basis != n_basis) {
            long int i_dim = 0;
            T accumulated_result = 1;
            while(i_dim != n_dim) {
                // get argument
                T argument = ptr_arguments[i_arguments * n_dim + i_dim];
                // get order of polynomial
                // then to to first (highest) coefficient 
                long int n_order = static_cast<long int>
                    (*local_ptr_poly_coeffs++);
                // initialize result variable with highest coefficient
                // then go to next coefficient
                T evaluation_result = *local_ptr_poly_coeffs++;
                // use horners method to evaluate the polynomial
                for(long int i_coeff = 0; i_coeff < n_order; ++i_coeff) {
                    evaluation_result = evaluation_result*argument +
                        *local_ptr_poly_coeffs++; 
                }
                // accumulate to overall result
                accumulated_result *= evaluation_result;
                // increment dimension counter
                i_dim++;
            }
            // multiply accumulated_result with row of gpc coefficient matrix
            for (long int i_gpc_coeffs = 0; i_gpc_coeffs < n_gpc_coeffs;
                ++i_gpc_coeffs) {
                ptr_result[i_arguments * n_gpc_coeffs + i_gpc_coeffs] += 
                    ptr_gpc_coeffs[i_basis * n_gpc_coeffs + i_gpc_coeffs] *
                    accumulated_result;
            }
            // increment basis counter
            i_basis++;
        }
    }
    return 0;
}


#endif
