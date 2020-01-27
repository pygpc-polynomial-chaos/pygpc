#ifndef PYGPC_EXTENSIONS_CREATE_GPC_MATRIX_H
#define PYGPC_EXTENSIONS_CREATE_GPC_MATRIX_H


template<typename T, typename U>
int create_gpc_matrix_omp_t(T* ptr_arguments, T* ptr_coeffs, T* ptr_result,
    U n_arguments, U n_dim, U n_basis, U n_grad)
{
    
    #pragma omp parallel for schedule(static)
    for(U i_arguments = 0; i_arguments < n_arguments; ++i_arguments) {
        U i_basis = 0;
        T* local_ptr_coeffs = ptr_coeffs;
        while(i_basis != n_basis) {
            for(U i_grad = 0; i_grad < n_grad; i_grad++) {
                U i_dim = 0;
                T accumulated_result = 1;
                while(i_dim != n_dim) {
                    // get argument
                    T argument = ptr_arguments[i_arguments * n_dim + i_dim];
                    // get order of polynomial
                    // then to to first (highest) coefficient 
                    U n_order = static_cast<U>
                        (*local_ptr_coeffs++);
                    // initialize result variable with highest coefficient
                    // then go to next coefficient
                    T evaluation_result = *local_ptr_coeffs++;
                    // use horners method to evaluate the polynomial
                    for(U i_coeff = 0; i_coeff < n_order; ++i_coeff) {
                        evaluation_result = evaluation_result*argument +
                            *local_ptr_coeffs++; 
                    }
                    // accumulate to overall result
                    accumulated_result *= evaluation_result;
                    // increment dimension counter
                    i_dim++;
                }
                // write result
                ptr_result[i_arguments * n_basis * n_grad + i_basis * n_grad +
                i_grad] = accumulated_result;
            }
            // increment basis counter
            i_basis++;
        }
    }
    return 0;
}

template<typename T, typename U>
int create_gpc_matrix_cpu_t(T* ptr_arguments, T* ptr_coeffs, T* ptr_result,
    U n_arguments, U n_dim, U n_basis, U n_grad)
{
    
    for(U i_arguments = 0; i_arguments < n_arguments; ++i_arguments) {
        U i_basis = 0;
        T* local_ptr_coeffs = ptr_coeffs;
        while(i_basis != n_basis) {
            for(U i_grad = 0; i_grad < n_grad; i_grad++) {
                U i_dim = 0;
                T accumulated_result = 1;
                while(i_dim != n_dim) {
                    // get argument
                    T argument = ptr_arguments[i_arguments * n_dim + i_dim];
                    // get order of polynomial
                    // then to to first (highest) coefficient 
                    U n_order = static_cast<U>
                        (*local_ptr_coeffs++);
                    // initialize result variable with highest coefficient
                    // then go to next coefficient
                    T evaluation_result = *local_ptr_coeffs++;
                    // use horners method to evaluate the polynomial
                    for(U i_coeff = 0; i_coeff < n_order; ++i_coeff) {
                        evaluation_result = evaluation_result*argument +
                            *local_ptr_coeffs++; 
                    }
                    // accumulate to overall result
                    accumulated_result *= evaluation_result;
                    // increment dimension counter
                    i_dim++;
                }
                // write result
                ptr_result[i_arguments * n_basis * n_grad + i_basis * n_grad +
                i_grad] = accumulated_result;
            }
            // increment basis counter
            i_basis++;
        }
    }
    return 0;
}


#endif
