#ifndef PYGPC_EXTENSIONS_CUDA_CREATE_GPC_MATRIX_H
#define PYGPC_EXTENSIONS_CUDA_CREATE_GPC_MATRIX_H


#include <cuda.h>
#include <cuda_runtime_api.h>


template<typename T, typename U>
__global__ void create_gpc_matrix_cuda_tk(T* ptr_arguments, T* ptr_coeffs,
    T* ptr_result, U n_arguments, U n_dim, U n_basis, U n_grad)
{
    
    U i_arguments = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_arguments < n_arguments) {
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
}

template<typename T, typename U>
int create_gpc_matrix_cuda_t(T* ptr_arguments, T* ptr_coeffs,
    T* ptr_result, U n_arguments, U n_dim, U n_basis, U n_grad, U n_coeffs)
{
    constexpr U n_blocks = 512;

    dim3 block_dim(n_blocks);
    dim3 grid_dim((n_arguments/n_blocks)+1);

    T* dptr_arguments = NULL;
    T* dptr_coeffs = NULL;
    T* dptr_result = NULL;

    cudaMalloc((T**) &dptr_arguments, n_arguments*sizeof(T));
    cudaMalloc((T**) &dptr_coeffs, n_coeffs*sizeof(T));
    cudaMalloc((T**) &dptr_result, n_basis*n_arguments*sizeof(T));

    cudaMemcpy(dptr_arguments, ptr_arguments, n_arguments*sizeof(T),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dptr_arguments, ptr_arguments, n_arguments*sizeof(T),
        cudaMemcpyHostToDevice);
    
    create_gpc_matrix_cuda_tk<T,U><<<grid_dim,block_dim>>>(dptr_arguments,
        dptr_coeffs, dptr_result, n_arguments, n_dim, n_basis, n_grad);

    cudaMemcpy(ptr_result, dptr_result, n_basis*n_arguments*sizeof(T),
        cudaMemcpyDeviceToHost);

    cudaFree(dptr_arguments);
    cudaFree(dptr_coeffs);
    cudaFree(dptr_result);

    return 0;
}

#endif
