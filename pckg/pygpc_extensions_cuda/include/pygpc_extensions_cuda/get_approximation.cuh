#ifndef PYGPC_EXTENSIONS_CUDA_GET_APPROXIMATION_H
#define PYGPC_EXTENSIONS_CUDA_GET_APPROXIMATION_H


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>


template<typename T, typename U>
__global__ void create_gpc_matrix_tk(T* ptr_arguments, T* ptr_poly_coeffs,
    T* ptr_gpc_matrix, U n_arguments, U n_dim, U n_basis)
{
    
    U i_arguments = blockIdx.x * blockDim.x + threadIdx.x;
    if(i_arguments < n_arguments) {
        U i_basis = 0;
        T* local_ptr_poly_coeffs = ptr_poly_coeffs;
        while(i_basis != n_basis) {
            U i_dim = 0;
            T accumulated_result = 1;
            while(i_dim != n_dim) {
                // get argument
                T argument = ptr_arguments[i_arguments * n_dim + i_dim];
                // get order of polynomial
                // then to to first (highest) coefficient 
                U n_order = static_cast<U>
                    (*local_ptr_poly_coeffs++);
                // initialize result variable with highest coefficient
                // then go to next coefficient
                T evaluation_result = *local_ptr_poly_coeffs++;
                // use horners method to evaluate the polynomial
                for(U i_coeff = 0; i_coeff < n_order; ++i_coeff) {
                    evaluation_result = evaluation_result*argument +
                        *local_ptr_poly_coeffs++; 
                }
                // accumulate to overall result
                accumulated_result *= evaluation_result;
                // increment dimension counter
                i_dim++;
            }
            // write result
            ptr_gpc_matrix[i_arguments * n_basis + i_basis] =
                accumulated_result;
            // increment basis counter
            i_basis++;
        }
    }
}

template<typename T, typename U>
U get_approximation_t(T* ptr_arguments, T* ptr_poly_coeffs, T* ptr_gpc_coeffs,
    T* ptr_result, U n_arguments, U n_dim, U n_basis, U n_poly_coeffs,
    U n_gpc_coeffs)
{
    constexpr U n_threads = 512;

    dim3 block_dim(n_threads);
    dim3 grid_dim((n_arguments/n_threads)+1);

    T* dptr_arguments = NULL;
    T* dptr_poly_coeffs = NULL;
    T* dptr_gpc_coeffs = NULL;
    T* dptr_gpc_matrix = NULL;
    T* dptr_result = NULL;

    cudaMalloc((T**) &dptr_arguments, n_arguments*n_dim*sizeof(T));
    cudaMalloc((T**) &dptr_poly_coeffs, n_poly_coeffs*sizeof(T));
    cudaMalloc((T**) &dptr_gpc_coeffs, n_gpc_coeffs*n_basis*sizeof(T));
    cudaMalloc((T**) &dptr_gpc_matrix, n_arguments*n_basis*sizeof(T));
    cudaMalloc((T**) &dptr_result, n_gpc_coeffs*n_arguments*sizeof(T));

    cudaMemcpy(dptr_arguments, ptr_arguments, n_arguments*n_dim*sizeof(T),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dptr_poly_coeffs, ptr_poly_coeffs, n_poly_coeffs*sizeof(T),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dptr_gpc_coeffs, ptr_gpc_coeffs,
        n_gpc_coeffs*n_basis*sizeof(T), cudaMemcpyHostToDevice);
    
    create_gpc_matrix_tk<T,U><<<grid_dim,block_dim>>>(dptr_arguments,
        dptr_poly_coeffs, dptr_gpc_matrix, n_arguments, n_dim, n_basis);

    // NOT COMPATIBLE WITH TEMPLATE FUNCTION

        // matrix multiply with CuBLAS
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);

        const double alpha_value = 1.0;
        const double beta_value = 0.0;
        const double *alpha = &alpha_value;
        const double *beta = &beta_value;

        //multiply polynomial chaos matrix with coefficient matrix
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n_gpc_coeffs,
            n_arguments, n_basis, alpha, dptr_gpc_coeffs, n_gpc_coeffs,
            dptr_gpc_matrix, n_basis, beta, dptr_result, n_gpc_coeffs);

        //destroy cublas handle
        cublasDestroy(cublas_handle);

    // ATTENTION

    cudaMemcpy(ptr_result, dptr_result, n_gpc_coeffs*n_arguments*sizeof(T),
        cudaMemcpyDeviceToHost );

    cudaFree(dptr_arguments);
    cudaFree(dptr_poly_coeffs);
    cudaFree(dptr_gpc_coeffs);
    cudaFree(dptr_gpc_matrix);
    cudaFree(dptr_result);

    return 0;
}


#endif
