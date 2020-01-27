cmake_minimum_required(VERSION 3.10)
project(pygpc_extensions_cuda LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpreter Development)

find_package(CUDA REQUIRED)

set(SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_ROOT_PATH})

# set(CMAKE_POSITION_INDEPENDENT_CODE ON) 
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --compiler-options -fPIC)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${NumPy_INCLUDE_DIRS})
include_directories(${INCLUDE_PATH})

CUDA_ADD_LIBRARY(create_gpc_matrix_wrapper STATIC src/create_gpc_matrix_wrapper.cu) 

Python3_add_library("pygpc_extensions_cuda.${EXT_PREFIX}" MODULE src/pygpc_extensions_cuda.cpp)
target_link_libraries("pygpc_extensions_cuda.${EXT_PREFIX}" PRIVATE create_gpc_matrix_wrapper)
