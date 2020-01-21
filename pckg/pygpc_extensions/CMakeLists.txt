cmake_minimum_required(VERSION 3.14)
project(pygpc_extensions)

# set paths 
set(SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TESTS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# disable compiler language extensions
set(CMAKE_CXX_EXTENSIONS OFF)
# set c++-standard
set(CMAKE_CXX_STANDARD 11)

# find Python and NumPy libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy)
# find OpenMP
find_package(OpenMP REQUIRED)

# add include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${INCLUDE_PATH})

# enable -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# place python extension in root path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYGPC_PACKAGE_PATH})

# add python library target
Python3_add_library(pygpc_extensions MODULE ${SOURCE_PATH}/pygpc_extensions.cpp)
