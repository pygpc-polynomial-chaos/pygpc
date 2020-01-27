#include <Python.h>
#include <numpy/arrayobject.h>


#include "pygpc_extensions_cuda/create_gpc_matrix_wrapper.cuh"


extern "C" {

static PyObject* create_gpc_matrix_cuda(PyObject* self, PyObject* args)
{
    PyObject* py_arguments = NULL;
    PyObject* py_result = NULL;
    PyObject* py_coeffs = NULL;
    PyObject* arguments = NULL;
    PyObject* result = NULL;
    PyObject* coeffs = NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &py_arguments,
        &PyArray_Type, &py_coeffs, &PyArray_Type, &py_result))
        return NULL;
    
    arguments = PyArray_FROM_OTF(py_arguments, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    coeffs = PyArray_FROM_OTF(py_coeffs, NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY);
    result = PyArray_FROM_OTF(py_result, NPY_DOUBLE,
        NPY_ARRAY_OUT_ARRAY);

    npy_intp* ptr_dim_arguments = PyArray_DIMS(arguments);
    npy_intp n_arguments = ptr_dim_arguments[0];
    npy_intp n_dim = ptr_dim_arguments[1];
    double* ptr_arguments = (double*)PyArray_DATA(arguments);

    npy_intp* ptr_dim_result = PyArray_DIMS(result);
    npy_intp n_basis = ptr_dim_result[1];
    npy_intp n_grad = ptr_dim_result[2];
    double* ptr_result = (double*)PyArray_DATA(result);

    npy_intp* ptr_dim_coeffs = PyArray_DIMS(coeffs);
    npy_intp n_coeffs = ptr_dim_coeffs[0];
    double* ptr_coeffs = (double*)PyArray_DATA(coeffs);

    create_gpc_matrix_cuda_wrapper(ptr_arguments,
        ptr_coeffs, ptr_result, n_arguments, n_dim, n_basis, n_grad, n_coeffs);

    Py_DECREF(arguments);
    Py_DECREF(coeffs);
    Py_DECREF(result);

    return Py_None;
}

static PyMethodDef methods[] =
{
    {"create_gpc_matrix_cuda", create_gpc_matrix_cuda, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pygpc_extensions_cuda =
{
    PyModuleDef_HEAD_INIT, 
    "", 
    "", 
    -1, 
    methods
};

PyMODINIT_FUNC
PyInit_pygpc_extensions_cuda(void)
{
    import_array();
    return PyModule_Create(&pygpc_extensions_cuda);
}

}
