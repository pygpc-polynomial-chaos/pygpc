#include <Python.h>
#include <numpy/arrayobject.h>


#include "pygpc_extensions/create_gpc_matrix.hpp"


extern "C" {

staeic PyObject* create_gpc_matrix_cpu(PyObject* self, PyObject* args)
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

    long int* ptr_dim_arguments = PyArray_DIMS(arguments);
    long int n_arguments = ptr_dim_arguments[0];
    long int n_dim = ptr_dim_arguments[1];
    double* ptr_arguments = (double*)PyArray_DATA(arguments);

    long int* ptr_dim_result = PyArray_DIMS(result);
    long int n_basis= ptr_dim_result[1];
    long int n_grad = ptr_dim_result[2];
    double* ptr_result = (double*)PyArray_DATA(result);

    double* ptr_coeffs = (double*)PyArray_DATA(coeffs);

        create_gpc_matrix_cpu_t<double>(ptr_arguments,
        ptr_coeffs, ptr_result, n_arguments, n_dim, n_basis, n_grad);

    Py_DECREF(arguments);
    Py_DECREF(coeffs);
    Py_DECREF(result);

    return Py_None;
}

/*  define functions in module */
static PyMethodDef methods[] =
{
    {"create_gpc_matrix_cpu", create_gpc_matrix_cpu, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
static struct PyModuleDef pygpc_extensions =
{
    PyModuleDef_HEAD_INIT, 
    "", 
    "", 
    -1, 
    methods
};

PyMODINIT_FUNC
PyInit_pygpc_extensions(void)
{
    import_array();
    return PyModule_Create(&pygpc_extensions);
}

}
