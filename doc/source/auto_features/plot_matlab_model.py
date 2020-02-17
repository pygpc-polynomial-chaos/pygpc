"""
Analyzing MATLAB models with pygpc
==================================

You can easily investigate your models written in MATLAB with pygpc. In order to do so, you have to
install the MATLAB Engine API for Python.
"""
import matplotlib.pyplot as plt

_ = plt.figure(figsize=[15, 7])
_ = plt.imshow(plt.imread("../images/python_matlab_interface.png"))
_ = plt.axis('off')

#%%
# **Install MATLAB Engine API for Python**
#
# To start the MATLAB engine within a Python session, you first must install the engine API as a Python package.
# MATLAB provides a standard Python setup.py file for building and installing the engine using the distutils module.
# You can use the same setup.py commands to build and install the engine on Windows, Mac, or Linux systems.
#
# Before you install, verify your Python and MATLAB configurations.
#
# - Check that your system has a supported version of Python and MATLAB R2014b or later.
# To check that Python is installed on your system, run Python at the operating system prompt.
# - Add the folder that contains the Python interpreter to your path, if it is not already there.
# - Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. Copy the path returned by matlabroot.
#
# To install the engine API, choose one of the following. (You might need administrator privileges
# to execute these commands.)
#
# **Windows**
#
# .. code-block:: bash
#
#    > cd "matlabroot\extern\engines\python"
#    > python setup.py install
#
# **macOS or Linux**
#
# .. code-block:: bash
#
#    > cd "matlabroot/extern/engines/python"
#    > python setup.py install
#
# **Withing MATLAB**
#
# .. code-block:: bash
#
#    cd (fullfile(matlabroot,'extern','engines','python'))
#    system('python setup.py install')
#
# After you installed the MATLAB Engine API for Python, you can set
#
# .. code-block:: python
#
#    options["matlab_model"] = True
#
# in your gPC run-file.
#
# You can find an example model-file in :code:`.../templates/MyModel_matlab.py` and the associated gPC
# run-file in :code:`.../templates/MyGPC_matlab.py`.
#
# For additional readings visit the `Calling MATLAB from Python
# <https://www.mathworks.com/help/matlab/matlab-engine-for-python.html?s_tid=CRUX_lftnav>`_ homepage.
#
# Setting up the Matlab model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the following, you see an example .m model file we want to analyze using pygpc:
#
# .. code-block:: matlab
#
#     % Three-dimensional test function of Ishigami.
#
#     function y = Ishigami(x1, x2, x3, a, b)
#
#     y = sin(x1) + a .* sin(x2).^2 + b .* x3.^4 .* sin(x1);
#
# Using the model with pypgc
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# In order to call the Matlab function within a model of pygpc, we have to set up the model as shown below.
# During initialization we pass the function name "fun_path", which tells pygpc where to find the .m function.
# During computation, pygpc creates and passes a matlab_engine instance. Before the model can be called,
# the input parameters from the parameters dictionary p have to be converted to lists the matlab engine can read.
# The example shown below can be found in the templates folder of pygpc (`/templates/MyModel_matlab.py
# <../../../../templates/MyModel_matlab.py>`_)
#
# .. code-block:: python
#
#     import inspect
#     import numpy as np
#     import matlab.engine
#     from pygpc.AbstractModel import AbstractModel
#
#
#     class MyModel_matlab(AbstractModel):
#         '''
#         MyModel evaluates something using Matlab. The parameters of the model (constants and random parameters)
#         are stored in the dictionary p. Their type is defined during the problem definition.
#
#         Parameters
#         ----------
#         p["x1"]: float or ndarray of float [n_grid]
#             Parameter 1
#         p["x2"]: float or ndarray of float [n_grid]
#             Parameter 2
#         p["x3"]: float or ndarray of float [n_grid]
#             Parameter 3
#
#         Returns
#         -------
#         y: ndarray of float [n_grid x n_out]
#             Results of the n_out quantities of interest the gPC is conducted for
#         additional_data: dict or list of dict [n_grid]
#             Additional data, will be saved under its keys in the .hdf5 file during gPC simulations.
#             If multiple grid-points are evaluated in one function call, return a dict for every grid-point in a list
#         '''
#
#         def __init__(self, fun_path):
#             self.fun_path = fun_path
#             self.fname = inspect.getfile(inspect.currentframe())
#
#         def validate(self):
#             pass
#
#         def simulate(self, matlab_engine, process_id=None):
#
#             # add path of Matlab function
#             matlab_engine.addpath(self.fun_path, nargout=0)
#
#             # convert input parameters to matlab format (only lists can be converted)
#             x1 = matlab.double(np.array(self.p["x1"]).tolist())
#             x2 = matlab.double(np.array(self.p["x2"]).tolist())
#             x3 = matlab.double(np.array(self.p["x3"]).tolist())
#             a = matlab.double(np.array(self.p["a"]).tolist())
#             b = matlab.double(np.array(self.p["b"]).tolist())
#
#             # call Matlab function
#             y = matlab_engine.Ishigami(x1, x2, x3, a, b)
#
#             # convert the output back to numpy and ensure that the output is [n_grid x n_out]
#             y = np.array(y).transpose()
#
#             if y.ndim == 0:
#                 y = np.array([[y]])
#             elif y.ndim == 1:
#                 y = y[:, np.newaxis]
#
#             # delete matlab engine after simulations because it can not be saved in the gpc object
#             del self.matlab_engine
#
#             return y