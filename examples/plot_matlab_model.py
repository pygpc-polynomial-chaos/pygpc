"""
Analyzing MATLAB models with pygpc
==================================

You can easily investigate your models written in MATLAB with pygpc. In order to do so, you have to
install the MATLAB Engine API for Python.

**Install MATLAB Engine API for Python**

To start the MATLAB engine within a Python session, you first must install the engine API as a Python package.
MATLAB provides a standard Python setup.py file for building and installing the engine using the distutils module.
You can use the same setup.py commands to build and install the engine on Windows, Mac, or Linux systems.

Before you install, verify your Python and MATLAB configurations.

- Check that your system has a supported version of Python and MATLAB R2014b or later.
To check that Python is installed on your system, run Python at the operating system prompt.
- Add the folder that contains the Python interpreter to your path, if it is not already there.
- Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. Copy the path returned by matlabroot.

To install the engine API, choose one of the following. (You might need administrator privileges
to execute these commands.)

**Windows**

.. code-block:: bash

   > cd "matlabroot\extern\engines\python"
   > python setup.py install

**macOS or Linux**

.. code-block:: bash

   > cd "matlabroot/extern/engines/python"
   > python setup.py install

**Withing MATLAB**

.. code-block:: bash

   cd (fullfile(matlabroot,'extern','engines','python'))
   system('python setup.py install')

After you installed the MATLAB Engine API for Python, you can set

.. code-block:: python

   options["matlab_model"] = True

in your gPC run-file.

You can find an example model-file in :code:`.../templates/MyModel_matlab.py` and the associated gPC
run-file in :code:`.../templates/MyGPC_matlab.py`.

For additional readings visit the `Calling MATLAB from Python
<https://www.mathworks.com/help/matlab/matlab-engine-for-python.html?s_tid=CRUX_lftnav>`_ homepage.
"""