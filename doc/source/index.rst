.. pygpc documentation master file, created by
   sphinx-quickstart on Wed Oct  3 12:55:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Documentation of pygpc
======================

A Sensitivity and uncertainty analysis toolbox for Python based on the generalized polynomial chaos method

Basic features:
---------------

- Highly efficient **uncertainty analysis of N-dimensional systems**
- Sensitivity analysis using **Sobol indices** and **Global derivative based sensitivity indices**
- Easy **coupling** to user defined models written in Python, Matlab, etc... 
- The **parallelization** concept allows to run model evaluations in parallel
- Highly efficient **adaptive algorithms** allow for analysis of complex systems
- Includes highly efficient **CPU** and **GPU (CUDA)** implementations to significantly accelerate algorithmic and post-processing routines for high-dimensional and complex problems
- Includes **state-of-the-art techniques** such as:
    - **Projection:** determination of optimal reduced basis
    - **l1-minimization:** reduction of necessary model evaluations by making use of concepts from compressed sensing  
    - **Gradient enhanced gPC:** use of gradient information of the model function to increase accuracy
    - **Multi-element gPC:** analyzing systems with discontinuities and sharp transitions
    
Areas of application:
---------------------

pygpc can be used to analyze a variety of different of problems. It is used for example in the frameworks of:

- Nondestructive testing (`Weise, K., Carlstedt, M., Ziolkowski, M., & Brauer, H. (2015). Uncertainty analysis in Lorentz force eddy current testing. IEEE Transactions on Magnetics, 52(3), 1-4. <https://ieeexplore.ieee.org/abstract/document/7272103>`_)

- Noninvasive brain stimulation (`Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to conductivity uncertainty analysis in electric field calculations. NeuroImage, 188, 821-834. <https://www.sciencedirect.com/science/article/pii/S1053811918322031>`_)

- Transcranial magnetic stimulation (`Weise, K., Di Rienzo, L., Brauer, H., Haueisen, J., & Toepfer, H. (2015). Uncertainty analysis in transcranial magnetic stimulation using nonintrusive polynomial chaos expansion. IEEE Transactions on Magnetics, 51(7), 1-8. <https://ieeexplore.ieee.org/abstract/document/7006714>`_)

- Transcranial direct current stimulation (`Kalloch, B., Weise, K., Bazin, P.-L., Lampea, L., Villringera, A., Hlawitschk, M., & Sehm, B. (2019). The influence of white matter lesions on the electrical fieldduring transcranial electric stimulation - Preliminary results of a computational sensitivity analysis, SfN Annual Meeting 2019, Chicago, Illinois, USA, October 19th-23rd 2019 <https://www.fens.org/News-Activities/Calendar/Meetings/2019/10/SfN-Annual-Meeting-2019/>`_)

If you use pygpc in your studies, please contact `Konstantin Weise <https://www.cbs.mpg.de/person/51222/2470>`_ to extend the list above. 

Installation
------------

**Installation using pip:**

pygpc can be installed via the `pip` command with Python >= 3.6 and then simply run the following line from a terminal:

.. code-block:: bash

   > pip install pygpc

If you want to use the plot functionalities from pygpc, please also install matplotlib:

.. code-block:: bash

   > pip install matplotlib


**Installation using the GitHub repository:**

Alternatively, it is possible to clone this repository and run the setup manually. This requires Cython to compile the C-extensions and Numpy for some headers. You can get Cython and Numpy by running the following command:

.. code-block:: bash

   > pip install cython numpy

Alternatively you can install the build dependencies with the following command:

.. code-block:: bash

   > pip install -r requirements.txt

Afterwards, pygpc can be installed by running the following line from the directory in which the repository was cloned:

.. code-block:: bash

   > python setup.py install

Analyzing MATLAB models with pygpc
----------------------------------

You can easily investigate your models written in MATLAB with pygpc. In order to do so, you have to install the MATLAB Engine API for Python.

**Install MATLAB Engine API for Python**

To start the MATLAB engine within a Python session, you first must install the engine API as a Python package. MATLAB provides a standard Python setup.py file for building and installing the engine using the distutils module. You can use the same setup.py commands to build and install the engine on Windows, Mac, or Linux systems.

Before you install, verify your Python and MATLAB configurations.

- Check that your system has a supported version of Python and MATLAB R2014b or later. To check that Python is installed on your system, run Python at the operating system prompt.
- Add the folder that contains the Python interpreter to your path, if it is not already there.
- Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. Copy the path returned by matlabroot.

To install the engine API, choose one of the following. (You might need administrator privileges to execute these commands.)

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

You can find an example model-file in :code:`.../templates/MyModel_matlab.py` and the associated gPC run-file in :code:`.../templates/MyGPC_matlab.py`.

For additional readings visit the `Calling MATLAB from Python <https://www.mathworks.com/help/matlab/matlab-engine-for-python.html?s_tid=CRUX_lftnav>`_ homepage.

Documentation
-------------

For a full API of pygpc, see https://pygpc.readthedocs.io/en/latest/.
For examplary simulations and model configurations, please have a look at the jupyter notebooks provided in the :code:`/tutorial` folder and the templates in the :code:`/templates` folder.

Reference
---------

If you use pygpc, please cite:

`Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to conductivity uncertainty analysis in electric field calculations. NeuroImage, 188, 821-834. <https://www.sciencedirect.com/science/article/pii/S1053811918322031>`_

Contact
-------

If you have questions, problems or suggestions regarding pygpc, please contact `Konstantin Weise <https://www.cbs.mpg.de/person/51222/2470>`_.


Table of contents
-----------------

.. toctree::

   pygpc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
