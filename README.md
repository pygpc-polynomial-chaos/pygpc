# 


[![](https://img.shields.io/github/license/pyrates-neuroscience/PyRates.svg)](https://github.com/pyrates-neuroscience/PyRates) 
[![Build Status](https://travis-ci.com/pyrates-neuroscience/PyRates.svg?branch=master)](https://travis-ci.com/pyrates-neuroscience/PyRates)
<img src="https://avatars3.githubusercontent.com/u/52486646?s=200&v=4" width="20%" heigth="20%" align="right">
[![PyPI version](https://badge.fury.io/py/pyrates.svg)](https://badge.fury.io/py/pyrates)
 
# pygpc
A Sensitivity and uncertainty analysis toolbox for Python based on the generalized polynomial chaos method

Basic features:
---------------
- Highly efficient **uncertainty analysis of N-dimensional systems** with arbitrary number of quantities of interest
- Sensitivity analysis using **Sobol indices** and **Global derivative based sensitivity indices**
- Easy **coupling** to user defined models (also outside of Python) 
- Included **parallelization** concept allows to run model evaluations in parallel
- Highly efficient **adaptive algorithms** allow for analysis of complex systems
- Supports **GPUs (CUDA)** to significantly accelerate algorithmic and post-processing routines for high-dimensional and complex problems
- Includes **state-of-the-art techniques** such as:
    - **Projection:** determination of optimal reduced basis
    - **l1-minimization:** reduction of necessary model evaluations by making use of concepts from compressed sensing  
    - **Gradient enhanced gPC:** use of gradient information of the model function to increase accuracy
    - **Multi-element gPC:** analyzing systems with discontinuities and sharp transitions
    
Installation
------------
pygpc can be installed via the `pip` command with Python >= 3.6 and then simply run the following line from a terminal:
```
pip install pygpc
```
Alternatively, it is possible to clone this repository and run the setup manually. This requires Cython to compile the C-extensions. You can get Cython by running the following command:
```
pip install cython
```
Afterwards, pygpc can be installed by running the following line from the directory in which the repository was cloned:
```
python setup.py install
```

Documentation
-------------
For a full API of pygpc, see https://github.com/pygpc-polynomial-chaos/pygpc/blob/master/doc/build/html/index.html.
For examplary simulations and model configurations, please have a look at the jupyter notebooks provided in the /tutorial folder and the templates in the /example folder.

Reference
---------
If you use this framework, please cite:

Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to conductivity uncertainty analysis in electric field calculations. NeuroImage, 188, 821-834.

Contact
-------
If you have questions, problems or suggestions regarding pygpc, please contact [Konstantin Weise](https://www.cbs.mpg.de/person/51222/2470).
