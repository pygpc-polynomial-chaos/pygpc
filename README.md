[![](https://img.shields.io/pypi/dm/pygpc.svg)](https://pypi.org/project/pygpc/)
[![](https://img.shields.io/pypi/wheel/pygpc.svg)](https://pypi.org/project/pygpc/)
[![](https://img.shields.io/appveyor/ci/pygpc/pygpc.svg)](https://ci.appveyor.com/project/pygpc/pygpc)
[![](https://img.shields.io/travis/pygpc-polynomial-chaos/pygpc.svg)](https://travis-ci.com/pygpc-polynomial-chaos/pygpc)
[![](https://img.shields.io/readthedocs/pygpc.svg)](https://pygpc.readthedocs.io/en/latest/)
<img src="https://avatars3.githubusercontent.com/u/52486646?s=200&v=4" width="20%" heigth="20%" align="right">

# pygpc
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
- Nondestructive testing ([Weise, K., Carlstedt, M., Ziolkowski, M., & Brauer, H. (2015). Uncertainty analysis in Lorentz force eddy current testing. IEEE Transactions on Magnetics, 52(3), 1-4.](https://ieeexplore.ieee.org/abstract/document/7272103))
- Noninvasive brain stimulation ([Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to conductivity uncertainty analysis in electric field calculations. NeuroImage, 188, 821-834.](https://www.sciencedirect.com/science/article/pii/S1053811918322031))
    - Transcranial magnetic stimulation ([Weise, K., Di Rienzo, L., Brauer, H., Haueisen, J., & Toepfer, H. (2015). Uncertainty analysis in transcranial magnetic stimulation using nonintrusive polynomial chaos expansion. IEEE Transactions on Magnetics, 51(7), 1-8.](https://ieeexplore.ieee.org/abstract/document/7006714))
    - Transcranial direct current stimulation ([Kalloch, B., Weise, K., Bazin, P.-L., Lampea, L., Villringera, A., Hlawitschk, M., & Sehm, B. (2019). The influence of white matter lesions on the electrical fieldduring transcranial electric stimulation - Preliminary results of a computational sensitivity analysis, SfN Annual Meeting 2019, Chicago, Illinois, USA, October 19th-23rd 2019](https://www.fens.org/News-Activities/Calendar/Meetings/2019/10/SfN-Annual-Meeting-2019/))

If you use pygpc in your studies, please contact [Konstantin Weise](https://www.cbs.mpg.de/person/51222/2470) to extend the list above. 

Installation
------------
**Installation using pip:**
pygpc can be installed via the `pip` command with Python >= 3.6 and then simply run the following line from a terminal:
```
pip install pygpc
```

If you want to use the plot functionalities from pygpc, please also install matplotlib:
```
pip install matplotlib
```

**Installation using the GitHub repository:**
Alternatively, it is possible to clone this repository and run the setup manually. This requires Cython to compile the C-extensions and Numpy for some headers. You can get Cython and Numpy by running the following command:
```
pip install cython numpy
```
Alternatively you can install the build dependencies with the following command:
```
pip install -r requirements.txt
```
Afterwards, pygpc can be installed by running the following line from the directory in which the repository was cloned:
```
python setup.py install
```

Documentation
-------------
For a full API of pygpc, see https://pygpc.readthedocs.io/en/latest/.
For examplary simulations and model configurations, please have a look at the jupyter notebooks provided in the /tutorial folder and the templates in the /example folder.

Reference
---------
If you use this framework, please cite:

Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to conductivity uncertainty analysis in electric field calculations. NeuroImage, 188, 821-834.

Contact
-------
If you have questions, problems or suggestions regarding pygpc, please contact [Konstantin Weise](https://www.cbs.mpg.de/person/51222/2470).
