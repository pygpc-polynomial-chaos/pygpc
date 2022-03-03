"""
Comparison of sampling schemes
==============================

For texting the efficiency of the sampling schemes in pygpc for a sparse recovery specifically we
compared L1 optimal sampling, LHS sampling and random sampling in a larger analysis. The analysis was
performed in the context of L1 minimization using the least-angle regression algorithm to fit the gPC
regression models (using the LARS-solver from scipy). The sampling schemes were investigated by evaluating the quality of
the constructed surrogate models considering distinct test cases representing different problem classes
covering low, medium and high dimensional problems as well as on
an application example to estimate the sensitivity of the self-impedance of a probe, which is used to
measure the impedance of biological tissues at different frequencies. Due to the random nature, we
compared the sampling schemes using statistical stability measures and evaluated the success rates
to construct a surrogate model with an accuracy of < 0.1\\%. We observed strong differences in the
convergence properties of the methods between the analyzed test functions.

The four test cases were investigated with different pygpc set-ups, the most notable are listed here:

 .. image:: ../../../examples/images/Testfunctions_table.png
     :width: 700
     :align: center

Benchmark on the Ishigami function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the testcases for different grids was the Ishigami function, which can be easily called in pygpc.

"""

import numpy as np
from pygpc.testfunctions import plot_testfunction as plot
from collections import OrderedDict

parameters = OrderedDict()
parameters["x1"] = np.linspace(-np.pi, np.pi, 100)
parameters["x2"] = np.linspace(-np.pi, np.pi, 100)

constants = OrderedDict()
constants["a"] = 7.
constants["b"] = 0.1
constants["x3"] = 0.

plot("Ishigami", parameters, constants, plot_3d=False)

# %%
# Through many repetitons we recorded the normalized root mean squared deviation (NRMSD) as error measure.
#
# .. image:: ../../../examples/images/Ishigami_nrmsd.png
#     :width: 1400
#     :align: center
#
# In this figure the sampling designs are abbreviated as follows: STD - standard LHS sampling, MM - maximin LHS sampling,
# SC-ESE - ESE algorithm for LHS sampling, MC - mutual coherence optimal L1 sampling, MC-CC mutual coherence and average
# cross correlation optimal L1 sampling, CO - coherence optimal L1 sampling, D - :math:`D` optimal sampling and D-COH -
# :math:`D` and coherence optimal sampling.
#
# Over four different test cases a grand average can be seen in the following table whereby the Ishigami and Rosenbrock
# function serve as well known test cases and the LPP (Linear-Paired-Product) function is a very high dimensional (30 to
# be precise) test function with little complexity. The Electrode impedance model is a practical example as mentioned
# above, that was picked as a point of reference. In the following table the relative and average number of grid points
# :math:`\hat{N}_{\varepsilon}` of different sampling schemes to reach an NRMSD of 10−3 with respect to standard
# random sampling. The columns for :math:`N_{sr}^{(95\%)}` and :math:`N_{sr}^{(99\%)}` show the number
# of samples needed to reach a success rate of 95% and 99% respectively.
#
# .. image:: ../../../examples/images/Average_table_pygpc.png
#     :width: 900
#     :align: center
#

