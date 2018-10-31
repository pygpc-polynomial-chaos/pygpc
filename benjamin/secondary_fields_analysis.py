#!/usr/bin/python
# -*- coding: utf-8 -*-

import pygpc
import os
import sys
import numpy as np

from MySimulationModel import MyModel

# set simulation parameters
random_vars = ['scalp', 'skull', 'gm', 'wm', 'lesion']      # labels of the random variables
pdftype = ["beta", "beta", "beta", "beta", "beta"]          # Type of input PDFs in each dimension
a = [0.28, 0.0016, 0.22, 0.09, 0.04]                        # lower bounds of conductivities in S/m
b = [0.87, 0.033, 0.67, 0.29, 1.5]                          # upper bounds of conductivities in S/m
p = [3, 3, 3, 3, 3]                                         # first shape parameter of pdf
q = [3, 3, 3, 3, 3]                                         # second shape parameter of pdf
pdfshape = [p, q]
limits = [a, b]
order=[4,4,4,4,4]
interaction_order_max=2

num_iterations=122  # the number of iterations that were neccessary to converge below 1e-03 for ElPot gPC

# we do not dynamically expand the random grid, but we directly fix the
# settings & size of the grid that we used at the end of the ElPot gPC expansion
grid_init = pygpc.randomgrid(pdftype, pdfshape, limits, num_iterations, seed=1)

regobj = pygpc.reg(pdftype,
                   pdfshape,
                   limits,
                   order=order,
                   order_max=order[0],
                   interaction_order=interaction_order_max,
                   grid=grid_init,
                   random_vars=random_vars)

# read the data from the simulation results
magE = [] 

for iteration in range( 0, num_iterations ):
    with open("/home/kalloch/OpenFOAM/kalloch-3.0.1/run/PyGPC_LI02828972_WML/results_of_all_iterations/" + repr(iteration)  + "_iter/magE") as f:
        for line in f:
            if "internalField" in line:
                num_lines_to_read = int(next(f))
                next(f)
                break

        magE.append( np.zeros(num_lines_to_read, dtype='float64') )

        print( np.shape( magE) )

        for i in range(0, num_lines_to_read):
            magE[iteration][i] = float(next(f))

# determine gpc coefficients
gpc_coefficients = regobj.expand( np.array( magE ) )

# check the error
regobj.LOOCV( np.array( magE ) )
print("Relative error of leave one out cross validation = {}".format(regobj.relerror_loocv[-1]))

# postprocessing
mean_magE   = regobj.mean( gpc_coefficients )
out_std_reg = regobj.std( gpc_coefficients )
out_sobol_reg, out_sobol_idx_reg = regobj.sobol( gpc_coefficients )
sobol_1st, sobol_idx_1st = pygpc.extract_sobol_order(out_sobol_reg, out_sobol_idx_reg, order=1)
sobol_2nd, sobol_idx_2nd = pygpc.extract_sobol_order(out_sobol_reg, out_sobol_idx_reg, order=2)

MyModel.write_result_field("magE_mean", mean_magE[0])
MyModel.write_result_field("magE_stddev", out_std_reg[0])
MyModel.write_result_field("magE_sobol1", sobol_1st[0])
MyModel.write_result_field("magE_sobol2", sobol_1st[1])
MyModel.write_result_field("magE_sobol3", sobol_1st[2])
MyModel.write_result_field("magE_sobol4", sobol_1st[3])
MyModel.write_result_field("magE_sobol5", sobol_1st[4])
MyModel.write_result_field("magE_sobol_interact_01", sobol_2nd[0])
MyModel.write_result_field("magE_sobol_interact_03", sobol_2nd[1])
MyModel.write_result_field("magE_sobol_interact_02", sobol_2nd[2])
MyModel.write_result_field("magE_sobol_interact_04", sobol_2nd[3])
MyModel.write_result_field("magE_sobol_interact_12", sobol_2nd[4])
MyModel.write_result_field("magE_sobol_interact_13", sobol_2nd[5])
MyModel.write_result_field("magE_sobol_interact_14", sobol_2nd[6])
MyModel.write_result_field("magE_sobol_interact_24", sobol_2nd[7])
MyModel.write_result_field("magE_sobol_interact_23", sobol_2nd[8])
MyModel.write_result_field("magE_sobol_interact_34", sobol_2nd[9])


print("Finish")
