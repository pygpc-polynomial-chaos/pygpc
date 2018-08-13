#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script implements the PyGPC framework to perform a senstivity analysis
on tDCS simulations including information of white matter lesion tissue.

@author: Benjamin Kalloch
"""
import pygpc

from MySimulationModel import MyModel


class WMLSim:
    def run_adaptive_gpc(self):
        # Statistical parameters
#        """"
        # head case
        random_vars = ['scalp', 'skull', 'gm', 'wm']
        pdftype = ["beta", "beta", "beta", "beta"]
        a = [0.28, 0.0016, 0.22, 0.09]  # lower bounds of conductivities in S/m
        b = [0.87, 0.033, 0.67, 0.29]  # upper bounds of conductivities in S/m
        p = [3, 3, 3, 3]  # first shape parameter of pdf
        q = [3, 3, 3, 3]  # second shape parameter of pdf
 #       """
        """"
        # capacitor case
        random_vars = ['scalp', 'skull', 'gm']
        pdftype = ["beta", "beta", "beta"]
        a = [0.28, 0.0016, 0.22]    # lower bounds of conductivities in S/m
        b = [0.87, 0.033, 0.67]     # upper bounds of conductivities in S/m
        p = [3, 3, 3]               # first shape parameter of pdf
        q = [3, 3, 3]               # second shape parameter of pdf
        """

        eps = 1E-3  # relative error bound
        pdfshape = [p, q]
        limits = [a, b]

        ########################################################################################
        # run adaptive gpc (regression) passing the goal function func(x, args())
        reg, phi = pygpc.run_reg_adaptive2_parallel(random_vars=random_vars,
                                                    pdftype=pdftype,
                                                    pdfshape=pdfshape,
                                                    limits=limits,
                                                    Model=MyModel,
                                                    args=(),
                                                    order_start=0,
                                                    order_end=10,
                                                    interaction_order_max=2,
                                                    eps=eps,
                                                    print_out=True,
                                                    seed=1,
                                                    save_res_fn='/home/kalloch/OpenFOAM/kalloch-3.0.1/run/PyGPC_capacitor/pygpc_data/pygpc_foam',
                                                    n_cpu=4)

        ########################################################################################

        # perform final gpc expansion including all simulations
        coeffs_phi = reg.expand(phi)

        # postprocessing
        mean = reg.mean(coeffs_phi)
        std = reg.std(coeffs_phi)
        sobol, sobol_idx = reg.sobol(coeffs_phi)
        sobol_1st, sobol_idx_1st = pygpc.extract_sobol_order(sobol, sobol_idx, order=1)
        sobol_2nd, sobol_idx_2nd = pygpc.extract_sobol_order(sobol, sobol_idx, order=2)
        globalsens = reg.globalsens(coeffs_phi)

        MyModel.write_result_field("ElPot_mean", mean[0])
        MyModel.write_result_field("ElPot_stddev", std[0])
        MyModel.write_result_field("ElPot_sobol1", sobol_1st[0])
        MyModel.write_result_field("ElPot_sobol2", sobol_1st[1])
        MyModel.write_result_field("ElPot_sobol3", sobol_1st[2])
        print sobol_idx_1st
        print sobol_idx_2nd

        print("done!\n")


# ******************* main *********************

sim = WMLSim();
sim.run_adaptive_gpc();
