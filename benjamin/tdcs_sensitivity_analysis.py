#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script implements the PyGPC framework to perform a senstivity analysis
on tDCS simulations including information of white matter lesion tissue.

@author: Benjamin Kalloch
"""
import pygpc

from MySimulationModel import MyModel
from collections import OrderedDict
from pygpc.Problem import Problem
from pygpc.Problem import RandomParameter

class WMLSim:
    def run_adaptive_gpc(self):

        parameters = OrderedDict()
        parameters["scalp_cond"] = RandomParameter("beta", [3,3], [0.28,0.87])
        parameters["skull_cond"] = RandomParameter("beta", [3,3], [0.0016,0.033])
        parameters["gm_cond"] = RandomParameter("beta", [3,3], [0.22,0.67])
        parameters["wm_cond"] = RandomParameter("beta", [3,3], [0.09,0.29])
        parameters["lesion_cond"] = RandomParameter("beta", [3,3], [0.04,1.5])

        eps = 1E-3  # relative error bound


        lesionSensitivity = Problem(MyModel, parameters)

        ########################################################################################
        # run adaptive gpc (regression) passing the goal function func(x, args())
        reg, phi = pygpc.run_reg_adaptive2_parallel(problem=lesionSensitivity,
                                                    order_start=0,
                                                    order_end=16,
                                                    interaction_order_max=2,
                                                    eps=eps,
                                                    print_out=True,
                                                    seed=1,
                                                    save_res_fn='/home/kalloch/OpenFOAM/kalloch-3.0.1/run/PyGPC_LI02828972_WML/pygpc_data/fazekas2', #PyGPC_LI02443371_WML/pygpc_data/fazekas1',#PyGPC_LI02828972_WML/pygpc_data/fazekas2',
                                                    n_cpu=8)

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
        MyModel.write_result_field("ElPot_sobol4", sobol_1st[3])
        MyModel.write_result_field("ElPot_sobol5", sobol_1st[4])
        MyModel.write_result_field("ElPot_sobol_interact_1", sobol_2nd[0])
        MyModel.write_result_field("ElPot_sobol_interact_2", sobol_2nd[1])
        MyModel.write_result_field("ElPot_sobol_interact_3", sobol_2nd[2])

        print sobol_idx_1st
        print sobol_idx_2nd

        print("done!\n")


# ******************* main *********************

sim = WMLSim();
sim.run_adaptive_gpc();
