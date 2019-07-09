import numpy as np
from pyrates.backend.funcs import *
def assign_add_51(v_19,c_295,v_old_31,c_292,c_293,I_ext_13,I_exc_old_30,I_inh_old_30,c_291,c_288,r_old_42,c_289,c_290,c_294):
    v_19[:] += np.multiply(c_295,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_31,c_292),c_293),I_ext_13),np.multiply(np.subtract(I_exc_old_30,I_inh_old_30),c_291)),np.power(np.multiply(np.multiply(c_288,r_old_42),c_289),c_290)),c_294))
    return v_19