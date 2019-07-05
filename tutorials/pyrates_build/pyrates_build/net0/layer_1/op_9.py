import numpy as np
from pyrates.backend.funcs import *
def assign_add_23(v_7,c_125,v_old_13,c_122,c_123,I_ext_5,I_exc_old_12,I_inh_old_12,c_121,c_118,r_old_18,c_119,c_120,c_124):
    v_7[:] += np.multiply(c_125,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_13,c_122),c_123),I_ext_5),np.multiply(np.subtract(I_exc_old_12,I_inh_old_12),c_121)),np.power(np.multiply(np.multiply(c_118,r_old_18),c_119),c_120)),c_124))
    return v_7