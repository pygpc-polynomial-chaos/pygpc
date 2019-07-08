import numpy as np
from pyrates.backend.funcs import *
def assign_add_23(v_7,c_137,v_old_13,c_134,c_135,I_ext_5,I_exc_old_12,I_inh_old_12,c_133,c_130,r_old_18,c_131,c_132,c_136):
    v_7[:] += np.multiply(c_137,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_13,c_134),c_135),I_ext_5),np.multiply(np.subtract(I_exc_old_12,I_inh_old_12),c_133)),np.power(np.multiply(np.multiply(c_130,r_old_18),c_131),c_132)),c_136))
    return v_7