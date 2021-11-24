import numpy as np
from pyrates.backend.funcs import *
def assign_add_1(v,c_16,v_old_0,c_13,c_14,I_ext,I_exc_old,I_inh_old,c_12,c_9,r_old_0,c_10,c_11,c_15):
    v[:] += np.multiply(c_16,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_0,c_13),c_14),I_ext),np.multiply(np.subtract(I_exc_old,I_inh_old),c_12)),np.power(np.multiply(np.multiply(c_9,r_old_0),c_10),c_11)),c_15))
    return v