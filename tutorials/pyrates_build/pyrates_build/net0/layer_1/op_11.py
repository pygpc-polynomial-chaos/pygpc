import numpy as np
from pyrates.backend.funcs import *
def assign_add_25(I_inh_7,c_130,c_129,r_old_19,r_inh_5,I_inh_old_13,c_128):
    I_inh_7[:] += np.multiply(c_130,np.subtract(np.add(np.multiply(c_129,r_old_19),r_inh_5),np.divide(I_inh_old_13,c_128)))
    return I_inh_7