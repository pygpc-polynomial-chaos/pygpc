import numpy as np
from pyrates.backend.funcs import *
def assign_add_11(I_inh_1,c_63,c_62,r_old_7,r_inh_1,I_inh_old_4,c_61):
    I_inh_1[:] += np.multiply(c_63,np.subtract(np.add(np.multiply(c_62,r_old_7),r_inh_1),np.divide(I_inh_old_4,c_61)))
    return I_inh_1