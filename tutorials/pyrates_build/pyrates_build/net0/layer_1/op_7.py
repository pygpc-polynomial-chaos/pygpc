import numpy as np
from pyrates.backend.funcs import *
def assign_add_21(I_inh_6,c_109,r_inh_4,I_inh_old_11,c_108):
    I_inh_6[:] += np.multiply(c_109,np.subtract(r_inh_4,np.divide(I_inh_old_11,c_108)))
    return I_inh_6