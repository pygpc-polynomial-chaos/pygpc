import numpy as np
from pyrates.backend.funcs import *
def assign_add_17(I_inh_5,c_88,r_inh_3,I_inh_old_9,c_87):
    I_inh_5[:] += np.multiply(c_88,np.subtract(r_inh_3,np.divide(I_inh_old_9,c_87)))
    return I_inh_5