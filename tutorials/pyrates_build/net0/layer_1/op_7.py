import numpy as np
from pyrates.backend.funcs import *
def assign_add_49(I_inh_18,c_279,r_inh_12,I_inh_old_29,c_278):
    I_inh_18[:] += np.multiply(c_279,np.subtract(r_inh_12,np.divide(I_inh_old_29,c_278)))
    return I_inh_18