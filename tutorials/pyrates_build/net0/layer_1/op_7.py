import numpy as np
from pyrates.backend.funcs import *
def assign_add_161(I_inh_66,c_911,r_inh_44,I_inh_old_101,c_910):
    I_inh_66[:] += np.multiply(c_911,np.subtract(r_inh_44,np.divide(I_inh_old_101,c_910)))
    return I_inh_66