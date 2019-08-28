import numpy as np
from pyrates.backend.funcs import *
def assign_add_3(I_inh,c_21,r_inh,I_inh_old_0,c_20):
    I_inh[:] += np.multiply(c_21,np.subtract(r_inh,np.divide(I_inh_old_0,c_20)))
    return I_inh