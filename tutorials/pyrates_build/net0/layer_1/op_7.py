import numpy as np
from pyrates.backend.funcs import *
def assign_add_7(I_inh_0,c_42,r_inh_0,I_inh_old_2,c_41):
    I_inh_0[:] += np.multiply(c_42,np.subtract(r_inh_0,np.divide(I_inh_old_2,c_41)))
    return I_inh_0