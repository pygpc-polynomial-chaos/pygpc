import numpy as np
from pyrates.backend.funcs import *
def assign_add_45(I_inh_17,c_258,r_inh_11,I_inh_old_27,c_257):
    I_inh_17[:] += np.multiply(c_258,np.subtract(r_inh_11,np.divide(I_inh_old_27,c_257)))
    return I_inh_17