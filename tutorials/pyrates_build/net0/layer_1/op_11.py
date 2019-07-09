import numpy as np
from pyrates.backend.funcs import *
def assign_add_25(I_inh_7,c_142,c_141,r_old_19,r_inh_5,I_inh_old_13,c_140):
    I_inh_7[:] += np.multiply(c_142,np.subtract(np.add(np.multiply(c_141,r_old_19),r_inh_5),np.divide(I_inh_old_13,c_140)))
    return I_inh_7