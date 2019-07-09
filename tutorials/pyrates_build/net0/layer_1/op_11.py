import numpy as np
from pyrates.backend.funcs import *
def assign_add_53(I_inh_19,c_300,c_299,r_old_43,r_inh_13,I_inh_old_31,c_298):
    I_inh_19[:] += np.multiply(c_300,np.subtract(np.add(np.multiply(c_299,r_old_43),r_inh_13),np.divide(I_inh_old_31,c_298)))
    return I_inh_19