import numpy as np
from pyrates.backend.funcs import *
def assign_add_157(I_inh_65,c_890,r_inh_43,I_inh_old_99,c_889):
    I_inh_65[:] += np.multiply(c_890,np.subtract(r_inh_43,np.divide(I_inh_old_99,c_889)))
    return I_inh_65