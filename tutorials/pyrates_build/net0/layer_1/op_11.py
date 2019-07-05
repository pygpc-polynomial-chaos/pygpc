import numpy as np
from pyrates.backend.funcs import *
def assign_add_165(I_inh_67,c_932,c_931,r_old_139,r_inh_45,I_inh_old_103,c_930):
    I_inh_67[:] += np.multiply(c_932,np.subtract(np.add(np.multiply(c_931,r_old_139),r_inh_45),np.divide(I_inh_old_103,c_930)))
    return I_inh_67