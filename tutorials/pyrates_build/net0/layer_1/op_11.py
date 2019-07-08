import numpy as np
from pyrates.backend.funcs import *
def assign_add_81(I_inh_31,c_458,c_457,r_old_67,r_inh_21,I_inh_old_49,c_456):
    I_inh_31[:] += np.multiply(c_458,np.subtract(np.add(np.multiply(c_457,r_old_67),r_inh_21),np.divide(I_inh_old_49,c_456)))
    return I_inh_31