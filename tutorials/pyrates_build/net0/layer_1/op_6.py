import numpy as np
from pyrates.backend.funcs import *
def assign_add_20(I_exc_6,c_119,c_118,r_old_16,r_exc_6,I_exc_old_11,c_117):
    I_exc_6[:] += np.multiply(c_119,np.subtract(np.add(np.multiply(c_118,r_old_16),r_exc_6),np.divide(I_exc_old_11,c_117)))
    return I_exc_6