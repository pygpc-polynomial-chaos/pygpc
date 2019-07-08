import numpy as np
from pyrates.backend.funcs import *
def assign_add_20(I_exc_6,c_107,c_106,r_old_16,r_exc_6,I_exc_old_11,c_105):
    I_exc_6[:] += np.multiply(c_107,np.subtract(np.add(np.multiply(c_106,r_old_16),r_exc_6),np.divide(I_exc_old_11,c_105)))
    return I_exc_6