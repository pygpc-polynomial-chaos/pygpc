import numpy as np
from pyrates.backend.funcs import *
def assign_add_160(I_exc_66,c_909,c_908,r_old_136,r_exc_66,I_exc_old_101,c_907):
    I_exc_66[:] += np.multiply(c_909,np.subtract(np.add(np.multiply(c_908,r_old_136),r_exc_66),np.divide(I_exc_old_101,c_907)))
    return I_exc_66