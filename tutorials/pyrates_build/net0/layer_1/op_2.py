import numpy as np
from pyrates.backend.funcs import *
def assign_add_72(I_exc_29,c_414,c_413,r_old_61,r_exc_29,I_exc_old_45,c_412):
    I_exc_29[:] += np.multiply(c_414,np.subtract(np.add(np.multiply(c_413,r_old_61),r_exc_29),np.divide(I_exc_old_45,c_412)))
    return I_exc_29