import numpy as np
from pyrates.backend.funcs import *
def assign_add_16(I_exc_5,c_98,c_97,r_old_13,r_exc_5,I_exc_old_9,c_96):
    I_exc_5[:] += np.multiply(c_98,np.subtract(np.add(np.multiply(c_97,r_old_13),r_exc_5),np.divide(I_exc_old_9,c_96)))
    return I_exc_5