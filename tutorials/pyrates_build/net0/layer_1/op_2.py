import numpy as np
from pyrates.backend.funcs import *
def assign_add_44(I_exc_17,c_256,c_255,r_old_37,r_exc_17,I_exc_old_27,c_254):
    I_exc_17[:] += np.multiply(c_256,np.subtract(np.add(np.multiply(c_255,r_old_37),r_exc_17),np.divide(I_exc_old_27,c_254)))
    return I_exc_17