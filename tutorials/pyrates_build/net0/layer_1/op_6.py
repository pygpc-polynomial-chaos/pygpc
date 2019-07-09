import numpy as np
from pyrates.backend.funcs import *
def assign_add_48(I_exc_18,c_277,c_276,r_old_40,r_exc_18,I_exc_old_29,c_275):
    I_exc_18[:] += np.multiply(c_277,np.subtract(np.add(np.multiply(c_276,r_old_40),r_exc_18),np.divide(I_exc_old_29,c_275)))
    return I_exc_18