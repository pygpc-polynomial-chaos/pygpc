import numpy as np
from pyrates.backend.funcs import *
def assign_add_10(I_exc_1,c_60,r_exc_1,I_exc_old_4,c_59):
    I_exc_1[:] += np.multiply(c_60,np.subtract(r_exc_1,np.divide(I_exc_old_4,c_59)))
    return I_exc_1