import numpy as np
from pyrates.backend.funcs import *
def assign_add_24(I_exc_7,c_127,r_exc_7,I_exc_old_13,c_126):
    I_exc_7[:] += np.multiply(c_127,np.subtract(r_exc_7,np.divide(I_exc_old_13,c_126)))
    return I_exc_7