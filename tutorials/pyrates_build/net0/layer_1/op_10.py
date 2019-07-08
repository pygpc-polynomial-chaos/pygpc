import numpy as np
from pyrates.backend.funcs import *
def assign_add_80(I_exc_31,c_455,r_exc_31,I_exc_old_49,c_454):
    I_exc_31[:] += np.multiply(c_455,np.subtract(r_exc_31,np.divide(I_exc_old_49,c_454)))
    return I_exc_31