import numpy as np
from pyrates.backend.funcs import *
def assign_add_52(I_exc_19,c_297,r_exc_19,I_exc_old_31,c_296):
    I_exc_19[:] += np.multiply(c_297,np.subtract(r_exc_19,np.divide(I_exc_old_31,c_296)))
    return I_exc_19