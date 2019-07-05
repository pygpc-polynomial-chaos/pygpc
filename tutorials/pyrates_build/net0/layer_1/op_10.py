import numpy as np
from pyrates.backend.funcs import *
def assign_add_164(I_exc_67,c_929,r_exc_67,I_exc_old_103,c_928):
    I_exc_67[:] += np.multiply(c_929,np.subtract(r_exc_67,np.divide(I_exc_old_103,c_928)))
    return I_exc_67