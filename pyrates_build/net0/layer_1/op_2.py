import numpy as np
from pyrates.backend.funcs import *
def assign_add_2(I_exc,c_19,c_18,r_old_1,r_exc,I_exc_old_0,c_17):
    I_exc[:] += np.multiply(c_19,np.subtract(np.add(np.multiply(c_18,r_old_1),r_exc),np.divide(I_exc_old_0,c_17)))
    return I_exc