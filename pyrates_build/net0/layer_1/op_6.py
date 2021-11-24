import numpy as np
from pyrates.backend.funcs import *
def assign_add_6(I_exc_0,c_40,c_39,r_old_4,r_exc_0,I_exc_old_2,c_38):
    I_exc_0[:] += np.multiply(c_40,np.subtract(np.add(np.multiply(c_39,r_old_4),r_exc_0),np.divide(I_exc_old_2,c_38)))
    return I_exc_0