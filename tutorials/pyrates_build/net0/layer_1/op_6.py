import numpy as np
from pyrates.backend.funcs import *
def assign_add_76(I_exc_30,c_435,c_434,r_old_64,r_exc_30,I_exc_old_47,c_433):
    I_exc_30[:] += np.multiply(c_435,np.subtract(np.add(np.multiply(c_434,r_old_64),r_exc_30),np.divide(I_exc_old_47,c_433)))
    return I_exc_30