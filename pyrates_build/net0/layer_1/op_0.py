import numpy as np
from pyrates.backend.funcs import *
def assign_add(r,c_8,c_6,c_1,r_old,v_old,c_7):
    r[:] += np.multiply(c_8,np.divide(np.add(c_6,np.multiply(np.multiply(c_1,r_old),v_old)),c_7))
    return r