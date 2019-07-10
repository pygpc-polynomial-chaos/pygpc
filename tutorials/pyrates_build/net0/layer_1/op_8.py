import numpy as np
from pyrates.backend.funcs import *
def assign_add_8(r_1,c_50,c_48,c_43,r_old_5,v_old_3,c_49):
    r_1[:] += np.multiply(c_50,np.divide(np.add(c_48,np.multiply(np.multiply(c_43,r_old_5),v_old_3)),c_49))
    return r_1