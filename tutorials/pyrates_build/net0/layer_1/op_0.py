import numpy as np
from pyrates.backend.funcs import *
def assign_add_14(r_10,c_87,c_85,c_80,r_old_11,v_old_8,c_86):
    r_10[:] += np.multiply(c_87,np.divide(np.add(c_85,np.multiply(np.multiply(c_80,r_old_11),v_old_8)),c_86))
    return r_10