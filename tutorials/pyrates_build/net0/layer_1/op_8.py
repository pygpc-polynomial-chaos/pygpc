import numpy as np
from pyrates.backend.funcs import *
def assign_add_22(r_12,c_129,c_127,c_122,r_old_17,v_old_12,c_128):
    r_12[:] += np.multiply(c_129,np.divide(np.add(c_127,np.multiply(np.multiply(c_122,r_old_17),v_old_12)),c_128))
    return r_12