import numpy as np
from pyrates.backend.funcs import *
def assign_add_22(r_12,c_117,c_115,c_110,r_old_17,v_old_12,c_116):
    r_12[:] += np.multiply(c_117,np.divide(np.add(c_115,np.multiply(np.multiply(c_110,r_old_17),v_old_12)),c_116))
    return r_12