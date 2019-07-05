import numpy as np
from pyrates.backend.funcs import *
def assign_add_18(r_11,c_96,c_94,c_89,r_old_14,v_old_10,c_95):
    r_11[:] += np.multiply(c_96,np.divide(np.add(c_94,np.multiply(np.multiply(c_89,r_old_14),v_old_10)),c_95))
    return r_11