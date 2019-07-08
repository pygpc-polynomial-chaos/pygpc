import numpy as np
from pyrates.backend.funcs import *
def assign_add_18(r_11,c_108,c_106,c_101,r_old_14,v_old_10,c_107):
    r_11[:] += np.multiply(c_108,np.divide(np.add(c_106,np.multiply(np.multiply(c_101,r_old_14),v_old_10)),c_107))
    return r_11