import numpy as np
from pyrates.backend.funcs import *
def assign_32(idx_12,r_exc_10,r_18,idx_11,c_133):
    r_exc_10[idx_12] = np.multiply(r_18[idx_11],c_133)
    return r_exc_10