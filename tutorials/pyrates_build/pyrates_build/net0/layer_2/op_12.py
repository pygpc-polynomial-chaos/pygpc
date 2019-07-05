import numpy as np
from pyrates.backend.funcs import *
def assign_30(idx_8,r_exc_8,r_16,idx_7,c_131):
    r_exc_8[idx_8] = np.multiply(r_16[idx_7],c_131)
    return r_exc_8