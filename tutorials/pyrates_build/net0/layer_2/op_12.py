import numpy as np
from pyrates.backend.funcs import *
def assign_102(idx_40,r_exc_32,r_60,idx_39,c_462):
    r_exc_32[idx_40] = np.multiply(r_60[idx_39],c_462)
    return r_exc_32