import numpy as np
from pyrates.backend.funcs import *
def assign_68(idx_28,r_exc_22,r_40,idx_27,c_312):
    r_exc_22[idx_28] = np.multiply(r_40[idx_27],c_312)
    return r_exc_22