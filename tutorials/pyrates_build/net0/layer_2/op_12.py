import numpy as np
from pyrates.backend.funcs import *
def assign_12(idx_0,r_exc_2,r_5,idx,c_67):
    r_exc_2[idx_0] = np.multiply(r_5[idx],c_67)
    return r_exc_2