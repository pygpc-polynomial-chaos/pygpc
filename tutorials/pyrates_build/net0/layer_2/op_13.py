import numpy as np
from pyrates.backend.funcs import *
def assign_67(idx_26,r_exc_21,r_39,idx_25,c_308):
    r_exc_21[idx_26] = np.multiply(r_39[idx_25],c_308)
    return r_exc_21