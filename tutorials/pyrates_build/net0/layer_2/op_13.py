import numpy as np
from pyrates.backend.funcs import *
def assign_103(idx_42,r_exc_33,r_61,idx_41,c_466):
    r_exc_33[idx_42] = np.multiply(r_61[idx_41],c_466)
    return r_exc_33