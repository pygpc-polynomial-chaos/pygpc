import numpy as np
from pyrates.backend.funcs import *
def assign_14(idx_4,r_exc_4,r_7,idx_3,c_75):
    r_exc_4[idx_4] = np.multiply(r_7[idx_3],c_75)
    return r_exc_4