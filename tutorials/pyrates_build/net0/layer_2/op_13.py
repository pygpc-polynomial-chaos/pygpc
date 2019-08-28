import numpy as np
from pyrates.backend.funcs import *
def assign_13(idx_2,r_exc_3,r_6,idx_1,c_71):
    r_exc_3[idx_2] = np.multiply(r_6[idx_1],c_71)
    return r_exc_3