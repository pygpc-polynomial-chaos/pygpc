import numpy as np
from pyrates.backend.funcs import *
def assign_31(idx_10,r_exc_9,r_17,idx_9,c_132):
    r_exc_9[idx_10] = np.multiply(r_17[idx_9],c_132)
    return r_exc_9