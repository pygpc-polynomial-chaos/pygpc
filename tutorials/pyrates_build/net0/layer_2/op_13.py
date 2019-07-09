import numpy as np
from pyrates.backend.funcs import *
def assign_31(idx_10,r_exc_9,r_17,idx_9,c_150):
    r_exc_9[idx_10] = np.multiply(r_17[idx_9],c_150)
    return r_exc_9