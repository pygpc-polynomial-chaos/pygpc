import numpy as np
from pyrates.backend.funcs import *
def assign_212(idx_92,r_exc_70,r_128,idx_91,c_944):
    r_exc_70[idx_92] = np.multiply(r_128[idx_91],c_944)
    return r_exc_70