import numpy as np
from pyrates.backend.funcs import *
def assign_211(idx_90,r_exc_69,r_127,idx_89,c_940):
    r_exc_69[idx_90] = np.multiply(r_127[idx_89],c_940)
    return r_exc_69