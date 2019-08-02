import numpy as np
from pyrates.backend.funcs import *
def assign_15(idx_6,r_inh_2,r_8,idx_5,c_79):
    r_inh_2[idx_6] = np.multiply(r_8[idx_5],c_79)
    return r_inh_2