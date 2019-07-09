import numpy as np
from pyrates.backend.funcs import *
def assign_69(idx_30,r_inh_14,r_41,idx_29,c_316):
    r_inh_14[idx_30] = np.multiply(r_41[idx_29],c_316)
    return r_inh_14