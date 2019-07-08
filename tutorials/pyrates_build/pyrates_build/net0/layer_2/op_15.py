import numpy as np
from pyrates.backend.funcs import *
def assign_33(idx_14,r_inh_6,r_19,idx_13,c_134):
    r_inh_6[idx_14] = np.multiply(r_19[idx_13],c_134)
    return r_inh_6