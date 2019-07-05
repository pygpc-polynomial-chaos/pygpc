import numpy as np
from pyrates.backend.funcs import *
def assign_213(idx_94,r_inh_46,r_129,idx_93,c_948):
    r_inh_46[idx_94] = np.multiply(r_129[idx_93],c_948)
    return r_inh_46