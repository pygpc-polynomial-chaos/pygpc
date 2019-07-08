import numpy as np
from pyrates.backend.funcs import *
def assign_105(idx_46,r_inh_22,r_63,idx_45,c_474):
    r_inh_22[idx_46] = np.multiply(r_63[idx_45],c_474)
    return r_inh_22