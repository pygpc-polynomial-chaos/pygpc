import numpy as np
from pyrates.backend.funcs import *
def assign_66(idx_24,r_exc_20,r_38,idx_23,c_304):
    r_exc_20[idx_24] = np.multiply(r_38[idx_23],c_304)
    return r_exc_20