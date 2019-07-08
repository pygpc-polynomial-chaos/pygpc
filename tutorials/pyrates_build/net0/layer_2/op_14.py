import numpy as np
from pyrates.backend.funcs import *
def assign_104(idx_44,r_exc_34,r_62,idx_43,c_470):
    r_exc_34[idx_44] = np.multiply(r_62[idx_43],c_470)
    return r_exc_34