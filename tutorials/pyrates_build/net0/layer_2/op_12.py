import numpy as np
from pyrates.backend.funcs import *
def assign_210(idx_88,r_exc_68,r_126,idx_87,c_936):
    r_exc_68[idx_88] = np.multiply(r_126[idx_87],c_936)
    return r_exc_68