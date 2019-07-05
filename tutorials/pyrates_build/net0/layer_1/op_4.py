import numpy as np
from pyrates.backend.funcs import *
def assign_add_158(r_121,c_898,c_896,c_891,r_old_134,v_old_100,c_897):
    r_121[:] += np.multiply(c_898,np.divide(np.add(c_896,np.multiply(np.multiply(c_891,r_old_134),v_old_100)),c_897))
    return r_121