import numpy as np
from pyrates.backend.funcs import *
def assign_add_42(r_32,c_245,c_243,c_238,r_old_35,v_old_26,c_244):
    r_32[:] += np.multiply(c_245,np.divide(np.add(c_243,np.multiply(np.multiply(c_238,r_old_35),v_old_26)),c_244))
    return r_32