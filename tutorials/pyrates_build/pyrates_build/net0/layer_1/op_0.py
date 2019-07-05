import numpy as np
from pyrates.backend.funcs import *
def assign_add_14(r_10,c_75,c_73,c_68,r_old_11,v_old_8,c_74):
    r_10[:] += np.multiply(c_75,np.divide(np.add(c_73,np.multiply(np.multiply(c_68,r_old_11),v_old_8)),c_74))
    return r_10