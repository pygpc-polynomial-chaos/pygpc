import numpy as np
from pyrates.backend.funcs import *
def assign_add_78(r_56,c_445,c_443,c_438,r_old_65,v_old_48,c_444):
    r_56[:] += np.multiply(c_445,np.divide(np.add(c_443,np.multiply(np.multiply(c_438,r_old_65),v_old_48)),c_444))
    return r_56