import numpy as np
from pyrates.backend.funcs import *
def assign_add_162(r_122,c_919,c_917,c_912,r_old_137,v_old_102,c_918):
    r_122[:] += np.multiply(c_919,np.divide(np.add(c_917,np.multiply(np.multiply(c_912,r_old_137),v_old_102)),c_918))
    return r_122