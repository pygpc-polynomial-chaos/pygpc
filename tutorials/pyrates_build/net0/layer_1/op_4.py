import numpy as np
from pyrates.backend.funcs import *
def assign_add_4(r_0,c_29,c_27,c_22,r_old_2,v_old_1,c_28):
    r_0[:] += np.multiply(c_29,np.divide(np.add(c_27,np.multiply(np.multiply(c_22,r_old_2),v_old_1)),c_28))
    return r_0