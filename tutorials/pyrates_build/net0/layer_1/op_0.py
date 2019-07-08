import numpy as np
from pyrates.backend.funcs import *
def assign_add_70(r_54,c_403,c_401,c_396,r_old_59,v_old_44,c_402):
    r_54[:] += np.multiply(c_403,np.divide(np.add(c_401,np.multiply(np.multiply(c_396,r_old_59),v_old_44)),c_402))
    return r_54