import numpy as np
from pyrates.backend.funcs import *
def assign_add_50(r_34,c_287,c_285,c_280,r_old_41,v_old_30,c_286):
    r_34[:] += np.multiply(c_287,np.divide(np.add(c_285,np.multiply(np.multiply(c_280,r_old_41),v_old_30)),c_286))
    return r_34