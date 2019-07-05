import numpy as np
from pyrates.backend.funcs import *
def assign_add_154(r_120,c_877,c_875,c_870,r_old_131,v_old_98,c_876):
    r_120[:] += np.multiply(c_877,np.divide(np.add(c_875,np.multiply(np.multiply(c_870,r_old_131),v_old_98)),c_876))
    return r_120