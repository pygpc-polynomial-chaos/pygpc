import numpy as np
from pyrates.backend.funcs import *
def assign_add_74(r_55,c_424,c_422,c_417,r_old_62,v_old_46,c_423):
    r_55[:] += np.multiply(c_424,np.divide(np.add(c_422,np.multiply(np.multiply(c_417,r_old_62),v_old_46)),c_423))
    return r_55