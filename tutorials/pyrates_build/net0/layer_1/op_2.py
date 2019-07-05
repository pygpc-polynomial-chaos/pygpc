import numpy as np
from pyrates.backend.funcs import *
def assign_add_156(I_exc_65,c_888,c_887,r_old_133,r_exc_65,I_exc_old_99,c_886):
    I_exc_65[:] += np.multiply(c_888,np.subtract(np.add(np.multiply(c_887,r_old_133),r_exc_65),np.divide(I_exc_old_99,c_886)))
    return I_exc_65