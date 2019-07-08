import numpy as np
from pyrates.backend.funcs import *
def assign_add_77(I_inh_30,c_437,r_inh_20,I_inh_old_47,c_436):
    I_inh_30[:] += np.multiply(c_437,np.subtract(r_inh_20,np.divide(I_inh_old_47,c_436)))
    return I_inh_30