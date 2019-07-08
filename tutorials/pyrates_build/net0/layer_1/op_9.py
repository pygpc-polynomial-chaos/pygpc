import numpy as np
from pyrates.backend.funcs import *
def assign_add_79(v_31,c_453,v_old_49,c_450,c_451,I_ext_21,I_exc_old_48,I_inh_old_48,c_449,c_446,r_old_66,c_447,c_448,c_452):
    v_31[:] += np.multiply(c_453,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_49,c_450),c_451),I_ext_21),np.multiply(np.subtract(I_exc_old_48,I_inh_old_48),c_449)),np.power(np.multiply(np.multiply(c_446,r_old_66),c_447),c_448)),c_452))
    return v_31