import numpy as np
from pyrates.backend.funcs import *
def assign_add_19(v_6,c_116,v_old_11,c_113,c_114,I_ext_4,I_exc_old_10,I_inh_old_10,c_112,c_109,r_old_15,c_110,c_111,c_115):
    v_6[:] += np.multiply(c_116,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_11,c_113),c_114),I_ext_4),np.multiply(np.subtract(I_exc_old_10,I_inh_old_10),c_112)),np.power(np.multiply(np.multiply(c_109,r_old_15),c_110),c_111)),c_115))
    return v_6