import numpy as np
from pyrates.backend.funcs import *
def assign_add_19(v_6,c_104,v_old_11,c_101,c_102,I_ext_4,I_exc_old_10,I_inh_old_10,c_100,c_97,r_old_15,c_98,c_99,c_103):
    v_6[:] += np.multiply(c_104,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_11,c_101),c_102),I_ext_4),np.multiply(np.subtract(I_exc_old_10,I_inh_old_10),c_100)),np.power(np.multiply(np.multiply(c_97,r_old_15),c_98),c_99)),c_103))
    return v_6