import numpy as np
from pyrates.backend.funcs import *
def assign_add_15(v_5,c_83,v_old_9,c_80,c_81,I_ext_3,I_exc_old_8,I_inh_old_8,c_79,c_76,r_old_12,c_77,c_78,c_82):
    v_5[:] += np.multiply(c_83,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_9,c_80),c_81),I_ext_3),np.multiply(np.subtract(I_exc_old_8,I_inh_old_8),c_79)),np.power(np.multiply(np.multiply(c_76,r_old_12),c_77),c_78)),c_82))
    return v_5