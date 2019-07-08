import numpy as np
from pyrates.backend.funcs import *
def assign_add_15(v_5,c_95,v_old_9,c_92,c_93,I_ext_3,I_exc_old_8,I_inh_old_8,c_91,c_88,r_old_12,c_89,c_90,c_94):
    v_5[:] += np.multiply(c_95,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_9,c_92),c_93),I_ext_3),np.multiply(np.subtract(I_exc_old_8,I_inh_old_8),c_91)),np.power(np.multiply(np.multiply(c_88,r_old_12),c_89),c_90)),c_94))
    return v_5