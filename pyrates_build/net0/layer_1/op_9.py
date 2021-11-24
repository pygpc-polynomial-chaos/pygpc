import numpy as np
from pyrates.backend.funcs import *
def assign_add_9(v_1,c_58,v_old_4,c_55,c_56,I_ext_1,I_exc_old_3,I_inh_old_3,c_54,c_51,r_old_6,c_52,c_53,c_57):
    v_1[:] += np.multiply(c_58,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_4,c_55),c_56),I_ext_1),np.multiply(np.subtract(I_exc_old_3,I_inh_old_3),c_54)),np.power(np.multiply(np.multiply(c_51,r_old_6),c_52),c_53)),c_57))
    return v_1