import numpy as np
from pyrates.backend.funcs import *
def assign_add_5(v_0,c_37,v_old_2,c_34,c_35,I_ext_0,I_exc_old_1,I_inh_old_1,c_33,c_30,r_old_3,c_31,c_32,c_36):
    v_0[:] += np.multiply(c_37,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_2,c_34),c_35),I_ext_0),np.multiply(np.subtract(I_exc_old_1,I_inh_old_1),c_33)),np.power(np.multiply(np.multiply(c_30,r_old_3),c_31),c_32)),c_36))
    return v_0