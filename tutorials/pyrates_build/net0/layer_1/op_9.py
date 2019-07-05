import numpy as np
from pyrates.backend.funcs import *
def assign_add_163(v_67,c_927,v_old_103,c_924,c_925,I_ext_45,I_exc_old_102,I_inh_old_102,c_923,c_920,r_old_138,c_921,c_922,c_926):
    v_67[:] += np.multiply(c_927,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_103,c_924),c_925),I_ext_45),np.multiply(np.subtract(I_exc_old_102,I_inh_old_102),c_923)),np.power(np.multiply(np.multiply(c_920,r_old_138),c_921),c_922)),c_926))
    return v_67