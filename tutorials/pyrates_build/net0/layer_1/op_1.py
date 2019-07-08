import numpy as np
from pyrates.backend.funcs import *
def assign_add_71(v_29,c_411,v_old_45,c_408,c_409,I_ext_19,I_exc_old_44,I_inh_old_44,c_407,c_404,r_old_60,c_405,c_406,c_410):
    v_29[:] += np.multiply(c_411,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_45,c_408),c_409),I_ext_19),np.multiply(np.subtract(I_exc_old_44,I_inh_old_44),c_407)),np.power(np.multiply(np.multiply(c_404,r_old_60),c_405),c_406)),c_410))
    return v_29