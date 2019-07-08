import numpy as np
from pyrates.backend.funcs import *
def assign_add_75(v_30,c_432,v_old_47,c_429,c_430,I_ext_20,I_exc_old_46,I_inh_old_46,c_428,c_425,r_old_63,c_426,c_427,c_431):
    v_30[:] += np.multiply(c_432,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_47,c_429),c_430),I_ext_20),np.multiply(np.subtract(I_exc_old_46,I_inh_old_46),c_428)),np.power(np.multiply(np.multiply(c_425,r_old_63),c_426),c_427)),c_431))
    return v_30