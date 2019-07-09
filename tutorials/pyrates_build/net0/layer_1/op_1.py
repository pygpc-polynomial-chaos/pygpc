import numpy as np
from pyrates.backend.funcs import *
def assign_add_43(v_17,c_253,v_old_27,c_250,c_251,I_ext_11,I_exc_old_26,I_inh_old_26,c_249,c_246,r_old_36,c_247,c_248,c_252):
    v_17[:] += np.multiply(c_253,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_27,c_250),c_251),I_ext_11),np.multiply(np.subtract(I_exc_old_26,I_inh_old_26),c_249)),np.power(np.multiply(np.multiply(c_246,r_old_36),c_247),c_248)),c_252))
    return v_17