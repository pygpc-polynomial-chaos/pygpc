import numpy as np
from pyrates.backend.funcs import *
def assign_add_159(v_66,c_906,v_old_101,c_903,c_904,I_ext_44,I_exc_old_100,I_inh_old_100,c_902,c_899,r_old_135,c_900,c_901,c_905):
    v_66[:] += np.multiply(c_906,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_101,c_903),c_904),I_ext_44),np.multiply(np.subtract(I_exc_old_100,I_inh_old_100),c_902)),np.power(np.multiply(np.multiply(c_899,r_old_135),c_900),c_901)),c_905))
    return v_66