import numpy as np
from pyrates.backend.funcs import *
def assign_add_47(v_18,c_274,v_old_29,c_271,c_272,I_ext_12,I_exc_old_28,I_inh_old_28,c_270,c_267,r_old_39,c_268,c_269,c_273):
    v_18[:] += np.multiply(c_274,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_29,c_271),c_272),I_ext_12),np.multiply(np.subtract(I_exc_old_28,I_inh_old_28),c_270)),np.power(np.multiply(np.multiply(c_267,r_old_39),c_268),c_269)),c_273))
    return v_18