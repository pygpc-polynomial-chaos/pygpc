import numpy as np
from pyrates.backend.funcs import *
def assign_add_155(v_65,c_885,v_old_99,c_882,c_883,I_ext_43,I_exc_old_98,I_inh_old_98,c_881,c_878,r_old_132,c_879,c_880,c_884):
    v_65[:] += np.multiply(c_885,np.divide(np.subtract(np.add(np.add(np.add(np.power(v_old_99,c_882),c_883),I_ext_43),np.multiply(np.subtract(I_exc_old_98,I_inh_old_98),c_881)),np.power(np.multiply(np.multiply(c_878,r_old_132),c_879),c_880)),c_884))
    return v_65