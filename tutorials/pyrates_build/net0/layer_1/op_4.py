import numpy as np
from pyrates.backend.funcs import *
def assign_add_46(r_33,c_266,c_264,c_259,r_old_38,v_old_28,c_265):
    r_33[:] += np.multiply(c_266,np.divide(np.add(c_264,np.multiply(np.multiply(c_259,r_old_38),v_old_28)),c_265))
    return r_33