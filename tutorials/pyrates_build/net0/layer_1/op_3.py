import numpy as np
from pyrates.backend.funcs import *
def assign_add_73(I_inh_29,c_416,r_inh_19,I_inh_old_45,c_415):
    I_inh_29[:] += np.multiply(c_416,np.subtract(r_inh_19,np.divide(I_inh_old_45,c_415)))
    return I_inh_29