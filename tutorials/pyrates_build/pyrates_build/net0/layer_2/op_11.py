import numpy as np
from pyrates.backend.funcs import *
def assign_29(I_inh_old_16,I_inh_10):
    I_inh_old_16[:] = I_inh_10
    return I_inh_old_16