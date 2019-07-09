import numpy as np
from pyrates.backend.funcs import *
def assign_57(I_inh_old_32,I_inh_20):
    I_inh_old_32[:] = I_inh_20
    return I_inh_old_32