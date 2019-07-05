import numpy as np
from pyrates.backend.funcs import *
def assign_25(I_inh_old_15,I_inh_9):
    I_inh_old_15[:] = I_inh_9
    return I_inh_old_15