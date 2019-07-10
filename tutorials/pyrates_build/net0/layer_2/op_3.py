import numpy as np
from pyrates.backend.funcs import *
def assign_21(I_inh_old_14,I_inh_8):
    I_inh_old_14[:] = I_inh_8
    return I_inh_old_14