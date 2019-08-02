import numpy as np
from pyrates.backend.funcs import *
def assign_11(I_inh_old_7,I_inh_4):
    I_inh_old_7[:] = I_inh_4
    return I_inh_old_7