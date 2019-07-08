import numpy as np
from pyrates.backend.funcs import *
def assign_93(I_inh_old_50,I_inh_32):
    I_inh_old_50[:] = I_inh_32
    return I_inh_old_50