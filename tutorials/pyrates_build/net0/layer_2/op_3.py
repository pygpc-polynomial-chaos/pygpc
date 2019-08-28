import numpy as np
from pyrates.backend.funcs import *
def assign_3(I_inh_old_5,I_inh_2):
    I_inh_old_5[:] = I_inh_2
    return I_inh_old_5