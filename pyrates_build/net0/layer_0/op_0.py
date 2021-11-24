import numpy as np
from pyrates.backend.funcs import *
def assign_17(I_ext_2,var_inp,in_var_idx):
    I_ext_2[:] = var_inp[in_var_idx]
    return I_ext_2