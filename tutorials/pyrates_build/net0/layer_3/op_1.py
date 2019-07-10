import numpy as np
from pyrates.backend.funcs import *
def assign_add_12(out_var_idx_0,upd):
    out_var_idx_0[:] += upd
    return out_var_idx_0