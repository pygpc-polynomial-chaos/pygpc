import numpy as np
from pyrates.backend.funcs import *
def assign_100(I_exc_old_52,I_exc_34):
    I_exc_old_52[:] = I_exc_34
    return I_exc_old_52