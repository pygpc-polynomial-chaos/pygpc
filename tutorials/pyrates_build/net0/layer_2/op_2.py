import numpy as np
from pyrates.backend.funcs import *
def assign_92(I_exc_old_50,I_exc_32):
    I_exc_old_50[:] = I_exc_32
    return I_exc_old_50