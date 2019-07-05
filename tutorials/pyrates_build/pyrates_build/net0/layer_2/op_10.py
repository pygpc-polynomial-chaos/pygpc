import numpy as np
from pyrates.backend.funcs import *
def assign_28(I_exc_old_16,I_exc_10):
    I_exc_old_16[:] = I_exc_10
    return I_exc_old_16