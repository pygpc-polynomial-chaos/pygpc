import numpy as np
from pyrates.backend.funcs import *
def assign_2(I_exc_old_5,I_exc_2):
    I_exc_old_5[:] = I_exc_2
    return I_exc_old_5