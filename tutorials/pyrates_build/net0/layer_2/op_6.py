import numpy as np
from pyrates.backend.funcs import *
def assign_60(I_exc_old_33,I_exc_21):
    I_exc_old_33[:] = I_exc_21
    return I_exc_old_33