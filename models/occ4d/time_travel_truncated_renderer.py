import torch
import torch.nn as nn

import time
from tqdm import tqdm
import numpy as np

from functools import partial
from collections import defaultdict

import multiprocessing

# tri-linear (or polynomial) interplation of feature at certain octree level at certain spatial point x
def interpolate(x, level, polynomial_on = True):
    coords = ((2**level)*(x*0.5+0.5))
    d_coords = torch.frac(coords)
    if polynomial_on:
        tx = 3*(d_coords[:,0]**2) - 2*(d_coords[:,0]**3)
        ty = 3*(d_coords[:,1]**2) - 2*(d_coords[:,1]**3)
        tz = 3*(d_coords[:,2]**2) - 2*(d_coords[:,2]**3)
    else:  # linear
        tx = d_coords[:,0]
        ty = d_coords[:,1]
        tz = d_coords[:,2]
    _1_tx = 1-tx
    _1_ty = 1-ty
    _1_tz = 1-tz
    p0 = _1_tx*_1_ty*_1_tz
    p1 = _1_tx*_1_ty*tz
    p2 = _1_tx*ty*_1_tz
    p3 = _1_tx*ty*tz
    p4 = tx*_1_ty*_1_tz
    p5 = tx*_1_ty*tz
    p6 = tx*ty*_1_tz
    p7 = tx*ty*tz

    p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.unsqueeze(2)
    return p