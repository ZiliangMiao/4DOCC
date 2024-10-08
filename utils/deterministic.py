import torch
import numpy as np
import random

def set_deterministic(random_seed=666):
    # tf32 core
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.set_float32_matmul_precision('highest')  # highest, float32; high, tensorfloat32

    # deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)