import random
import numpy as np
import torch
import os

def set_seed(seed=2025):
    """
    Set global random seed to ensure experiment reproducibility.
    Args:
        seed: Random seed value, defaults to 2025.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass