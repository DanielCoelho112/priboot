import torch 
import random 
import numpy as np
import numbers

        
def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.deterministic = True  # Forces deterministic algorithm use in CUDA.
    torch.backends.cudnn.benchmark = False  # If true, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

        
def recursive_format(dictionary, function):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, recursive_format(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, list):
        return type(dictionary)(recursive_format(value, function) for value in dictionary)
    if isinstance(dictionary, numbers.Number):
        return function(dictionary)
    return dictionary

def format_number(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        # return float("%.3g" % float(x))
        return float(f"{float(x):.3f}")

class DotDict(dict):
    """
    Dot notation access to dictionary attributes, recursively.
    """
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    __setattr__ = dict.__setitem__

    def __delattr__(self, attr):
        del self[attr]

    def __missing__(self, key):
        self[key] = DotDict()
        return self[key]