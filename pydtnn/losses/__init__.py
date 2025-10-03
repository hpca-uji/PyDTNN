"""
Loss classes for classification CNNs

If you want to add a new loss:
    1) create a new Python file in this directory,
    2) define your loss class as derived from Loss (or any Loss derived class),
    3) and, optionally, import your layer on this file.
"""

from .binary_cross_entropy import BinaryCrossEntropy
from .categorical_cross_entropy import CategoricalCrossEntropy
from .loss import Loss
from ..utils import get_derived_classes

# Aliases
categorical_cross_entropy = CategoricalCrossEntropy
binary_cross_entropy = BinaryCrossEntropy

# Search this module for Loss derived classes and expose them
get_derived_classes(Loss, locals())

def switch_losses(loss_func_name: str) -> Loss:
    # From snake to camel, if it's necessary
    _loss_func_name = loss_func_name.split("_")
    if len(_loss_func_name) > 1:
        _loss_func_name = "".join(map(lambda x: x.lower().capitalize(), _loss_func_name))
    else:
        _loss_func_name = loss_func_name

    match _loss_func_name:
        case CategoricalCrossEntropy.__name__:
            return categorical_cross_entropy
        case BinaryCrossEntropy.__name__:
            return binary_cross_entropy
        case _:
            raise NotImplementedError(f"\'{loss_func_name}\' is not implemented!")
