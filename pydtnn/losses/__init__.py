"""
Loss classes for classification CNNs

If you want to add a new loss:
    1) create a new Python file in this directory,
    2) define your loss class as derived from Loss (or any Loss derived class),
    3) and, optionally, import your layer on this file.
"""

from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy as _BinaryCrossEntropy
from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy as _CategoricalCrossEntropy
from pydtnn.losses.loss import Loss as _Loss
# from pydtnn.utils import get_derived_classes

# # Aliases
# categorical_cross_entropy = CategoricalCrossEntropy
# binary_cross_entropy = BinaryCrossEntropy

# # Search this module for Loss derived classes and expose them
# get_derived_classes(Loss, locals())

# TODO: remove imports and to proper dynamic import
def select(loss_func_name: str) -> type[_Loss]:
    # From snake to camel, if it's necessary
    _loss_func_name = loss_func_name.split("_")
    if len(_loss_func_name) > 1:
        _loss_func_name = "".join(map(lambda x: x.lower().capitalize(), _loss_func_name))
    else:
        _loss_func_name = loss_func_name

    match _loss_func_name:
        case _CategoricalCrossEntropy.__name__:
            return _CategoricalCrossEntropy
        case _BinaryCrossEntropy.__name__:
            return _BinaryCrossEntropy
        case _:
            raise NotImplementedError(f"\'{loss_func_name}\' is not implemented!")
