"""
PyDTNN optimizers

If you want to add a new optimizer:
    1) create a new Python file in this directory,
    2) define your optimizer class as derived from Optimizer,
    3) and, optionally, import your optimizer on this file.
"""


from pydtnn.optimizers.adam import Adam
from pydtnn.optimizers.nadam import Nadam
from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.optimizers.rmsprop import RMSProp
from pydtnn.optimizers.sgd import SGD
from pydtnn.utils import get_derived_classes

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn import Model
else:
    Model = object


# Search this module for Optimizer derived classes and expose them
get_derived_classes(Optimizer, locals())

# Aliases
adam = Adam
nadam = Nadam
rmsprop = RMSProp
sgd = SGD


def get_optimizer(model: Model) -> Optimizer:
    """Get optimizer object from model attributes"""
    match model.optimizer_name:

        case "rmsprop":
            opt: RMSProp = RMSProp(learning_rate=model.learning_rate,
                                   rho=model.rho,
                                   epsilon=model.epsilon,
                                   decay=model.decay,
                                   dtype=model.dtype)
        case "adam":
            opt: Adam = Adam(learning_rate=model.learning_rate,
                             beta1=model.beta1,
                             beta2=model.beta2,
                             epsilon=model.epsilon,
                             decay=model.decay,
                             dtype=model.dtype)
        case "nadam":
            opt: Nadam = Nadam(learning_rate=model.learning_rate,
                               beta1=model.beta1,
                               beta2=model.beta2,
                               epsilon=model.epsilon,
                               decay=model.decay,
                               dtype=model.dtype)
        case "sgd":
            opt: SGD = SGD(learning_rate=model.learning_rate,
                           momentum=model.momentum,
                           nesterov=model.nesterov,
                           decay=model.decay,
                           dtype=model.dtype)
        case _:
            raise SystemExit(f"Optimizer '{model.optimizer}' not supported yet!")

    if model.enable_cudnn:
        opt.set_gpudirect(model.gpudirect)
    return opt
