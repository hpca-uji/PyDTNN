from pydtnn.optimizers.adam import Adam as _Adam
from pydtnn.optimizers.nadam import Nadam as _Nadam
from pydtnn.optimizers.optimizer import Optimizer as _Optimizer
from pydtnn.optimizers.rmsprop import RMSProp as _RMSProp
from pydtnn.optimizers.sgd import SGD as _SGD

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model


def get_optimizer(model: "Model") -> _Optimizer:
    """Get optimizer object from model attributes"""
    match model.optimizer_name:

        case "rmsprop":
            opt = _RMSProp(learning_rate=model.learning_rate,
                                   rho=model.rho,
                                   epsilon=model.epsilon,
                                   decay=model.decay,
                                   dtype=model.dtype)
        case "adam":
            opt = _Adam(learning_rate=model.learning_rate,
                             beta1=model.beta1,
                             beta2=model.beta2,
                             epsilon=model.epsilon,
                             decay=model.decay,
                             dtype=model.dtype)
        case "nadam":
            opt = _Nadam(learning_rate=model.learning_rate,
                               beta1=model.beta1,
                               beta2=model.beta2,
                               epsilon=model.epsilon,
                               decay=model.decay,
                               dtype=model.dtype)
        case "sgd":
            opt = _SGD(learning_rate=model.learning_rate,
                           momentum=model.momentum,
                           nesterov=model.nesterov,
                           decay=model.decay,
                           dtype=model.dtype)
        case _:
            raise SystemExit(f"Optimizer '{model.optimizer}' not supported yet!")
    return opt
