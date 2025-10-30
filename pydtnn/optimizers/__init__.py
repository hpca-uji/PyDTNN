from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.optimizers.optimizer import Optimizer as _Optimizer


def select(model: "_Model") -> "_Optimizer":
    """Get optimizer object from model attributes"""
    match model.optimizer_name:

        case "rmsprop":
            from pydtnn.optimizers.rmsprop import RMSProp
            opt = RMSProp(learning_rate=model.learning_rate,
                          rho=model.rho,
                          epsilon=model.epsilon,
                          decay=model.decay,
                          dtype=model.dtype)
        case "adam":
            from pydtnn.optimizers.adam import Adam
            opt = Adam(learning_rate=model.learning_rate,
                       beta1=model.beta1,
                       beta2=model.beta2,
                       epsilon=model.epsilon,
                       decay=model.decay,
                       dtype=model.dtype)
        case "nadam":
            from pydtnn.optimizers.nadam import Nadam
            opt = Nadam(learning_rate=model.learning_rate,
                        beta1=model.beta1,
                        beta2=model.beta2,
                        epsilon=model.epsilon,
                        decay=model.decay,
                        dtype=model.dtype)
        case "sgd":
            from pydtnn.optimizers.sgd import SGD
            opt = SGD(learning_rate=model.learning_rate,
                      momentum=model.momentum,
                      nesterov=model.nesterov,
                      decay=model.decay,
                      dtype=model.dtype)
        case _:
            raise ValueError(f"Optimizer {model.optimizer_name!r} not found!")

    return opt
