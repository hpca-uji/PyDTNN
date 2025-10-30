from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.optimizers.optimizer import Optimizer as _Optimizer


def select(model: "_Model") -> "_Optimizer":
    """Get optimizer object from model attributes"""
    match model.optimizer_name:

        case "rmsprop":
            from pydtnn.optimizers.rmsprop import RMSProp
            opt = RMSProp.from_model(model)
        case "adam":
            from pydtnn.optimizers.adam import Adam
            opt = Adam.from_model(model)
        case "nadam":
            from pydtnn.optimizers.nadam import Nadam
            opt = Nadam.from_model(model)
        case "sgd":
            from pydtnn.optimizers.sgd import SGD
            opt = SGD.from_model(model)
        case _:
            raise ValueError(f"Optimizer {model.optimizer_name!r} not found!")

    return opt
