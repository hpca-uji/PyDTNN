from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.optimizers.optimizer import Optimizer as _Optimizer


def select(model: "_Model") -> "_Optimizer":
    """Get optimizer object from model attributes"""
    from pydtnn.optimizers.rmsprop import RMSProp
    from pydtnn.optimizers.adam import Adam
    from pydtnn.optimizers.nadam import Nadam
    from pydtnn.optimizers.sgd import SGD

    optimizer = {
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
        "sgd": SGD,
    }

    try:
        cls = optimizer[model.optimizer_name]
    except KeyError:
        raise ValueError(f"Optimizer {model.optimizer_name!r} not found!") from None

    return cls.from_model(model)
