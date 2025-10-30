from typing import TYPE_CHECKING

from pydtnn.utils import find_component

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.optimizers.optimizer import Optimizer as _Optimizer


def select(model: "_Model") -> "_Optimizer":
    """Get optimizer object from model attributes"""
    cls = find_component("optimizers", model.optimizer_name)
    return cls.from_model(model)
