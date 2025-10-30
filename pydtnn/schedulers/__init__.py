from typing import TYPE_CHECKING

from pydtnn.utils import find_component

if TYPE_CHECKING:
    from pydtnn.model import Model as _Model
    from pydtnn.schedulers.scheduler import Scheduler as _LRScheduler


def select(model: "_Model") -> "list[_LRScheduler]":
    """Get Scheduler objects from model attributes"""

    schedulers = []
    for scheduler_name in filter(None, model.schedulers_names.split(",")):
        cls = find_component("schedulers", scheduler_name)
        schedulers.append(cls.from_model(model))

    return schedulers
