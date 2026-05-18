"""
PyDTNN model module providing the base Model class for the framework.
"""

from pydtnn.model.repr import Repr
from pydtnn.model.state import State
from pydtnn.model.train import Train
from pydtnn.utils.constants import Array

__all__ = ("Model",)


class Model[T: Array](Train[T], State[T], Repr[T]):
    """
    # PyDTNN model
    The Model class serves as the primary interface for PyDTNN, integrating
    training, inference, state management, and representation capabilities.

    ## Hierarchy structure diagram:
    ```
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Base ─ Utils ┬ Layers ┬ Init ─ Sync ─ Eval ─ Train ┬ Model ┃
    ┃              │        └── State ───────────────────┤       ┃
    ┃              └── Repr ─────────────────────────────┘       ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    ```
    """
