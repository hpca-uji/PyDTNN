from pydtnn.model.repr import Repr
from pydtnn.model.state import State
from pydtnn.model.train import Train
from pydtnn.utils.constants import Array

__all__ = ("Model",)


class Model[T: Array](Train[T], State[T], Repr[T]):
    """
    # PyDTNN model

    # Hierarchy structure diagram:
    ```
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Base ─ Utils ┬ Layers ┬ Init ─ Sync ─ Eval ─ Train ┬ Model ┃
    ┃              │        └── State ───────────────────┤       ┃
    ┃              └── Repr ─────────────────────────────┘       ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    ```
    """
