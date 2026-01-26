

from collections.abc import Callable, Sequence
from pydtnn.layer_base import LayerBase
from pydtnn.utils import find_component
from pydtnn.utils.constants import ArrayShape

def select(name: str) -> Callable[[ArrayShape, ArrayShape], Sequence[LayerBase]]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
