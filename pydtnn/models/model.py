

from collections.abc import Callable, Sequence
from pydtnn.layer import LayerAndActivationBase
from pydtnn.utils import find_component


def select(name: str) -> Callable[[Sequence[int], Sequence[int]], Sequence[LayerAndActivationBase]]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
