

from collections.abc import Callable, Sequence
from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import ArrayShape


def select(name: str) -> Callable[[ArrayShape, ArrayShape], Sequence[Layerable]]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
