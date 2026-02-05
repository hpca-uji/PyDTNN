from pydtnn.layers.layer import Layer
from pydtnn.utils import find_component


def select(name: str) -> type[Layer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
