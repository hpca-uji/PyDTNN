from __future__ import annotations
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array

import logging
logger = logging.getLogger(__name__)


class LayerFuse[T: Array]():
    def __init__(self, *args, **kwargs):
        from_parent = kwargs.pop("from_parent", None)
        if from_parent is None:
            super().__init__(*args, **kwargs)
        else:
            self.__dict__.update(from_parent)


def select(name: str) -> type[LayerFuse]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
