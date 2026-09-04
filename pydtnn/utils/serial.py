"""Utilities for serializing NumPy objects to YAML format."""

from typing import Any

import numpy as np
import yaml

__all__ = ("ReprYaml",)


class ReprYaml(yaml.SafeDumper):
    """Custom YAML dumper that supports serialization of extra data types."""

    def represent_inline(self, data: Any) -> yaml.ScalarNode:
        """Represent a data type as a YAML scalar."""
        return self.represent_scalar("!repr", repr(data))

    def represent_block(self, data: Any) -> yaml.ScalarNode:
        """Represent a data type as a YAML scalar using block style."""
        return self.represent_scalar("!repr", repr(data), style="|")


REPR_TYPES = {
    ReprYaml.represent_inline: [
        np.dtype,
        np.str_,
        np.object_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
    ReprYaml.represent_block: [np.ndarray],
}


for representer, types in REPR_TYPES.items():
    for cls in types:
        ReprYaml.add_representer(cls, representer)
