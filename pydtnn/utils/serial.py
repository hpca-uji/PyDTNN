"""
Utilities for serializing NumPy objects to YAML format.
"""

import numpy as np
import yaml

__all__ = ("NumpyYaml",)


class NumpyYaml(yaml.SafeDumper):
    """
    Custom YAML dumper that supports serialization of NumPy arrays and data types.
    """

    def represent_dtype(self, data: np.ndarray) -> yaml.ScalarNode:
        """
        Represent a NumPy data type as a YAML scalar.
        """
        return self.represent_scalar("!np.type", repr(data))

    def represent_ndarray(self, data: np.ndarray) -> yaml.ScalarNode:
        """
        Represent a NumPy array as a YAML scalar using block style.
        """
        return self.represent_scalar("!np.array", repr(data), style="|")


_numeric = [
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
]

NumpyYaml.add_representer(np.ndarray, NumpyYaml.represent_ndarray)
for dtype in _numeric:
    NumpyYaml.add_representer(dtype, NumpyYaml.represent_dtype)
