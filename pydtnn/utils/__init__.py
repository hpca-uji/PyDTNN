"""Utility functions and classes for the PyDTNN framework."""

import sys
import math
import ctypes
import string
import zipfile
import logging
import threading
from collections.abc import Iterable
from importlib import import_module, resources
from pathlib import PurePath
from queue import Queue
from typing import Any, Literal, Self

import numpy as np

from pydtnn import package_name

__all__ = (
    "BackgroundGenerator",
    "convert_size",
    "convert_size_bytes",
    "find_component",
    "get_npz_shape",
    "load_library",
    "parse_bool",
    "header",
    "read_file",
    "string_substitute",
)

logger = logging.getLogger(__name__)


class BackgroundGenerator[T](threading.Thread):
    """Decorate a iterable to use a separate thread for compute"""

    def __init__(self, generator: Iterable[T], max_prefetch: int = 0) -> None:
        """Initialize the background generator thread."""
        super().__init__()
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.done = False
        self.start()

    def run(self) -> None:
        """Run the generator in a background thread."""
        for item in self.generator:
            self.queue.put(item)
        for _ in range(threading.active_count()):
            self.queue.put(self)

    def __next__(self) -> T:
        """Retrieve the next item from the queue."""
        if self.done:
            raise StopIteration()
        next_item = self.queue.get()
        self.done = next_item is self
        if self.done:
            raise StopIteration()
        return next_item

    def __iter__(self) -> Self:
        """Return the iterator object."""
        return self


def header(header: str, text: Any = "") -> None:
    """Print header with and optional value"""
    from pydtnn.utils.term import BOLD, RESET
    lines = [""]
    lines.append(f"# {BOLD}{header}{RESET}")
    if text:
        lines.append(str(text))
    info_to_print = "\n".join(lines)
    logger.info(info_to_print)


def parse_bool(x: str | Literal["true", "1", "yes", "y", "t"]) -> bool:
    """Returns True if value is a user truthy value"""
    return str(x).lower() in {"true", "1", "yes", "y", "t"}


def string_substitute(template: str, /, **mappings: Any) -> str:
    """Shell-like opportunistic substitution"""
    return string.Template(template).safe_substitute(mappings)


def convert_size(units: int, scale: int = 1000) -> str:
    """Convert number to human readable"""
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    if units > 0:
        i = int(math.log(units, scale))
        p = math.pow(scale, i)
        s = round(units / p, 2)
    else:
        i = 0
        s = 0
    return f"{s}{size_name[i]}"


def convert_size_bytes(size_bytes: int) -> str:
    """Convert byte count to human readable"""
    return f"{convert_size(size_bytes, scale=1024)}B"


def find_component(package: str, name: str) -> Any:
    """Find a file+class combo inside a package (with normalization)"""

    def normalize(text: str) -> str:
        """Normalize string by lowercasing and removing underscores."""
        return text.lower().replace("_", "")

    try:
        module = import_module(f"{package}.{name}")
    except Exception as e:
        raise ValueError(f"{name!r} not found in {package!r}!") from e

    for attr in dir(module):
        if not attr.startswith("_") and normalize(name) == normalize(attr):
            return getattr(module, attr)
    else:
        raise ValueError(f"{name!r} not found in {module!r}!")


def load_library(name: str) -> ctypes.CDLL:
    """
    Loads an external library using ctypes.CDLL.

    Parameters
    ----------
    name : str
        The library name without any prefix like lib, suffix like .so, .dylib or
        version number (this is the form used for the posix linker option -l).

    Returns
    -------
    The loaded library.
    """
    match sys.platform:
        case "linux" | "linux2":
            return ctypes.cdll.LoadLibrary(f"lib{name}.so")
        case "win32":
            return ctypes.windll.LoadLibrary(f"{name}.dll")  # pyright: ignore[reportAttributeAccessIssue]
        case "darwin":
            return ctypes.cdll.LoadLibrary(f"lib{name}.dylib")
        case _:
            raise NotImplementedError(f"{name} platform is not yet supported!")


def get_npz_shape(file: str) -> dict[str, tuple[int, ...]]:
    """Get NPZ member shapes without loading the archive"""
    shapes = dict[str, tuple[int, ...]]()

    with zipfile.ZipFile(file, "r") as z:
        for name in z.namelist():
            stem = PurePath(name).stem

            with z.open(name) as f:
                version = np.lib.format.read_magic(f)

                match version:
                    case (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                    case (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                    case version:
                        raise ValueError(f"Unsupported version: {version}")

                shapes[stem] = shape

    return shapes


def read_file(path: str, replaces: dict[str, str] = {}) -> str:
    """Read file's content from inside the package

    Args:
        path (str): Path to the file.
        replaces (dict[str, str] | None)
    Returns:
       file (str): The file as a string (str).

    """

    text = resources.read_text(package_name, path)
    # "prepocessor" (replacing generic "defines" and other sections of code with the actual code)
    for rep in replaces.items():
        text = text.replace(*rep)

    return text


def read_dir(path: str) -> list[str]:
    """List directory content from inside the package"""
    return [resource.name for resource in resources.files(package_name).joinpath(path).iterdir()]
