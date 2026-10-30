"""Utility functions and classes for the PyDTNN framework."""

import itertools
import logging
import math
import string
import threading
import zipfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from gzip import GzipFile
from importlib import import_module, resources
from pathlib import Path, PurePosixPath
from queue import Queue
from typing import IO, Any, Literal, Self

import numpy as np

try:
    from rapidgzip import RapidgzipFile
except ModuleNotFoundError:
    RapidgzipFile = None

from pydtnn import package_name

__all__ = (
    "BackgroundGenerator",
    "convert_size",
    "convert_size_bytes",
    "find_component",
    "get_npz_shape",
    "gzip_open",
    "header",
    "parse_bool",
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


def gzip_open(filename: str) -> IO[bytes]:
    """
    Open a gZIP file.

    This method handles opening gZIP files, creating or loading their
    seek tables for efficient random access.

    Args:
        filename: The path to the gZIP file.

    Returns:
        A file-like object opened in binary read mode.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If there's an error during gzip file operations.
    """
    f = None
    mode = "rb"
    path = Path(filename)
    plain = path.with_suffix("")
    idx = path.with_suffix(f"{path.suffix}.idx")

    if plain.exists():
        return open(plain, mode=mode)
    elif not path.exists():
        raise FileNotFoundError(path)
    elif RapidgzipFile:
        f = RapidgzipFile(path, parallelization=1)
        try:
            f.import_index(str(idx))
        except Exception:
            f.export_index(str(idx))
    else:
        f = GzipFile(path, mode=mode)

    return f


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
    raise ValueError(f"{name!r} not found in {module!r}!")


def get_npz_shape(file: str) -> dict[str, tuple[int, ...]]:
    """Get NPZ member shapes without loading the archive"""
    shapes = dict[str, tuple[int, ...]]()

    with zipfile.ZipFile(file, "r") as z:
        for name in z.namelist():
            stem = PurePosixPath(name).stem

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


def map_merge[T: Mapping](*maps: T) -> T:
    """Merge mappings recursively"""
    dst = {}

    for src in maps:
        for key, value in src.items():
            if key in dst and isinstance(dst[key], Mapping) and isinstance(value, Mapping):
                dst[key] = map_merge(dst[key], value)
            else:
                dst[key] = deepcopy(value)

    return dst


def map_factor[T: Mapping](*maps: T) -> tuple[T, list[T]]:
    """Factor mappings recursively"""
    if not maps:
        return {}, []

    share = {}
    diffs = [{} for _ in maps]
    keys = dict.fromkeys(itertools.chain.from_iterable(maps))

    for key in keys:
        values = [src[key] for src in maps if key in src]

        if all(isinstance(v, Mapping) for v in values):
            sub_share, sub_diffs = map_factor(*values)
            if sub_share:
                share[key] = sub_share
            for dst, src in zip(diffs, sub_diffs):
                if src:
                    dst[key] = src

        elif all(v == values[0] for v in values[1:]):
            share[key] = deepcopy(values[0])

    for i, src in enumerate(maps):
        for key, value in src.items():
            if key not in share and key not in diffs[i]:
                diffs[i][key] = deepcopy(value)

    return share, diffs
