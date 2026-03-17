import logging
logger = logging.getLogger(__name__)

import os
import sys
import math
import string
import ctypes
import threading
from queue import Queue
from importlib import import_module
from ctypes.util import find_library
from collections.abc import Iterable


class BackgroundGenerator[T](threading.Thread):
    """Decorate a iterable to use a separate thread for compute"""

    def __init__(self, generator: Iterable[T], max_prefetch=0):
        super().__init__()
        self.queue = Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.done = False
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        for _ in range(threading.active_count()):
            self.queue.put(self)

    def __next__(self) -> T:
        if self.done:
            raise StopIteration()
        next_item = self.queue.get()
        self.done = next_item is self
        if self.done:
            raise StopIteration()
        return next_item

    def __iter__(self):
        return self


def print_with_header(header: str, to_be_printed=None) -> None:
    """Print header with and optional value"""
    to_print = list[str]()
    to_print.append(f"# {header}")
    if to_be_printed is not None:
        to_print.append(to_be_printed)
    info_to_print = '\n'.join(to_print)
    logger.info(info_to_print)

def parse_bool(x) -> bool:
    """Returns True if value is a user truthy value"""
    return str(x).lower() in {'true', '1', 'yes', 'y', 't'}


def string_substitute(template: str, /, **mappings) -> str:
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


def find_component(package: str, name: str):
    """Find a file+class combo inside a package (with normalization)"""
    def normalize(text: str) -> str:
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


def load_library(name: str):
    """
    Loads an external library using ctypes.CDLL.

    It searches the library using ctypes.util.find_library(). If the library is
    not found, it traverses the LD_LIBRARY_PATH until it finds it. If it is not
    in any of the LD_LIBRARY_PATH paths, an ImportError exception is raised.

    Parameters
    ----------
    name : str
        The library name without any prefix like lib, suffix like .so, .dylib or
        version number (this is the form used for the posix linker option -l).

    Returns
    -------
    The loaded library.
    """
    path = find_library(name)
    if path is None:
        if sys.platform in ('linux2', 'linux'):
            full_name = f"lib{name}.so"
        elif sys.platform == 'darwin':
            full_name = f"lib{name}.dylib"
        elif sys.platform == 'win32':
            full_name = f"lib{name}.dll"
        else:
            raise NotImplementedError(f"Trying to load '{name}' library, but platform '{sys.platform}' is not yet supported!")

        for current_path in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
            if os.path.exists(os.path.join(current_path, full_name)):
                path = os.path.join(current_path, full_name)
                break
        else:
            # Didn't find the library
            raise ImportError(f"Library '{full_name}' could not be found. Please add its path to LD_LIBRARY_PATH "
                              f"using 'export LD_LIBRARY_PATH={name.upper()}_LIB_PATH:$LD_LIBRARY_PATH' and "
                              f"then call this application again.")
    return ctypes.CDLL(path)
