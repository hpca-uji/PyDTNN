"""Dynamic library loader utilities"""

import ctypes
import ctypes.util
import functools
import itertools
import logging
import os
import sys
from collections.abc import Iterable
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path

from pydtnn import package_name

__all__ = (
    "lib_names",
    "list_distribution",
    "list_resources",
    "load_library",
)

logger = logging.getLogger(__name__)


def lib_names(name: str, ver: str = "") -> Iterable[str]:
    """Generate posible library filenames"""
    pres = ["lib", ""]
    seps = [".", "-", "_"]
    vers = ver.split(".") if ver else []
    vers = [".".join(vers[:i]) for i in range(len(vers), -1, -1)]

    # Define template
    match sys.platform:
        case "win32":
            seps = ["-", "_", "."]
            tmp = "{name}{ver}.dll"
        case "darwin":
            tmp = "{name}{ver}.dylib"
        case _:
            tmp = "{name}.so{ver}"

    # Compute posibilites
    for ver, pre, sep in itertools.product(vers, pres, seps):
        kwds = {"name": name, "ver": ver}
        if pre:
            kwds["name"] = f"{pre}{name}"
        if ver:
            kwds["ver"] = f"{sep}{ver}"
        yield tmp.format(**kwds)


def list_distribution(name: str) -> Iterable:
    """List distribution contents"""
    try:
        paths = metadata.files(name) or []
    except metadata.PackageNotFoundError:
        paths = []
    else:
        paths = [path.locate() for path in paths]
    return paths


def list_resources(name: str) -> Iterable:
    """List resource container contents"""

    def walk(path: Traversable) -> Iterable[Traversable]:
        yield path
        if not path.is_dir():
            return
        for path in path.iterdir():
            yield from walk(path)

    try:
        path = resources.files(name)
    except ModuleNotFoundError:
        paths = []
    else:
        paths = [path for path in walk(path)]
    return paths


@functools.cache
def load_library(name: str, ver: str = "", src: str = package_name) -> ctypes.CDLL:
    """
    Loads an external library using ctypes.

    name: without prefix or suffix
    src: source package
    ver: full version

    NOTE: Supports only concrete sources
    """

    # Define loader
    match sys.platform:
        case "win32":
            loader = ctypes.windll.LoadLibrary  # pyright: ignore[reportAttributeAccessIssue]
        case _:
            loader = ctypes.cdll.LoadLibrary

    # Explore package
    order = list(lib_names(name, ver))
    libs = frozenset(order)

    def entry_map(entries: Iterable) -> dict:
        return {
            entry.name: entry
            for entry in entries
            if isinstance(entry, Path) and entry.name in libs
        }

    root = src.split(".", 1)[0]
    dist = entry_map(list_distribution(src))
    resrc = entry_map(list_resources(root))

    # Find library
    logger.debug(f"DLL finding {name!r} ({ver or 'any'} in {src!r} preferred)")
    for lib in order:
        if entry := dist.get(lib):
            module = loader(entry)
            logger.debug(f"DLL loaded {lib!r} (from distribution)")
            return module

        if entry := resrc.get(lib):
            module = loader(entry)
            logger.debug(f"DLL loaded {lib!r} (from resources)")
            return module

        try:
            module = loader(lib)
        except Exception:
            pass
        else:
            logger.debug(f"DLL loaded {lib!r} (from environment)")
            return module

    # Fallback
    if lib := ctypes.util.find_library(name):
        logger.debug(f"DLL loaded {name!r} (from system)")
        return loader(lib)

    logger.debug(f"DLL missing {name!r}")
    raise ModuleNotFoundError(f"No library named {name!r}", name=name, path=src)
