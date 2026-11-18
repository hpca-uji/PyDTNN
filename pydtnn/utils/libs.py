"""Dynamic library loader utilities"""

import errno
import ctypes
import functools
import itertools
import logging
import os
import shutil
import sys
from pathlib import Path
from collections.abc import Iterable
from importlib import metadata, resources
from importlib.resources.abc import Traversable

from pydtnn import package_name

__all__ = (
    "path_insert",
    "top_levels",
    "lib_names",
    "list_distribution",
    "list_resources",
    "load_library",
    "which"
)

logger = logging.getLogger(__name__)


def path_insert(path: str, value: str, index: int = 0) -> str:
    """Insert (if needed) a value into a PATH-like variable"""
    paths = path.split(os.pathsep) if path else []
    if value not in paths:
        paths.insert(index, value)
    return os.pathsep.join(paths)


def top_levels(name: str) -> Iterable[str]:
    """Get top levels from package"""
    dists = metadata.packages_distributions()

    for top, pkgs in dists.items():
        if name in pkgs:
            yield top


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
            entry.name: str(entry)
            for entry in entries
            if isinstance(entry, Path) and entry.name in libs and entry.exists()
        }

    dist = entry_map(list_distribution(src))
    resrc = entry_map(itertools.chain.from_iterable(list_resources(top) for top in top_levels(src)))

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
            logger.debug(f"DLL loaded {lib!r} (from system)")
            return module

    logger.debug(f"DLL missing {name!r}")
    raise FileNotFoundError(errno.ENOENT, f"No library named {name!r}", name)


@functools.cache
def which(name: str, mode: int = os.F_OK | os.X_OK, src: str = package_name) -> str:
    """
    Find an executable

    src: source package

    NOTE: Supports only concrete sources
    """

    def entry_exe(entries: Iterable) -> list:
        return [
            str(entry)
            for entry in entries
            if isinstance(entry, Path) and entry.name == name and os.access(entry, mode)
        ]

    dist = entry_exe(list_distribution(src))
    resrc = entry_exe(itertools.chain.from_iterable(list_resources(top) for top in top_levels(src)))

    # Find executable
    logger.debug(f"EXE finding {name!r} ({src or 'any'} preferred)")
    for entry in dist:
        logger.debug(f"EXE found {name!r} (from distribution)")
        return entry

    for entry in resrc:
        logger.debug(f"EXE found {name!r} (from resources)")
        return entry

    if entry := shutil.which(name, mode):
        logger.debug(f"EXE found {name!r} (from system)")
        return entry

    logger.debug(f"EXE missing {name!r}")
    raise FileNotFoundError(errno.ENOENT, f"No executable named {name!r}", name)
