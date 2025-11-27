import os
import sys
import math
import string
import ctypes
import inspect
import weakref
import functools
import threading
from glob import glob
from queue import Queue
from importlib import import_module
from ctypes.util import find_library
from collections.abc import Iterable

import numpy as np


class BackgroundGenerator[T](threading.Thread):
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


class Random:
    def __init__(self, seed=0) -> None:
        self._generators = weakref.WeakKeyDictionary[threading.Thread, np.random.Generator]()
        self.seed(seed)

    def seed(self, seed) -> None:
        self._seed = seed
        self._generators.clear()

    @property
    def _generator(self) -> np.random.Generator:
        thread = threading.current_thread()

        if thread not in self._generators:
            self._generators[thread] = np.random.default_rng(self._seed)

        return self._generators[thread]

    def __getattr__(self, key: str):
        return getattr(self._generator, key)


random: np.random.Generator = Random()  # type: ignore


def print_with_header(header, to_be_printed=None):
    print(f"# {header}")
    if to_be_printed is not None:
        print(to_be_printed)


def parse_bool(x):
    """Returns True if value is a user truthy value"""
    return str(x).lower() in {'true', '1', 'yes', 'y', 't'}


def get_attr_factory(o, name, factory):
    try:
        return getattr(o, name)
    except AttributeError:
        return factory()


def set_attr_default(o, name, value):
    try:
        return getattr(o, name)
    except AttributeError:
        setattr(o, name, value)
        return value


def set_attr_default_factory(o, name, factory):
    try:
        return getattr(o, name)
    except AttributeError:
        value = factory()
        setattr(o, name, value)
        return value

def load_library(name):
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


def blis():
    if not hasattr(blis, "lib"):
        blis.lib = load_library("blis")
    return blis.lib


def mkl():
    if not hasattr(mkl, "lib"):
        mkl.lib = load_library("mkl_rt")
    return mkl.lib


def convert_size(units: int, scale: int = 1000):
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    if units > 0:
        i = int(math.log(units, scale))
        p = math.pow(scale, i)
        s = round(units / p, 2)
    else:
        i = 0
        s = 0
    return f"{s}{size_name[i]}"


def convert_size_bytes(size_bytes):
    return f"{convert_size(size_bytes, scale=1024)}B"


# Matmul operation
# Warning: the output matrix can not be cached, as it will persist outside this method
def matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
    # if a.dtype == np.float32:
    #    c = slb.sgemm(1.0, a, b)
    # elif a.dtype == np.float64:
    #    c = slb.dgemm(1.0, a, b)
    # else:
    # Native numpy matmul gets more performance than scipy blas!
    if c is None:
        return a @ b
    else:
        return np.matmul(a, b, c)


def _matmul_xgemm(called_from, lib, a, b, c=None):
    order = 101  # 101 for row-major, 102 for column major data structures
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]
    if c is None:
        c = np.ones((m, n), a.dtype, order="C")
    # trans_{a,b} = 111 for no transpose, 112 for transpose, and 113 for conjugate transpose
    if a.flags["C_CONTIGUOUS"]:
        trans_a = 111
        lda = k
    elif a.flags["F_CONTIGUOUS"]:
        trans_a = 112
        lda = m
    else:
        raise ValueError(f"Matrix a data layout not supported by {called_from}().")
    if b.flags["C_CONTIGUOUS"]:
        trans_b = 111
        ldb = n
    elif b.flags["F_CONTIGUOUS"]:
        trans_b = 112
        ldb = k
    else:
        raise ValueError(f"Matrix b data layout not supported by {called_from}().")
    ldc = n
    alpha = 1.0
    beta = 0.0
    if a.dtype == np.float32:
        lib.cblas_sgemm(ctypes.c_int(order), ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_float(alpha),
                        ctypes.c_void_p(a.ctypes.data), ctypes.c_int(lda),
                        ctypes.c_void_p(b.ctypes.data), ctypes.c_int(ldb),
                        ctypes.c_float(beta), ctypes.c_void_p(c.ctypes.data), ctypes.c_int(ldc))
    elif a.dtype == np.float64:
        lib.cblas_dgemm(ctypes.c_int(order), ctypes.c_int(trans_a), ctypes.c_int(trans_b),
                        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k), ctypes.c_double(alpha),
                        ctypes.c_void_p(a.ctypes.data), ctypes.c_int(lda),
                        ctypes.c_void_p(b.ctypes.data), ctypes.c_int(ldb),
                        ctypes.c_double(beta), ctypes.c_void_p(c.ctypes.data), ctypes.c_int(ldc))
    else:
        raise TypeError(f"Type '{a.dtype}' not supported by {called_from}().")
    return c


def matmul_mkl(a, b, c=None):
    # os.environ['GOMP_CPU_AFFINITY'] = ""
    # os.environ['OMP_PLACES'] = ""
    return _matmul_xgemm("matmul_mkl", mkl(), a, b, c)


def matmul_blis(a, b, c=None):
    return _matmul_xgemm("matmul_blis", blis(), a, b, c)


def string_substitute(template, /, **mappings):
    """Shell-like opportunistic substitution"""
    return string.Template(template).safe_substitute(mappings)


def debug_line(*args):
    """Get line trace"""
    log = print

    frame_info = inspect.stack()[1]
    try:
        context = f"{frame_info.frame.f_globals["__name__"]}.{frame_info.function}:{frame_info.lineno}"
    finally:
        del frame_info

    log(f"{context} from {os.getpid()}:{threading.get_native_id()}", *args)


def debug_stack(*args, sep="|"):
    """Get stack trace"""
    log = print

    stack = inspect.stack()[1:]
    try:
        context = sep.join(
            f"{frame_info.frame.f_globals["__name__"]}.{frame_info.function}:{frame_info.lineno}"
            for frame_info in stack
        )
    finally:
        del stack

    log(f"{context} from {os.getpid()}:{threading.get_native_id()}", *args)


def debug_func(func):
    """Functions trace decorator"""
    log = print

    @functools.wraps(func)
    def wrapper(*args, **kwds):
        header = "DEBUG"
        frame_info = inspect.stack()[1]
        try:
            context = f"{func.__qualname__}{args!r}{kwds!r} from {frame_info.frame.f_globals["__name__"]}.{frame_info.function}:{frame_info.lineno} from {os.getpid()}:{threading.get_native_id()}"
        finally:
            del frame_info
        log(f"{header}: Call {context}")
        try:
            result = func(*args, **kwds)
        except BaseException as exc:
            log(f"{header}: Exc. {context} = {exc!r}")
            raise
        else:
            log(f"{header}: Ret. {context} = {result!r}")
            return result

    return wrapper


def find_component(package: str, name: str):
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
