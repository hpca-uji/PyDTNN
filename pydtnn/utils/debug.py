"""
Debugging utilities for the PyDTNN framework.

Provides tools for tracing function calls, stack execution, and logging
detailed exception tracebacks to files.
"""

import functools
import inspect
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from traceback import TracebackException

from pydtnn import timestamp

__all__ = (
    "debug_func",
    "debug_line",
    "debug_stack",
    "traceback_context",
)

logger = logging.getLogger(__name__)


def debug_line(*args) -> None:
    """
    Print the current file, function, and line number to standard output.

    Args:
        *args: Variable length argument list to be printed alongside the trace.
    """
    log = print

    frame_info = inspect.stack()[1]
    try:
        context = f"{frame_info.frame.f_globals['__name__']}.{frame_info.function}:{frame_info.lineno}"
    finally:
        del frame_info

    log(f"{context} from {os.getpid()}:{threading.get_native_id()}", *args)


def debug_stack(*args, sep="|") -> None:
    """
    Print the current call stack trace to standard output.

    Args:
        *args: Variable length argument list to be printed alongside the trace.
        sep: Separator string used to join stack frames.
    """
    log = print

    stack = inspect.stack()[1:]
    try:
        context = sep.join(f"{frame_info.frame.f_globals['__name__']}.{frame_info.function}:{frame_info.lineno}" for frame_info in stack)
    finally:
        del stack

    log(f"{context} from {os.getpid()}:{threading.get_native_id()}", *args)


def debug_func(func):
    """
    Decorator to trace function calls, arguments, return values, and exceptions.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function with tracing logic.
    """
    log = print

    @functools.wraps(func)
    def wrapper(*args, **kwds):
        header = "DEBUG"
        frame_info = inspect.stack()[1]
        try:
            context = f"{func.__qualname__}{args!r}{kwds!r} from {frame_info.frame.f_globals['__name__']}.{frame_info.function}:{frame_info.lineno} from {os.getpid()}:{threading.get_native_id()}"
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


@contextmanager
def traceback_context():
    """
    Context manager that catches exceptions and writes a detailed traceback to a log file.

    The traceback includes local variables and is saved with a timestamped filename.
    """
    try:
        yield
    except Exception as exc:
        path = Path(f"traceback-{timestamp}.log").resolve()
        with path.open(mode="w") as file:
            TracebackException.from_exception(exc, capture_locals=True).print(file=file)
        logger.info(f"Dumped traceback details to: {path}")
        raise
