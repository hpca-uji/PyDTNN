from contextlib import contextmanager
import functools
import inspect
import logging
import os
from pathlib import Path
import threading
from traceback import TracebackException

from pydtnn import timestamp

logger = logging.getLogger(__name__)


def debug_line(*args) -> None:
    """Get line trace"""
    log = print

    frame_info = inspect.stack()[1]
    try:
        context = f"{frame_info.frame.f_globals["__name__"]}.{frame_info.function}:{frame_info.lineno}"
    finally:
        del frame_info

    log(f"{context} from {os.getpid()}:{threading.get_native_id()}", *args)


def debug_stack(*args, sep="|") -> None:
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


@contextmanager
def traceback_context():
    try:
        yield
    except Exception as exc:
        path = Path(f"traceback-{timestamp}.log").resolve()
        with path.open(mode="w") as file:
            TracebackException.from_exception(exc, capture_locals=True).print(file=file)
        logger.info(f'Dumped traceback details to: {path}')
        raise
