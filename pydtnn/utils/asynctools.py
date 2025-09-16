"""Asynchronous utilities"""

from collections import abc
from concurrent import futures
from concurrent.futures import Future


__all__ = (
    "merge_futures",
    "future_set_running",
    "future_set_result",
    "future_set_exception"
)


def future_set_running(future: Future) -> bool:
    """Set future running (if plausible)"""
    try:
        return future.set_running_or_notify_cancel()
    except RuntimeError:
        return False


def future_set_result(future: Future, result) -> bool:
    """Set future result (if plausible)"""
    try:
        future.set_result(result)
    except futures.InvalidStateError:
        return False
    else:
        return True


def future_set_exception(future: Future, exc: BaseException) -> bool:
    """Set future exception (if plausible)"""
    try:
        future.set_exception(exc)
    except futures.InvalidStateError:
        return False
    else:
        return True


def merge_futures(fs: abc.Iterable[Future], return_when=futures.ALL_COMPLETED) -> Future:
    """
    Combines multiple futures

    FIRST_COMPLETED: when any future finishes or is cancelled, returns its result
    FIRST_EXCEPTION: when any future finishes by raising an exception, raises its exception
    ALL_COMPLETED: when all futures finish or are cancelled, returns None

    When no futures are provied, returns None
    """
    future = Future()
    fs = frozenset(fs)
    done = set()

    # Callbacks
    def handle_done(future):
        done.add(future)

        # Single case
        try:
            result = future.result()
        except Exception as exc:
            if return_when == futures.FIRST_COMPLETED or return_when == futures.FIRST_EXCEPTION:
                future_set_exception(future, exc)
        else:
            if return_when == futures.FIRST_COMPLETED:
                future_set_result(future, result)

        # Multi case
        if len(done) >= len(fs):
            future_set_result(future, None)

    future.futures = fs  # type: ignore
    future_set_running(future)

    for future in fs:
        future.add_done_callback(handle_done)

    # Empty case
    if len(fs) <= 0:
        future_set_result(future, None)

    return future
