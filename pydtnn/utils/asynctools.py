"""Asynchronous utilities"""

from collections import abc
from concurrent import futures
from concurrent.futures import Future


__all__ = (
    "ChainFuture",
)


class ChainFuture(Future):
    """
    Combines multiple futures

    FIRST_COMPLETED: when any future finishes or is cancelled, returns its result
    FIRST_EXCEPTION: when any future finishes by raising an exception, raises its exception
    ALL_COMPLETED: when all futures finish or are cancelled, returns None

    When no futures are provied, returns None
    """

    def __init__(self, futures: abc.Iterable[Future], complete_when=futures.ALL_COMPLETED) -> None:
        """Initialize chained future"""
        super().__init__()
        self.futures = frozenset(futures)
        self.complete_when = complete_when
        self._futures_done = set()

        for future in self.futures:
            future.add_done_callback(self._handle_done)

        # Empty case
        if len(self.futures) <= 0:
            self._set_result(None)

    def _set_result(self, result) -> None:
        """Set result (if plausible)"""
        try:
            self.set_result(result)
        except futures.InvalidStateError:
            pass

    def _set_exception(self, exc: BaseException) -> None:
        """Set exception (if plausible)"""
        try:
            self.set_exception(exc)
        except futures.InvalidStateError:
            pass

    def _handle_done(self, future: Future) -> None:
        """Handle future done callback"""
        self._futures_done.add(future)

        # Single case
        try:
            result = future.result()
        except Exception as exc:
            if self.complete_when == futures.FIRST_EXCEPTION:
                self._set_exception(exc)
        else:
            if self.complete_when == futures.FIRST_COMPLETED:
                self._set_result(result)

        # Multi case
        if len(self._futures_done) >= len(self.futures):
            self._set_result(None)
