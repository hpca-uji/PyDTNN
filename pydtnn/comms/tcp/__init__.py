"""TCP communications"""

# NOTE: Module considerations
#
# None

# FIXME: Use file descriptor to wakeup selector insted of polling

import selectors
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms


__all__ = (
    "Protocol",
)


class Protocol(comms.Communication):
    """Shared base TCP implementation"""
    _poll_interval = 1.0

    def __init__(self, addr: str, port: int) -> None:
        super().__init__(addr, port)
        self._selector = selectors.DefaultSelector()
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _handle_selector(self):
        """Handle selector loop"""
        while not self.closed:
            for _ in self._pool.map(self._handle_selector_event, self._selector.select(self._poll_interval)):
                pass

    def _handle_selector_event(self, event: tuple[selectors.SelectorKey, int]) -> None:
        """Handle selector event"""
        key, mask = event
        callback = key.data
        callback(key.fileobj, mask)

    def close(self) -> None:
        """Close the communication"""
        if self.closed:
            return
        super().close()
        self._selector.close()
        self._pool.shutdown()
