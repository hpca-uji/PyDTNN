"""TCP communications"""

# NOTE: Module considerations
#
# None

import socket
import selectors
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms


__all__ = (
    "Protocol",
)


# Sentinel objects
END_COMM = b"\0"


class Protocol(comms.Communication):
    """Shared base TCP implementation"""

    def __init__(self, addr: str, port: int) -> None:
        super().__init__(addr, port)
        self._selector = selectors.DefaultSelector()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._notify_close, wait_close = socket.socketpair()
        self._selector.register(wait_close, selectors.EVENT_READ, self._handle_close)

    def _start_loop(self) -> None:
        """Start connection handling loop"""
        Thread(target=self._handle_selector).start()

    def _handle_close(self, sock: socket.socket, event) -> None:
        """Handle close notification"""
        sock.recv(len(END_COMM))
        sock.close()

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _handle_selector(self):
        """Handle selector loop"""
        while not self.closed:
            for _ in self._pool.map(self._handle_selector_event, self._selector.select()):
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
        self._notify_close.sendall(END_COMM)
        self._notify_close.close()
        self._selector.close()
        self._pool.shutdown()
