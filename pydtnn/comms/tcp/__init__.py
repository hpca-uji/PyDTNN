"""TCP communications"""

# NOTE: Module considerations
#
# None

import socket
import selectors
from threading import Thread
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms


__all__ = (
    "Protocol",
)


# Sentinel objects
NOTIFY_SELECT = b"\0"


class Protocol(comms.Communication):
    """Shared base TCP implementation"""

    def __init__(self, addr: str, port: int) -> None:
        super().__init__(addr, port)
        self._selector = selectors.DefaultSelector()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._selector_notifier = socket.socketpair()
        self._selector.register(self._selector_notifier[0], selectors.EVENT_READ, self._handle_selector_notify)
        self._selector_thread = Thread(target=self._handle_selector)

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _start_loop(self) -> None:
        """Start connection handling loop"""
        self._selector_thread.start()

    def _notify_selector(self) -> None:
        """Interrupt selector loop"""
        self._selector_notifier[1].send(NOTIFY_SELECT)

    def _modify_selector(self, fileobj, events) -> None:
        """Modify registered events"""
        key = self._selector.get_key(fileobj)
        self._selector.modify(fileobj, events, key.data)
        self._notify_selector()

    def _handle_selector_notify(self, sock: socket.socket, event) -> None:
        """Handle selector notification"""
        self._selector_notifier[0].recv(len(NOTIFY_SELECT))

    def _handle_selector(self):
        """Handle selector loop"""
        while not self.closed:
            pending = []
            for key, mask in self._selector.select():
                try:
                    future = self._submit(key.data, key.fileobj, mask)
                except RuntimeError:
                    break
                else:
                    pending.append(future)
            futures.wait(pending)

    def close(self) -> None:
        """Close the communication"""
        if self.closed:
            return
        super().close()
        self._notify_selector()
        self._selector_notifier[1].close()
        self._selector_thread.join()
        self._selector_notifier[0].close()
        self._selector.close()
        self._pool.shutdown()
