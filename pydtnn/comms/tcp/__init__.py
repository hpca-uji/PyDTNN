"""TCP communications"""

import socket
import selectors
from concurrent.futures import ThreadPoolExecutor

from pydtnn import comms


__all__ = (
    "Protocol",
)


# Sentinel objects
NOTIFY_SELECT = b"\0"


class Protocol(comms.Communicator):
    """Shared base TCP implementation"""
    _max_message_size = 16 * 1024 ** 2 - 1
    _max_workers = 1

    def __init__(self, addr: str, port: int) -> None:
        """Inizialize comunicator"""
        super().__init__(addr, port)
        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"

        self._selector = selectors.DefaultSelector()
        self._selector_socket = socket.socketpair()
        self._selector.register(self._selector_socket[0], selectors.EVENT_READ, self._handle_selector_socket)

        self._pool = ThreadPoolExecutor(max_workers=1 + self._max_workers, thread_name_prefix=f"{thread_prefix}")
        self._selector_thread = self._pool.submit(self._handle_selector)

    def _modify_selector(self, fileobj, events) -> None:
        """Modify registered events"""
        key = self._selector.get_key(fileobj)
        if key.events != events:
            self._selector.modify(fileobj, events, key.data)

    def _notify_selector(self) -> None:
        """Interrupt selector loop"""
        assert len(NOTIFY_SELECT) == 1, "Sentinel of invalid size"
        self._selector_socket[1].send(NOTIFY_SELECT)

    def _handle_selector_socket(self, sock: socket.socket, mask) -> None:
        """Handle selector notification"""
        assert len(NOTIFY_SELECT) == 1, "Sentinel of invalid size"
        if sock.recv(len(NOTIFY_SELECT)) != NOTIFY_SELECT:
            raise comms.ResourceClosed()

    def _handle_selector(self) -> None:
        """Handle selector loop"""
        pending = []
        while True:
            for key, mask in self._selector.select():
                future = self._pool.submit(key.data, key.fileobj, mask)
                pending.append(future)
            for future in pending:
                future.result()
            pending.clear()

    def _close(self) -> None:
        """Close the communication"""
        self._selector_socket[1].close()
        try:
            self._selector_thread.result()
        except comms.ResourceClosed:
            pass
        self._selector_socket[0].close()
        self._selector.close()
        self._pool.shutdown()
        super()._close()
