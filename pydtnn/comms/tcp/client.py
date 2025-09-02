"""TCP client"""

import ssl
import copy
import socket
import selectors
import threading
from concurrent.futures import Future
import uuid

from pydtnn import comms
from pydtnn.comms import client
from pydtnn.utils import UUID_MAX
from pydtnn.comms.tcp import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions


__all__ = (
    "Client",
)


class Client(Protocol, client.Client):
    """TCP client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(copy.replace(options, workers=1))

        # TCP
        self._socket = socket.create_connection(self._options.netloc)

        if comms.SSL:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=comms.SSL_CERT)
            self._socket = context.wrap_socket(self._socket, server_hostname=self._options.netloc.host)

        self._socket.setblocking(False)

        self._selector.register(self._socket, selectors.EVENT_READ, self._handle_connection)

        self._put(self._session_ini(self._get_state(self._id)), self._id)

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection events"""
        state = self._get_state(self._id)

        if state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ)

        if event & selectors.EVENT_WRITE:
            self._c2s(sock)

        if event & selectors.EVENT_READ:
            self._s2c(sock)

        if not state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)

        self._notify_selector()

        if not state.state and state.put_empty():
            self._fin()

    def _s2c(self, sock: socket.socket) -> None:
        """Server to client communication"""
        state = self._get_state(self._id)

        try:
            data = sock.recv(self._options.connection.max_size)
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return

        if not data:
            assert not state.state and state.put_queue.empty(), "Lost connection unexpectedly"
            return

        state.get_write(data)

        if comms.SSL and (pending := sock.pending()):  # type: ignore
            data = sock.recv(pending)
            state.get_write(data)

        self._get_flush()

    def _c2s(self, sock: socket.socket) -> None:
        """Client to server communication"""
        state = self._get_state(self._id)

        state.put_flush()
        if state.put_buffer.empty():
            return
        with state.put_read() as view:
            try:
                size = sock.send(view)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                size = 0
            if size < len(view):
                state.put_buffer.unreadchunk(view[size:])

    def _extra_fin(self) -> None:
        """Close connection"""
        self._selector.unregister(self._socket)
        self._socket.close()
        del self._socket

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        self._modify_selector(self._socket, selectors.EVENT_READ | selectors.EVENT_WRITE)
        self._notify_selector()
        return future

    def _close(self) -> None:
        """Close the client"""
        self._put(self._session_fin(self._get_state(self._id)), self._id)

        with self._lock:
            while hasattr(self, "_socket"):
                self._lock.wait()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        super()._close()
