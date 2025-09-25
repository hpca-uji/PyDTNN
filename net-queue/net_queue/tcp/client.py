"""TCP client"""

import ssl
import uuid
import copy
import socket
import warnings
import selectors
from concurrent.futures import Future

from net_queue import client
from net_queue.tcp import Protocol
from net_queue.io_stream import Stream
from net_queue import CommunicatorOptions


__all__ = (
    "Communicator",
)


class Communicator(Protocol[socket.socket], client.Client[socket.socket]):
    """TCP client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(copy.replace(options, workers=1))

        # TCP
        self._comm = socket.create_connection(self.options.netloc)

        if self.options.security:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.options.security.certificate)
            self._comm = context.wrap_socket(self._comm, server_hostname=self.options.netloc.host)

        self._comm.setblocking(False)

        self._selector.register(self._comm, selectors.EVENT_READ, self._handle_connection)

        self._connection_ini(self._comm)

    def _handle_connection(self, comm: socket.socket, event) -> None:
        """Handle connection events"""
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        if state.put_empty():
            self._modify_selector(comm, selectors.EVENT_READ)

        if event & selectors.EVENT_WRITE:
            self._c2s(comm)

        if event & selectors.EVENT_READ:
            self._s2c(comm)

        if not state.put_empty():
            self._modify_selector(comm, selectors.EVENT_READ | selectors.EVENT_WRITE)

        self._notify_selector()

        if not state.state and state.put_empty():
            self._connection_fin(comm)

    def _s2c(self, comm: socket.socket) -> None:
        """Server to client communication"""
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        try:
            data = comm.recv(self.options.connection.max_size)
        except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return

        if not data:
            if state.state or not state.put_queue.empty():
                warnings.warn(f"Lost connection unexpectedly ({comm})", RuntimeWarning)
            return

        state.get_write(data)

        if self.options.security and (pending := comm.pending()):  # type: ignore
            data = comm.recv(pending)
            state.get_write(data)

        self._process_gets(peer)
        peer = state.peer

    def _c2s(self, comm: socket.socket) -> None:
        """Client to server communication"""
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        size = 0
        state.put_flush_queue()
        if state.put_buffer.empty():
            return
        with state.put_read() as view:
            try:
                size = comm.send(view)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                pass
            if size < len(view):
                state.put_buffer.unreadchunk(view[size:])
        self._put_commit(peer, size)

    def _connection_pre_fin(self, peer: uuid.UUID) -> None:
        """Close connection"""
        self._selector.unregister(self._comm)
        self._comm.close()
        del self._comm
        super()._connection_pre_fin(peer)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        self._modify_selector(self._comm, selectors.EVENT_READ | selectors.EVENT_WRITE)
        self._notify_selector()
        return future

    def _close(self) -> None:
        """Close the client"""
        comm = self._comm
        peer = self._comms.inverse[comm]
        self._session_fin(peer)

        with self._lock:
            while hasattr(self, "_comm"):
                self._lock.wait()

        super()._close()
