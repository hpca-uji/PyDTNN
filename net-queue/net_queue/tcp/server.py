"""TCP server"""

import ssl
import uuid
import socket
import warnings
import selectors
from concurrent.futures import Future

from net_queue import server
from net_queue.tcp import Protocol
from net_queue.io_stream import Stream
from net_queue import CommunicatorOptions, ResourceClosed


__all__ = (
    "Communicator",
)


class Communicator(Protocol[socket.socket], server.Server[socket.socket]):
    """TCP server"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # TCP
        self._comm = socket.create_server(self.options.netloc, reuse_port=True)

        if self.options.security:
            if self.options.security.certificate is None or self.options.security.key is None:
                raise RuntimeError("SSL certificate or key not provided")
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=self.options.security.certificate)
            context.load_cert_chain(certfile=self.options.security.certificate, keyfile=self.options.security.key)
            self._comm = context.wrap_socket(self._comm, server_side=True, do_handshake_on_connect=True)

        self._selector.register(self._comm, selectors.EVENT_READ, self._new_connection)
        self._notify_selector()

    def _new_connection(self, comm: socket.socket, event) -> None:
        """Handle connection accept"""
        comm, _ = comm.accept()
        comm.setblocking(False)
        self._selector.register(comm, selectors.EVENT_READ, self._handle_connection)

    def _handle_connection(self, comm: socket.socket, event) -> None:
        """Handle connection events"""
        # NOTE: communication thead
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        if state.put_empty():
            self._modify_selector(comm, selectors.EVENT_READ)

        if event & selectors.EVENT_WRITE:
            self._s2c(comm)

        if event & selectors.EVENT_READ:
            self._c2s(comm)

        if not state.put_empty():
            self._modify_selector(comm, selectors.EVENT_READ | selectors.EVENT_WRITE)

        self._notify_selector()

        if not state.state and state.put_empty():
            self._connection_fin(comm)

    def _c2s(self, comm: socket.socket) -> None:
        """Client to server communication"""
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

    def _s2c(self, comm: socket.socket) -> None:
        """Server to client communication"""
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
        comm = self._comms[peer]
        self._selector.unregister(comm)
        comm.close()
        super()._connection_pre_fin(peer)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        try:
            comm = self._comms[peer]
            future = super()._put(stream, peer)
        except (KeyError, ResourceClosed):
            raise ResourceClosed(peer)
        self._modify_selector(comm, selectors.EVENT_READ | selectors.EVENT_WRITE)
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        try:
            return super().put(obj, *peers)
        finally:
            self._notify_selector()

    def _close(self) -> None:
        """Close the server"""
        self._comm.close()

        # Wait peers to drain
        with self._lock:
            while self._comms:
                self._lock.wait()

        super()._close()
