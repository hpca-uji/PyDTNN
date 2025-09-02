"""TCP server"""

import ssl
import uuid
import socket
import selectors
import threading
from concurrent.futures import Future

from pydtnn import comms
from pydtnn.comms import server
from pydtnn.utils import UUID_MAX
from pydtnn.comms.tcp import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions, ResourceClosed


__all__ = (
    "Server",
)


class Server(Protocol, server.Server[socket.socket]):
    """TCP server"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # TCP
        self._socket = socket.create_server(self._options.netloc, reuse_port=True)

        if comms.SSL:
            if comms.SSL_CERT is None or comms.SSL_KEY is None:
                raise RuntimeError("SSL certificate or key not provided")
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=comms.SSL_CERT)
            context.load_cert_chain(certfile=comms.SSL_CERT, keyfile=comms.SSL_KEY)
            self._socket = context.wrap_socket(self._socket, server_side=True, do_handshake_on_connect=True)

        self._selector.register(self._socket, selectors.EVENT_READ, self._new_socket)
        self._notify_selector()

    def _new_socket(self, sock: socket.socket, event) -> None:
        sock, _ = self._socket.accept()
        sock.setblocking(False)
        self._selector.register(sock, selectors.EVENT_READ, self._handle_connection)

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection events"""
        # NOTE: communication thead
        peer = self._get_peer(sock)
        state = self._get_state(peer)

        if state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ)

        if event & selectors.EVENT_WRITE:
            self._s2c(sock)

        if event & selectors.EVENT_READ:
            self._c2s(sock)

        if not state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)

        self._notify_selector()

        if not state.state and state.put_empty():
            peer = self._get_peer(sock)
            self._fin(peer)

    def _c2s(self, sock: socket.socket) -> None:
        """Client to server communication"""
        peer = self._get_peer(sock)
        state = self._get_state(peer)

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

        peer = self._get_flush(peer)

    def _s2c(self, sock: socket.socket) -> None:
        """Server to client communication"""
        peer = self._get_peer(sock)
        state = self._get_state(peer)

        state.put_flush()
        if state.put_buffer.empty():
            return
        with state.put_read(self._options.connection.max_size) as view:
            try:
                size = sock.send(view)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                size = 0
            if size < len(view):
                state.put_buffer.unreadchunk(view[size:])

    def _extra_fin(self, peer: uuid.UUID) -> None:
        sock = self._peers[peer]
        self._selector.unregister(sock)
        sock.close()
        super()._extra_fin(peer)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        try:
            sock = self._peers[peer]
            future = super()._put(stream, peer)
        except (KeyError, ResourceClosed):
            raise ResourceClosed(peer)
        self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        try:
            return super().put(obj, *peers)
        finally:
            self._notify_selector()

    def _close(self) -> None:
        """Close the server"""
        self._socket.close()

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        # Unlock inflight external API
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        super()._close()
