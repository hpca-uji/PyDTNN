"""TCP client"""

import ssl
import uuid
import socket
import selectors
import threading
from queue import SimpleQueue
from concurrent.futures import Future

from pydtnn import comms
from pydtnn.comms.tcp import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_NIL, UUID_MAX
from pydtnn.comms import ConnectionState, Message, ResourceClosed, ConnectionData


__all__ = (
    "Client",
)


class Client(Protocol):
    """TCP client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._state = ConnectionData()

        # TCP
        self._socket = socket.create_connection((self._addr, self._port))

        if comms.SSL:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=comms.SSL_CERT)
            self._socket = context.wrap_socket(self._socket, server_hostname=self._addr)

        self._socket.setblocking(False)

        self._selector.register(self._socket, selectors.EVENT_READ, self._handle_connection)

        self._session_ini()

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection events"""
        state = self._state

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
            self._fin(sock)

    def _s2c(self, sock: socket.socket) -> None:
        """Server to client communication"""
        state = self._state

        while True:
            try:
                data = sock.recv(self._max_payload_size)
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                break

            if not data:
                assert not state.state and state.put_queue.empty(), "Lost connection unexpectedly"
                return

            state.get_buffer.write(data)
            self._get_flush()

    def _get_flush(self) -> None:
        state = self._state
        peer = state.peer

        while True:
            try:
                stream = state.get()
            except BlockingIOError:
                break

            if stream.empty():
                self._handle_session_fin(stream)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _c2s(self, sock: socket.socket) -> None:
        """Client to server communication"""
        state = self._state

        state.put_flush()
        if state.put_buffer.empty():
            return
        with state.put_read(self._max_payload_size) as view:
            try:
                size = sock.send(view)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                size = 0
            if size < len(view):
                state.put_buffer.unreadchunk(view[size:])

    def _fin(self, sock: socket.socket) -> None:
        """Close connection"""
        self._selector.unregister(sock)
        sock.close()

        with self._lock:
            del self._socket
            self._lock.notify_all()

    def _put(self, stream: Stream) -> Future[None]:
        """Put stream into queue and notify"""
        future = self._state.put(stream)
        self._modify_selector(self._socket, selectors.EVENT_READ | selectors.EVENT_WRITE)
        self._notify_selector()
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to server"""
        assert len(peers) == 0, "Client can not publish to another client"
        stream = self._serializer.dump(obj)
        return self._put(stream)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get from the server"""
        assert len(peers) == 0, "Client can not get from another client"
        peer = self._get_event.get()

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=state.peer, obj=obj)

    def _session_ini(self) -> None:
        """Send session ini message"""
        state = self._state
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        self.put(self._id)

    def _session_fin(self) -> None:
        """Send session fin message"""
        state = self._state
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        self._put(Stream())

    def _handle_session_ini(self, stream: Stream) -> None:
        """Handle session initialize message"""
        state = self._state
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

    def _handle_session_fin(self, stream: Stream) -> None:
        """Handle session finalize message"""
        state = self._state
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"

        stream.close()
        state.state &= ~ConnectionState.READABLE

    def _close(self) -> None:
        """Close the client"""
        self._session_fin()

        with self._lock:
            while hasattr(self, "_socket"):
                self._lock.wait()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        super()._close()
