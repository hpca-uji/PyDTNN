"""TCP client"""

import uuid
import socket
import selectors
import threading

from pydtnn.comms.tcp import Protocol
from pydtnn.utils.io_stream import AncillaryStream, Stream
from pydtnn.comms import Message, ResourceClosed, ConnectionState


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = Stream()


class Client(Protocol):
    """TCP client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._state = ConnectionState(buffer_size=self._max_message_size)

        # TCP
        self._socket = socket.create_connection((self._addr, self._port))
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._max_message_size)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._max_message_size)
        self._socket.setblocking(False)

        self._ini()

    def _ini(self) -> None:
        """Connection initialization"""
        stream = self._serializer.dump(self._id)
        self._state.put(stream, ancillary=True)
        self._selector.register(self._socket, selectors.EVENT_READ | selectors.EVENT_WRITE, self._handle_connection)
        self._notify_selector()

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection states"""
        state = self._state

        if state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ)

        if event & selectors.EVENT_READ:
            self._s2c(sock)

        if event & selectors.EVENT_WRITE:
            self._c2s(sock)

        if not state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)

        if state.closed and state.put_empty():
            self._fin(sock)

        self._notify_selector()

    def _fin(self, sock: socket.socket) -> None:
        self._selector.unregister(sock)
        with self._lock:
            sock.close()
            del self._socket
            self._lock.notify_all()

    def _s2c(self, sock: socket.socket) -> None:
        state = self._state
        data = sock.recv(self._max_message_size)

        if not data:
            self._fin(sock)
            return

        state.get_stream.write(data)

        while True:
            try:
                stream = state.get()
            except AncillaryStream as ancillary:
                with ancillary.stream as stream:
                    id = self._serializer.load(stream)

                # Server ID, INI
                if id != self._id:

                    # ACK
                    with self._lock:
                        self._server = id
                        self._lock.notify_all()

                # Client ID, FIN
                else:

                    # ACK
                    state.close()
            except BlockingIOError:
                break
            else:
                state.get_queue.put(stream)

    def _c2s(self, sock: socket.socket) -> None:
        state = self._state

        state.put_flush()
        if state.put_stream.empty():
            return
        with state.put_read() as view:
            size = sock.send(view)
            if size < len(view):
                state.put_stream.unreadchunk(view[size:])

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        stream = self._serializer.dump(obj)
        self._state.put(stream)
        self._modify_selector(self._socket, selectors.EVENT_READ | selectors.EVENT_WRITE)
        self._notify_selector()

    def get(self, *peers: uuid.UUID) -> Message:
        """Get from the server"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"
        with self._state.get_queue.get() as stream:
            if stream is END_COMM:
                raise ResourceClosed()
            obj = self._serializer.load(stream)
        return Message(peer=self._server, obj=obj)

    def _close(self) -> None:
        """Close the client"""
        state = self._state
        stream = self._serializer.dump(self._server)
        state.put(stream, ancillary=True)

        with self._lock:
            while hasattr(self, "_socket"):
                self._modify_selector(self._socket, selectors.EVENT_READ | selectors.EVENT_WRITE)
                self._notify_selector()
                self._lock.wait()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._state.get_queue.put(END_COMM)

        super()._close()
