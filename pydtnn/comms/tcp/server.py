"""TCP server"""

import uuid
import socket
import selectors
import threading
from queue import SimpleQueue
from concurrent.futures import Future

from bidict import bidict

from pydtnn.comms.tcp import Protocol
from pydtnn.utils.asynctools import chain_futures
from pydtnn.utils.io_stream import AncillaryStream
from pydtnn.comms import ResourceClosed, Message, ConnectionState


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = object()


class Server(Protocol):
    """TCP server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()

        # External
        self._get_event = SimpleQueue()

        # Internal
        self._peers = bidict[uuid.UUID, socket.socket]()
        self._state = dict[uuid.UUID, ConnectionState]()

        # TCP
        self._socket = socket.create_server((self._addr, self._port), reuse_port=True)
        self._selector.register(self._socket, selectors.EVENT_READ, self._new_connection)
        self._notify_selector()

    def _new_connection(self, sock: socket.socket, event) -> None:
        """Handle new incomming connections"""
        # NOTE: communication thead
        sock = self._socket.accept()[0]
        peer = uuid.uuid4()

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._max_message_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._max_message_size)
        sock.setblocking(False)

        with self._lock:
            self._peers[peer] = sock
            self._state[peer] = ConnectionState(buffer_size=self._max_message_size // 2)
            self._lock.notify_all()

        self._selector.register(sock, selectors.EVENT_READ, self._handle_connection)
        self._notify_selector()

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection states"""
        # NOTE: communication thead
        peer = self._peers.inverse[sock]
        state = self._state[peer]

        if state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ)

        if event & selectors.EVENT_READ:
            self._c2s(sock)

        if event & selectors.EVENT_WRITE:
            self._s2c(sock)

        if not state.put_empty():
            self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)

        if state.closed and state.put_empty():
            self._fin(sock)

        self._notify_selector()

    def _fin(self, sock: socket.socket) -> None:
        peer = self._peers.inverse[sock]
        state = self._state[peer]
        state.close()
        self._selector.unregister(sock)
        sock.close()

        # Remove peer
        with self._lock:
            del self._peers[peer]
            if self._state[peer].empty():
                del self._state[peer]
            self._lock.notify_all()

    def _c2s(self, sock: socket.socket) -> None:
        peer = self._peers.inverse[sock]
        state = self._state[peer]

        data = sock.recv(self._max_message_size)

        state.get_stream.write(data)

        while True:
            try:
                stream = state.get()
            except AncillaryStream as ancillary:
                with ancillary.stream as stream:
                    id = self._serializer.load(stream)

                # Client ID, INI
                if id != self._id:

                    # ACK
                    stream = self._serializer.dump(self._id)
                    state.put(stream, ancillary=True)

                    # New ID, move state from tmp ID
                    if id not in self._peers:
                        with self._lock:
                            self._state[id] = state = self._state.pop(peer)

                    # Change socket ID association
                    self._peers.inverse[sock] = peer = id

                # Server ID, FIN
                else:

                    # ACK
                    state.close()
                    stream = self._serializer.dump(peer)
                    state.put(stream, ancillary=True)

            except BlockingIOError:
                break
            else:
                state.get_queue.put(stream)
                peer = self._peers.inverse[sock]
                self._get_event.put(peer)

    def _s2c(self, sock: socket.socket) -> None:
        peer = self._peers.inverse[sock]
        state = self._state[peer]

        state.put_flush()
        if not state.put_stream.empty():
            pass
        else:
            return
        with state.put_read() as view:
            size = sock.send(view)
            if size < len(view):
                state.put_stream.unreadchunk(view[size:])

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        super().get(*peers)
        assert len(peers) == 0, "Server can not get from specific client"

        # Wait for a event
        peer = self._get_event.get()

        # Exit signaled
        if peer is self._id:
            raise ResourceClosed()

        state = self._state[peer]
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        # Remove finalized drained peer
        if peer not in self._peers and state.empty():
            with self._lock:
                if peer not in self._peers and state.empty():
                    del self._state[peer]

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=peer, obj=obj)

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        super().put(obj, *peers)

        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        futures = list[Future[None]]()
        errors = list[ResourceClosed]()
        with self._serializer.dump(obj) as stream:
            for peer in peers:
                try:
                    sock = self._peers[peer]
                    state = self._state[peer]
                except KeyError:
                    errors.append(ResourceClosed(peer))
                    continue
                try:
                    future = state.put(stream.copy())
                except ResourceClosed:
                    errors.append(ResourceClosed(peer))
                else:
                    self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)
                    futures.append(future)
        self._notify_selector()

        if errors:
            raise ExceptionGroup("Peer does not exist", errors)

        return chain_futures(futures)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        self._socket.close()

        # Unlock inflight external API
        for _ in range(threading.active_count()):
            self._get_event.put(self._id)

        super()._close()
