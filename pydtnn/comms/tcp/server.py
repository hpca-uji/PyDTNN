"""TCP server"""

import ssl
import uuid
import socket
import selectors
import threading
from queue import SimpleQueue
from concurrent.futures import Future

from bidict import bidict

from pydtnn import comms
from pydtnn.comms.tcp import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_NIL, UUID_MAX
from pydtnn.utils.asynctools import merge_futures
from pydtnn.comms import ConnectionState, ResourceClosed, Message, ConnectionData


__all__ = (
    "Server",
)


class Server(Protocol):
    """TCP server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._peers = bidict[uuid.UUID, socket.socket]()
        self._state = dict[uuid.UUID, ConnectionData]()

        # TCP
        self._socket = socket.create_server((self._addr, self._port), reuse_port=True)

        if comms.SSL:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=comms.SSL_CERT)
            context.load_cert_chain(certfile=comms.SSL_CERT, keyfile=comms.SSL_KEY)
            self._socket = context.wrap_socket(self._socket, server_side=True, do_handshake_on_connect=True)

        self._selector.register(self._socket, selectors.EVENT_READ, self._new_connection)
        self._notify_selector()

    def _new_connection(self, sock: socket.socket, event) -> None:
        """Handle new incomming connections"""
        # NOTE: communication thead
        sock, _ = self._socket.accept()
        peer = uuid.uuid4()  # temporary ID

        sock.setblocking(False)

        with self._lock:
            self._peers[peer] = sock
            self._state[peer] = ConnectionData()
            self._lock.notify_all()

        self._selector.register(sock, selectors.EVENT_READ, self._handle_connection)
        self._notify_selector()

        # ACK
        self._session_ini(peer)

    def _handle_connection(self, sock: socket.socket, event) -> None:
        """Handle connection events"""
        # NOTE: communication thead
        peer = self._peers.inverse[sock]
        state = self._state[peer]

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
            self._fin(sock)

    def _session_ini(self, peer: uuid.UUID) -> None:
        """Send session ini message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        self.put(self._id, peer)

    def _session_fin(self, peer: uuid.UUID) -> None:
        """Send session fin message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        self._put(Stream(), peer)

    def _handle_session_ini(self, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session initialize message"""
        sock = self._peers[peer]
        state = self._state[peer]
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

        # New ID, move state from tmp ID
        if id not in self._peers:
            with self._lock:
                self._state[id] = state = self._state.pop(peer)

        # Change socket ID association
        with self._lock:
            self._peers.inverse[sock] = id

    def _handle_session_fin(self, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session finalize message"""
        state = self._state[peer]
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"
        stream.close()
        state.state &= ~ConnectionState.READABLE

    def _get_flush(self, peer: uuid.UUID):
        state = self._state[peer]

        while True:
            try:
                stream = state.get()
            except BlockingIOError:
                break

            if stream.empty():
                self._handle_session_fin(peer, stream)
                self._session_fin(peer)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(peer, stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _c2s(self, sock: socket.socket) -> None:
        """Client to server communication"""
        peer = self._peers.inverse[sock]
        state = self._state[peer]

        while True:
            try:
                data = sock.recv(self._max_payload_size)
            except (BlockingIOError, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                break

            if not data:
                assert not state.state and state.put_queue.empty(), "Lost connection unexpectedly"
                return

            state.get_buffer.write(data)
            self._get_flush(peer)
            peer = state.peer

    def _s2c(self, sock: socket.socket) -> None:
        """Server to client communication"""
        peer = self._peers.inverse[sock]
        state = self._state[peer]

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
        peer = self._peers.inverse[sock]
        self._selector.unregister(sock)
        sock.close()

        # Remove peer
        with self._lock:
            del self._peers[peer]

            # TODO: reuse peer_cleanup
            if self._state[peer].get_empty():
                del self._state[peer]

            self._lock.notify_all()

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        """Remove finalized drained peer"""
        state = self._state[peer]

        if peer not in self._peers and state.get_empty():
            with self._lock:
                if peer not in self._peers and state.get_empty():
                    del self._state[peer]

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        assert len(peers) == 0, "Server can not get from specific client"
        peer = self._get_event.get()

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state[peer]
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        self._peer_cleanup(peer)

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=peer, obj=obj)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        try:
            sock = self._peers[peer]
            state = self._state[peer]
            future = state.put(stream)
        except (KeyError, ResourceClosed):
            raise ResourceClosed(peer)
        self._modify_selector(sock, selectors.EVENT_READ | selectors.EVENT_WRITE)
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        futures = list[Future[None]]()
        errors = list[ResourceClosed]()
        with self._serializer.dump(obj) as stream:
            for peer in peers:
                try:
                    future = self._put(stream.copy(), peer)
                except ResourceClosed as exc:
                    errors.append(exc)
                    continue
                else:
                    futures.append(future)
        self._notify_selector()

        if errors:
            raise ExceptionGroup("Peer does not exist", errors)

        return merge_futures(futures)

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
