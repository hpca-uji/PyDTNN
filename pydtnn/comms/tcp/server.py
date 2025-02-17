"""TCP server"""

import uuid
import socket
import selectors
import threading
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.tcp import Connection, Protocol


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = b""


class Server(Protocol):
    """TCP server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Lock()
        self._peers = bidict[uuid.UUID, str]()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._request_count = threading.Semaphore(value=0)
        self._response_count = threading.Semaphore(value=0)
        self._requests = dict[uuid.UUID, SimpleQueue[bytes]]()
        self._responses = dict[uuid.UUID, SimpleQueue[bytes]]()

        # TCP
        self._selector = selectors.DefaultSelector()
        self._server = Connection(socket.create_server((self._addr, self._port), reuse_port=True))
        self._selector.register(self._server, selectors.EVENT_READ, self._new_connection)
        self._submit(self._serve_forever)

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _serve_forever(self):
        while not self.closed:
            for key, mask in self._selector.select(self._poll_interval):
                callback = key.data
                callback(key.fileobj, mask)

    def _new_connection(self, connection: Connection, event) -> None:
        connection = Connection(connection._socket.accept()[0])
        self._syc(connection)

    def _handle_connection(self, connection: Connection, event) -> None:
        if event & selectors.EVENT_READ:
            self._c2s(connection)

        if event & selectors.EVENT_WRITE:
            self._s2c(connection)

    def _syc(self, connection: Connection) -> None:
        tcp_peer = connection.peer()
        data = connection.get()
        peer = self._deserialize(data)
        data = self._serialize(self.id)
        connection.put(data)

        with self._lock:
            self._peers[peer] = tcp_peer
            self._requests[peer] = SimpleQueue()
            self._responses[peer] = SimpleQueue()

        self._selector.register(connection, selectors.EVENT_READ | selectors.EVENT_WRITE, self._handle_connection)

    def _c2s(self, connection: Connection) -> None:
        tcp_peer = connection.peer()
        try:
            peer = self._peers.inverse[tcp_peer]
        except KeyError:
            return
        queue = self._requests[peer]

        try:
            connection.recv()
        except ResourceClosed:
            self._fin(connection)
            return

        while True:
            try:
                data = connection.get_nowait()
            except Empty:
                break

            queue.put(data)
            self._request_count.release()

    def _s2c(self, connection: Connection) -> None:
        tcp_peer = connection.peer()
        try:
            peer = self._peers.inverse[tcp_peer]
        except KeyError:
            return
        queue = self._responses[peer]

        while True:
            try:
                data = queue.get_nowait()
            except Empty:
                break

            connection.put_nowait(data)

        try:
            connection.send()
        except ResourceClosed:
            self._fin(connection)
            return

    def _fin(self, connection: Connection) -> None:
        tcp_peer = connection.peer()
        peer = self._peers.inverse[tcp_peer]

        self._selector.unregister(connection)

        with self._lock:
            del self._peers[peer]
            del self._requests[peer]
            del self._responses[peer]

        connection.close()

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        super().get(*peers)

        while True:
            # Wait for a request
            # FIXME: if peers is defined and other peers have messages this is a busy-wait
            self._request_count.acquire()

            # Acquire peers
            if peers:
                _peers = peers
            else:
                with self._lock:
                    _peers = tuple(self._requests)

            # Search for a request
            for peer in _peers:
                try:
                    data = self._requests[peer].get_nowait()
                except (KeyError, Empty):
                    continue
                else:
                    break

            # Request not found, revert notification and retry
            else:
                self._request_count.release()
                continue

            # Request found, continue
            break

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=peer, obj=obj)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to clients"""
        super().put(obj, *peers)

        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        data = self._serialize(obj)

        for peer in peers:
            self._responses[peer].put(data)

    def close(self) -> None:
        """Close the server"""
        if self.closed:
            return
        super().close()
        self._pool.shutdown()
        self._server.close()
        with self._lock:
            for queue in self._requests.values():
                queue.put(END_COMM)
            for key in self._selector.get_map().values():
                key.fileobj.close()  # type: ignore
