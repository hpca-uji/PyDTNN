"""TCP client"""

import uuid
import socket
import selectors
import threading
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from pydtnn.comms import Message, ResourceClosed
from pydtnn.comms.tcp import Connection, Protocol


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = b""


class Client(Protocol):
    """TCP client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Lock()
        self._requests = SimpleQueue[bytes]()
        self._responses = SimpleQueue[bytes]()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._selector = selectors.DefaultSelector()

        # TCP
        self._connection = Connection(socket.create_connection((self._addr, self._port)))
        data = self._serialize(self.id)
        self._connection.put(data)
        data = self._connection.get()
        self._server = self._deserialize(data)

        self._selector.register(self._connection, selectors.EVENT_READ | selectors.EVENT_WRITE, self._handle_connection)
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

    def _handle_connection(self, connection: Connection, event) -> None:
        if event & selectors.EVENT_READ:
            self._s2c(connection)

        if event & selectors.EVENT_WRITE:
            self._c2s(connection)

    def _s2c(self, connection: Connection) -> None:
        if self.closed:
            return

        try:
            connection.recv()
        except ResourceClosed:
            return

        while True:
            try:
                data = connection.get_nowait()
            except Empty:
                break

            self._responses.put(data)

    def _c2s(self, connection: Connection) -> None:
        if self.closed:
            return

        while True:
            try:
                data = self._requests.get_nowait()
            except Empty:
                break

            connection.put_nowait(data)

        try:
            connection.send()
        except ResourceClosed:
            return

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        data = self._serialize(obj)
        self._requests.put(data)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get server data"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"
        data = self._responses.get()

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=self._server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        super().close()
        self._pool.shutdown()
        self._connection.close()
        self._requests.put(END_COMM)
