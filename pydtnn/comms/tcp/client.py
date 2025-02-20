"""TCP client"""

import uuid
import socket
import selectors
import threading
from queue import Empty, SimpleQueue

from pydtnn.comms import Message, ResourceClosed
from pydtnn.comms.tcp import Protocol
from pydtnn.comms.tcp.connection import Connection


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

        # TCP
        self._connection = Connection(socket.create_connection((self._addr, self._port)))
        self._selector.register(self._connection, selectors.EVENT_READ | selectors.EVENT_WRITE, self._handle_connection)
        self._syc()
        self._submit(self._handle_selector)

    def _handle_connection(self, connection: Connection, event) -> None:
        """Handle connection states"""
        if event & selectors.EVENT_READ:
            self._s2c()

        if event & selectors.EVENT_WRITE:
            self._c2s()

    def _syc(self) -> None:
        """Client connection startup"""
        data = self._serialize(self.id)
        self._connection.put(data)
        data = self._connection.get()
        self.server = self._deserialize(data)

    def _s2c(self) -> None:
        """Server to client communication"""
        if self.closed:
            return

        try:
            self._connection.recv()
        except ResourceClosed:
            return

        while True:
            try:
                data = self._connection.get_nowait()
            except Empty:
                break

            self._responses.put(data)

    def _c2s(self) -> None:
        """Client to server communication"""
        if self.closed:
            return

        while True:
            try:
                data = self._requests.get_nowait()
            except Empty:
                break

            self._connection.put_nowait(data)

        try:
            self._connection.send()
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
        return Message(peer=self.server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        super().close()
        self._connection.close()
        self._requests.put(END_COMM)
