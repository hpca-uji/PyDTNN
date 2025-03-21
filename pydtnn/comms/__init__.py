"""Communications package"""

# NOTE: Communication conventions:
# - syc: connection state (generic)
# - ini: connection start (id exchange)
# - fin: connection stop  (flush)
# - com: message exchange (duplex)
# - c2s: message exchange (client -> server)
# - s2c: message exchange (server -> client)

# NOTE: Communication handshakes:
# Ini:
# - Client sends client ID
# - Server create session or continues session
# - Server responds server ID
#
# Fin:
# - Client flushes client queue
# - Client sends Server ID
# - Server flushes server queue
# - Server sends Client ID

# NOTE: Communication contract:
# Constructor
# - May block
# - Reusing ID retain server queues
#
# Put
# - Never blocks
# - Communication will not modify the object
# - Consumer must not modify the object after*
# - Guarantees peer reception or raises ResouceClosed
# - Once closed it always raises ResouceClosed
#
# Get
# - Always block
# - Returns a message or raises ResouceClosed
# - Once closed it continues working until exhausted then it raises ResouceClosed
#
# Close
# - May block
# - Server waits for peers to disconnect

# TODO: Implement two-way connection expiration and keep-alives. There
# is no reliable way to track connection drops between communication
# implementations. Most of them end up with memory leaks. If desired
# expiration periods could be long and client reconnections could be
# allowed, enabling MQTT-like reliability without the cost.

# TODO: Review Apache Kafka communication

import os
import abc
import uuid
import enum
import importlib
import threading
from dataclasses import dataclass
from queue import Empty, SimpleQueue

from pydtnn.utils.io_stream import PackerStream, StreamSerializer, Stream


__all__ = (
    "PROTOCOL",
    "Protocol",
    "Message",
    "ResourceClosed",
    "Communicator",
    "ConnectionState",
    "Server",
    "Client"
)


class Protocol(enum.StrEnum):
    """Comunication protocol"""
    GRPC = enum.auto()
    MQTT = enum.auto()
    TCP = enum.auto()


@dataclass(slots=True, frozen=True)
class Message[T]:
    """Message object"""
    peer: uuid.UUID
    obj: T


class ConnectionState:
    """Connection state data"""
    def __init__(self, buffer_size: int = 16 * 1024 ** 2 - 1) -> None:
        self.closed = False
        self._buffer = memoryview(bytearray(buffer_size))
        self.put_queue = SimpleQueue[Stream]()
        self.get_queue = SimpleQueue[Stream]()
        self.put_stream = PackerStream()
        self.get_stream = PackerStream()

    def close(self) -> None:
        # CALLED WHEN PEER INDICATES END
        self.closed = True

    def get_empty(self) -> bool:
        return self.get_queue.empty() and self.get_stream.empty()

    def put_empty(self) -> bool:
        return self.put_queue.empty() and self.put_stream.empty()

    def empty(self) -> bool:
        return self.get_empty() and self.put_empty()

    def put_flush(self) -> None:
        """Flush put queue"""
        while True:
            try:
                stream = self.put_queue.get_nowait()
            except Empty:
                break
            else:
                self.put_stream.pack(stream)

    def put_read(self) -> memoryview:
        """Read put stream"""
        view = self.put_stream.read1(len(self._buffer))

        # View if large or end chunks
        if len(view) == len(self._buffer) or self.put_stream.empty():
            return view

        # Buffer if multiple small chunks
        else:
            self.put_stream.unreadchunk(view)
            size = self.put_stream.readinto(self._buffer)
            return self._buffer[:size]


class ResourceClosed(RuntimeError):
    """Resource closed"""


class Communicator[T](abc.ABC):
    """Base communicator implementation"""

    def __init__(self, addr: str, port: int) -> None:
        """Communicator initialization"""
        super().__init__()
        self._id = uuid.uuid4()
        self._addr = addr
        self._port = port
        self._close_lock = threading.Lock()
        self._serializer = StreamSerializer()

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_lock.locked()

    @abc.abstractmethod
    def get(self, *peers: uuid.UUID) -> Message[T]:
        """Get data from peer"""

    @abc.abstractmethod
    def put(self, obj: T, *peers: uuid.UUID) -> None:
        """Publish data to peer"""
        if self._closed:
            raise ResourceClosed()

    def _close(self) -> None:
        """Communicator finalizer"""

    def close(self) -> None:
        """Close the communicator"""
        if self._close_lock.acquire(blocking=False):
            self._close()

    def __repr__(self) -> str:
        """Representation of instnace"""
        return f"<{self.__class__.__name__} id={self._id}>"

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass


# Exports
Server: type[Communicator]
Client: type[Communicator]

PROTOCOL: Protocol | None
if _env_protocol := os.environ.get("PYDTNN_COMM"):
    PROTOCOL = Protocol(_env_protocol)
else:
    PROTOCOL = None


def __getattr__(key):
    """Proxy all attributes to implementation"""
    if not PROTOCOL:
        raise AttributeError(key)
    try:
        module = importlib.import_module(f"pydtnn.comms.{PROTOCOL}.{key.lower()}")
    except ModuleNotFoundError:
        raise AttributeError(key)
    return getattr(module, key)
