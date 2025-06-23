"""Communications package"""

# NOTE: Communication conventions:
# - syc: connection state (generic)
# - ini: connection start (identify)
# - fin: connection stop  (flush)
# - com: message exchange (generic)
# - c2s: message exchange (client -> server)
# - s2c: message exchange (server -> client)

# NOTE: Communication handshakes:
# Ini:
# - Server & client sends ID
# - Server & client wait for ID
# - Server create session or continues session
#
# Fin:
# - Server & client flushes message queue
# - Server & client sends empty message
# - Server & client wait for empty message

# NOTE: Communication persistency:
# Ini:
# - Must be done on first or changing connection
#
# Fin:
# - Must be done on session end (not connection)

# NOTE: Communication contract:
# Constructor
# - Never blocks
# - Only one communicator per ID
# - Reusing ID retain server queues
#
# Put
# - Never blocks
# - Communication will not modify object
# - Consumer must not modify object util future resolved
# - Resolved futures acknowledge peer reception
# - Cancelled futures indicates peer diconnected
#
# Get
# - Always block
# - Returns a message or raises ResouceClosed
# - Once closed it continues working until exhausted then it raises ResouceClosed
#
# Close
# - Always block
# - Server waits for peers to disconnect

# FIXME: Implement put future handling

# TODO: Implement client reconnection

# TODO: Change ResouceClosed for queue.Empty exception

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
from pathlib import Path
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from concurrent.futures import Future


from pydtnn.utils import UUID_NIL, parse_bool
from pydtnn.utils.io_stream import Packer, Serializer, Stream, byteview


__all__ = (
    "PROTOCOL",
    "SSL",
    "Protocol",
    "Message",
    "ResourceClosed",
    "Communicator",
    "ConnectionData",
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


class ConnectionState(enum.Flag):
    """Connection state"""
    READABLE = enum.auto()
    WRITABLE = enum.auto()


class ConnectionData:
    """Connection data"""

    def __init__(self, merge_size: int = 4 * 1024 ** 2 - 1, efficient_size: int = 64 * 1024 ** 2 - 1) -> None:
        """Initialize connection state"""
        self.peer = UUID_NIL
        self.state = ConnectionState(value=0)

        self._merge_buffer = byteview(bytearray(merge_size))
        self._merge_size = min(merge_size, efficient_size)
        self._packer = Packer()

        self.put_queue = SimpleQueue[Stream]()
        self.get_queue = SimpleQueue[Stream]()
        self.put_buffer = Stream()
        self.get_buffer = Stream()

    def get_empty(self) -> bool:
        """Is get connection flushed"""
        return self.get_queue.empty() and self.get_buffer.empty()

    def put_empty(self) -> bool:
        """Is put connection flushed"""
        return self.put_queue.empty() and self.put_buffer.empty()

    def get(self) -> Stream:
        """Unpack from get buffer"""
        return self._packer.unpack(self.get_buffer)

    def put(self, stream: Stream) -> Future[None]:
        """Push stream to put queue"""
        self.put_queue.put(stream)
        return Future[None]()

    def put_read(self, size: int = -1) -> memoryview:
        """Read put stream (merging chunks if plausible)"""
        if self.put_buffer.nchunks > 1 and len(self.put_buffer._chunks[0]) < self._merge_size:
            size = self.put_buffer.readinto(self._merge_buffer)
            self.put_buffer.unreadchunk(self._merge_buffer[:size])

        return self.put_buffer.read1(size)

    def put_flush(self) -> None:
        """Flush put queue"""
        while True:
            try:
                stream = self.put_queue.get_nowait()
            except Empty:
                break
            else:
                self._packer.pack(self.put_buffer, stream)


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
        self._serializer = Serializer()

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_lock.locked()

    @abc.abstractmethod
    def get(self, *peers: uuid.UUID) -> Message[T]:
        """Get data from peer"""
        raise NotImplementedError()

    @abc.abstractmethod
    def put(self, obj: T, *peers: uuid.UUID) -> Future[None]:
        """Publish data to peer"""
        raise NotImplementedError()

    def _close(self) -> None:
        """Communicator finalizer"""

    def close(self) -> None:
        """Close the communicator"""
        if self._close_lock.acquire(blocking=False):
            self._close()

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
# PROTOCOL
PROTOCOL: Protocol | None
if _env_protocol := os.environ.get("PYDTNN_COMM"):
    PROTOCOL = Protocol(_env_protocol)
else:
    PROTOCOL = None

# SSL
SSL = parse_bool(os.environ.get("PYDTNN_COMM_SSL"))

if _ssl_key := os.environ.get("PYDTNN_COMM_SSL_KEY"):
    SSL_KEY = Path(_ssl_key).resolve()
else:
    SSL_KEY = None

if _ssl_cert := os.environ.get("PYDTNN_COMM_SSL_CERT"):
    SSL_CERT = Path(_ssl_cert).resolve()
else:
    SSL_CERT = None

# Proxy
Server: type[Communicator]
Client: type[Communicator]


def __getattr__(key):
    """Proxy all attributes to implementation"""
    if not PROTOCOL:
        raise AttributeError(key)
    try:
        module = importlib.import_module(f"pydtnn.comms.{PROTOCOL}.{key.lower()}")
    except ModuleNotFoundError:
        raise AttributeError(key)
    return getattr(module, key)
