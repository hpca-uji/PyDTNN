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
from typing import NamedTuple
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from collections import abc as col_abc
from concurrent.futures import Future, ThreadPoolExecutor


from pydtnn.utils import UUID_MAX, UUID_NIL, parse_bool
from pydtnn.utils.io_stream import Packer, Serializer, Stream, byteview


__all__ = (
    "CommunicatorOptions",
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


# type CommunicatorOptions = col_abc.Mapping[str, typing.Any]


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

    def __init__(self, merge_size: int = 4 * 1024 ** 2 - 1, efficient_size: int = 64 * 1024 ** 1 - 1) -> None:
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

    def get_write(self, b: col_abc.Buffer) -> int:
        """Write get buffer (merging chunks if plausible)"""
        size = self.get_buffer.write(b)

        if self.get_buffer.nchunks > 1 and size < self._merge_size:
            self._get_merge()

        return size

    def _get_merge(self) -> None:
        """Merge get buffer"""
        with Stream() as stream:

            while not self.get_buffer.empty() and stream.nbytes < len(self._merge_buffer):
                chunk = self.get_buffer.unwritechunk()

                if len(chunk) >= self._merge_size:
                    self.get_buffer.writechunk(chunk)
                    break

                stream.unreadchunk(chunk)

            if stream.nbytes >= self._merge_size:
                chunk = stream.read()
                self.get_buffer.writechunk(chunk)
            else:
                self.get_buffer.writechunks(stream.readchunks())

    def get_flush(self) -> col_abc.Iterable[Stream]:
        """Flush get buffer"""
        try:
            while True:
                yield self.get()
        except BlockingIOError:
            pass

    def _put_merge(self) -> None:
        """Merge put buffer"""
        merge_size = self.put_buffer.readinto(self._merge_buffer)
        self.put_buffer.unreadchunk(self._merge_buffer[:merge_size])

    def put_read(self, size: int = -1) -> memoryview:
        """Read put buffer (merging chunks if plausible)"""
        if self.put_buffer.nchunks > 1 and len(self.put_buffer.peekchunk()) < self._merge_size:
            self._put_merge()

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


class NetworkLocation(NamedTuple):
    """Network location"""
    host: str = "127.0.0.1"
    port: int = 50000

    def __str__(self):
        """Unified network location"""
        return f"{self.host}:{self.port}"


@dataclass(order=False, slots=True, frozen=True)
class ConnectionOptions:
    """Conncetion data options"""
    max_size: int
    merge_size: int
    efficient_size: int

    def __init__(self, max_size: int = 0, merge_size: int = 0, efficient_size: int = 0):
        """Inizialize connection options"""
        # NOTE: Frozen dataclasess must use object.__setattr__ during __init__
        object.__setattr__(self, "max_size", max_size if max_size else 4 * 1024 ** 2)
        object.__setattr__(self, "merge_size", merge_size if merge_size else self.max_size)
        object.__setattr__(self, "efficient_size", efficient_size if efficient_size else self.max_size // 64)


@dataclass(order=False, slots=True, frozen=True)
class CommunicatorOptions:
    """Comunicatior options"""
    netloc: NetworkLocation = NetworkLocation()
    workers: int = 1
    connection: ConnectionOptions = ConnectionOptions()


class Communicator[T](abc.ABC):
    """Base communicator implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Communicator initialization"""
        super().__init__()
        self._options = options

        self._id = uuid.uuid4()
        self._close_init = threading.Lock()
        self._close_done = threading.Event()

        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()

        self._serializer = Serializer()
        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"
        self._pool = ThreadPoolExecutor(max_workers=self._options.workers, thread_name_prefix=f"{thread_prefix}")

    def _new_state(self) -> ConnectionData:
        """Generate new connection state data"""
        return ConnectionData(merge_size=self._options.connection.merge_size, efficient_size=self._options.connection.efficient_size)

    def _session_ini(self, state: ConnectionData) -> Stream:
        """Send session ini message"""
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        return self._serializer.dump(self._id)

    def _session_fin(self, state: ConnectionData) -> Stream:
        """Send session fin message"""
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        return Stream()

    def _handle_session_ini(self, state: ConnectionData, stream: Stream) -> None:
        """Handle session initialize message"""
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

    def _handle_session_fin(self, state: ConnectionData, stream: Stream) -> None:
        """Handle session finalize message"""
        stream.close()
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"
        state.state &= ~ConnectionState.READABLE

    @abc.abstractmethod
    def _get_state(self, peer: uuid.UUID) -> ConnectionData:
        raise NotImplementedError()

    def _get(self, *peers: uuid.UUID) -> uuid.UUID:
        return self._get_event.get()

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        assert len(peers) == 0, "Server can not get from specific client"
        peer = self._get(*peers)

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._get_state(peer)
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        self._peer_cleanup(peer)

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=peer, obj=obj)

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        pass

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_init.locked()

    @abc.abstractmethod
    def put(self, obj: T, *peers: uuid.UUID) -> Future[None]:
        """Publish data to peer"""
        raise NotImplementedError()

    def _close(self) -> None:
        """Communicator finalizer"""
        self._pool.shutdown()

    def close(self) -> None:
        """Close the communicator"""
        if self._close_init.acquire(blocking=False):
            self._close()
            self._close_done.set()
        self._close_done.wait()

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
