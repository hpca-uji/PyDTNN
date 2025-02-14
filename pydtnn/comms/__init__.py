"""Communications package"""

# NOTE: Implement TCP communication
# NOTE: Review Apache Kafka communication

import os
import abc
import uuid
import enum
import pickle
import importlib
from dataclasses import dataclass


__all__ = (
    "PROTOCOL",
    "Protocol",
    "Message",
    "ResourceClosed",
    "Communication",
    "Server",
    "Client"
)


class Protocol(enum.StrEnum):
    """Comunication protocol"""
    GRPC = enum.auto()
    MQTT = enum.auto()


@dataclass(slots=True)
class Message[T]:
    """Message object"""
    peer: uuid.UUID
    obj: T


class ResourceClosed(RuntimeError):
    """Resource closed"""


class Communication[T](abc.ABC):
    """Base communication implementation"""
    _pickle_protocol = 5

    def __init__(self, addr: str, port: int) -> None:
        """Communication initialization"""
        super().__init__()
        self.id = uuid.uuid4()
        self._addr = addr
        self._port = port
        self.closed = False

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.close()

    @property
    def _netloc(self):
        """Service network location (address + port)"""
        return f"{self._addr}:{self._port}"

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object from comunication"""
        return pickle.loads(data)

    @abc.abstractmethod
    def get(self, *peers: uuid.UUID) -> Message[T]:
        """Get data from peer"""
        if self.closed:
            raise ResourceClosed()

    @abc.abstractmethod
    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to peer"""
        if self.closed:
            raise ResourceClosed()

    def close(self) -> None:
        """Close the connection"""
        if self.closed:
            return
        self.closed = True

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass


# Exports
Server: type[Communication]
Client: type[Communication]

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
        module = importlib.import_module(f"pydtnn.comms.{PROTOCOL}.comm")
    except ModuleNotFoundError:
        raise AttributeError(key)
    return getattr(module, key)
