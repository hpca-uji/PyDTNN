"""Communications package"""

import os
import abc
import enum
import pickle
import importlib
import functools


__all__ = (
    "PROTOCOL",
    "Protocol",
    "Communication",
    "Server",
    "Client"
)


# Modules
class Protocol(enum.StrEnum):
    """Comunication protocol"""
    GRPC = enum.auto()
    MQTT = enum.auto()


class Communication(abc.ABC):
    """Base communication implementation"""
    _pickle_protocol = 5
    _protocol_port = 50000

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

    @functools.cached_property
    def _addr(self) -> str:
        """Service address"""
        return os.environ.get("PYDTNN_COMM_ADDR", "localhost")

    @functools.cached_property
    def _port(self) -> int:
        """Service port"""
        return int(os.environ.get("PYDTNN_COMM_PORT", f"{self._protocol_port}"))

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object from comunication"""
        return pickle.loads(data)

    @abc.abstractmethod
    def get(self):
        """Get data from peer"""
        raise NotImplementedError()

    @abc.abstractmethod
    def put(self, obj) -> None:
        """Publish data to peer"""
        raise NotImplementedError()

    def close(self) -> None:
        """Close the connection"""

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
