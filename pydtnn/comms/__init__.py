"""Communications package"""

# NOTE: Commuications conventions:
# - syc: inizialize client (id exchange)
# - com: message exchange (duplex)
# - c2s: message exchange (client -> server)
# - s2c: message exchange (server -> client)
# - fin: finalize client (drain)

# NOTE: Expensive operations, such as serialization and blocking, are
# done at at the API consumers thread.

# FIXME: Allow multiple syc, if client exisits just swap connection.

# TODO: Rename syc to ini, bring insync with fin.

# FIXME: Put operations should try to send to as many peers as plausible
# before giving up on non-existent peers.

# FIXME: Get operations should error out on non-existent clients, only
# when no peers are specified it shoud block until some apear. Equally
# client should always recive from the explicit server UUID.

# FIXME: Close is not thread-safe.

# FIXME: Close should flush all buffers, as API consumers expect
# comunications to just-work, not to lose messages because it was closed.
# Therefore, fin messages shoud be responded with a

# TODO: Remove serialzization operations and accept only bytes.

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
import time
import math
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
    TCP = enum.auto()


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
    _backoff_initial_exponent = -10

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

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes) -> T:
        """Deserialize object from comunication"""
        return pickle.loads(data)

    def _new_backoff(self):
        """Exponential backoff blocker generator"""
        exponent = self._backoff_initial_exponent

        while True:
            max = yield
            backoff = 2 ** exponent

            if max <= 0.0:
                backoff = 0.0
                exponent = self._backoff_initial_exponent
            elif backoff >= max:
                backoff = max
                exponent = math.ceil(math.log2(max))
            else:
                exponent += 1

            time.sleep(backoff)

    @abc.abstractmethod
    def get(self, *peers: uuid.UUID) -> Message[T]:
        """Get data from peer"""

    @abc.abstractmethod
    def put(self, obj: T, *peers: uuid.UUID) -> None:
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
        module = importlib.import_module(f"pydtnn.comms.{PROTOCOL}.{key.lower()}")
    except ModuleNotFoundError:
        raise AttributeError(key)
    return getattr(module, key)
