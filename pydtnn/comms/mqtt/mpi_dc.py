"""Communication dataclasses"""

import enum
from dataclasses import dataclass


__all__ = (
    "CommunicationError",
    "UnavailableError",
    "Op",
    "SendRequest",
    "SendResponse",
    "RecvRequest",
    "RecvResponse",
)


class CommunicationError(RuntimeError):
    """Communication protocol error"""


class UnavailableError(RuntimeError):
    """Opertation not ready yet"""


@dataclass
class SteamEnd:
    ...


class Op(enum.Enum):
    BCAST = enum.auto()
    ALLGATHER = enum.auto()
    ALLREDUCE = enum.auto()


@dataclass
class SendRequest:
    rank: int
    data: bytes


@dataclass
class SendResponse:
    ...


@dataclass
class RecvRequest:
    rank: int
    size: int
    op: Op


@dataclass
class RecvResponse:
    data: bytes
