"""MPI communication"""

import enum
from dataclasses import dataclass


__all__ = (
    "Operation",
    "Request",
    "Response"
)


class Operation(enum.Enum):
    """Operation type"""
    BROADCAST = enum.auto()
    GATHER = enum.auto()
    REDUCE = enum.auto()


@dataclass(frozen=True, slots=True)
class Request:
    """Operation request"""
    rank: int
    size: int
    operation: Operation
    obj: ...


@dataclass(frozen=True, slots=True)
class Response:
    """Operation response"""
    obj: ...
