"""MPI communication"""

# NOTE: Module considerations
#
# Metadata is lazily initialized to prevent module imports execution

import os
import abc
import enum
import functools
from dataclasses import dataclass


__all__ = (
    "Rank",
    "get_size",
    "get_rank",
    "ReduceOperation",
    "StateRequest",
    "InitRequest",
    "FinalizeRequest",
    "OperationRequest",
    "BroadcastRequest",
    "AllGatherRequest",
    "AllReduceRequest",
    "OperationResponse"
)


Rank = int


@functools.cache
def get_size() -> int:
    """Communication size"""
    # NOTE: Lazily initialized, prevent module imports execution
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])


@functools.cache
def get_rank() -> Rank:
    """Communication identifier"""
    # NOTE: Lazily initialized, prevent module imports execution
    return int(os.environ["OMPI_COMM_WORLD_RANK"])


class ReduceOperation(enum.Enum):
    """Reduction operation."""
    SUM = enum.auto()


@dataclass(slots=True, frozen=True)
class StateRequest:
    """Generic state operation"""


@dataclass(slots=True, frozen=True)
class InitRequest(StateRequest):
    """Initialize request"""
    rank: int
    size: int


@dataclass(slots=True, frozen=True)
class FinalizeRequest(StateRequest):
    """Terminate request"""


@dataclass(slots=True, frozen=True)
class OperationRequest(abc.ABC):
    """Operation request"""

    @abc.abstractmethod
    def request_requirements(self, size: int) -> frozenset[Rank]:
        """Get request rank requirements"""
        raise NotImplementedError()

    @abc.abstractmethod
    def response_requirements(self, size: int) -> frozenset[Rank]:
        """Get response rank requirements"""
        raise NotImplementedError()


@dataclass(slots=True, frozen=True)
class BroadcastRequest[T](OperationRequest):
    """Broadcast request"""
    obj: T
    root: Rank = 0

    def request_requirements(self, size: int) -> frozenset[Rank]:
        """Get request rank requirements"""
        return frozenset([self.root])

    def response_requirements(self, size: int) -> frozenset[Rank]:
        """Get response rank requirements"""
        return frozenset(range(size))


@dataclass(slots=True, frozen=True)
class AllGatherRequest[T](OperationRequest):
    """All gather request"""
    obj: T

    def request_requirements(self, size: int) -> frozenset[Rank]:
        """Get request rank requirements"""
        return frozenset(range(size))

    def response_requirements(self, size: int) -> frozenset[Rank]:
        """Get response rank requirements"""
        return frozenset(range(size))


@dataclass(slots=True, frozen=True)
class AllReduceRequest[T](OperationRequest):
    """All reduce request"""
    obj: T
    op: ReduceOperation = ReduceOperation.SUM

    def request_requirements(self, size: int) -> frozenset[Rank]:
        """Get request rank requirements"""
        return frozenset(range(size))

    def response_requirements(self, size: int) -> frozenset[Rank]:
        """Get response rank requirements"""
        return frozenset(range(size))


@dataclass(slots=True, frozen=True)
class OperationResponse[T]:
    """Operation response"""
    group: frozenset[Rank]
    obj: T
