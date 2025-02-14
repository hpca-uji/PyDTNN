"""MPI communication"""

# NOTE: Module considerations
#
# Metadata is lazily initialized to prevent module imports execution.
#
# Dataclasses use explict super calls due to a bug:
# https://github.com/python/cpython/issues/90562

import os
import abc
import enum
import dataclasses
from collections import abc as coll_abc
from dataclasses import dataclass, InitVar

from intbitset import intbitset


__all__ = (
    "Rank",
    "RankGroup",
    "get_size",
    "get_rank",
    "get_addr",
    "get_port",
    "CommmunicationGroup",
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


type Rank = int
type RankGroup = intbitset


def get_size() -> int:
    """Communication size"""
    # NOTE: Lazily initialized, prevent module imports execution
    return int(os.environ["OMPI_COMM_WORLD_SIZE"])


def get_rank() -> Rank:
    """Communication identifier"""
    # NOTE: Lazily initialized, prevent module imports execution
    return int(os.environ["OMPI_COMM_WORLD_RANK"])


def get_addr() -> str:
    """Service address"""
    return os.environ.get("PYDTNN_MPI_ADDR") or "127.0.0.1"


def get_port() -> int:
    """Service port"""
    return int(os.environ.get("PYDTNN_MPI_PORT") or 50000)


@dataclass(slots=True, frozen=True)
class CommmunicationGroup:
    """Communication group"""
    src: RankGroup
    dst: RankGroup


class ReduceOperation(enum.Enum):
    """Reduction operation."""
    SUM = enum.auto()


@dataclass(slots=True, frozen=True)
class StateRequest:
    """Generic state operation"""


@dataclass(slots=True, frozen=True)
class InitRequest(StateRequest):
    """Initialize request"""
    rank: Rank
    size: int


@dataclass(slots=True, frozen=True)
class FinalizeRequest(StateRequest):
    """Terminate request"""


@dataclass(slots=True, frozen=True)
class OperationRequest(abc.ABC):
    """Operation request"""
    comm: CommmunicationGroup = dataclasses.field(init=False)
    rank: InitVar[Rank]
    size: InitVar[int]

    @abc.abstractmethod
    def __post_init__(self, src: coll_abc.Iterable[Rank], dst: coll_abc.Iterable[Rank]) -> None:  # type: ignore
        # NOTE: abc.abstractmethod, dataclass.__post_init__ combination not inferred by typecheker
        """Inizialize communication group"""
        comm = CommmunicationGroup(
            src=intbitset(src),
            dst=intbitset(dst),
        )

        # NOTE: Frozen dataclasess must use object.__setattr__ during __init__
        object.__setattr__(self, "comm", comm)


@dataclass(slots=True, frozen=True)
class BroadcastRequest[T](OperationRequest):
    """Broadcast request"""
    obj: T
    root: Rank = 0

    def __post_init__(self, rank: Rank, size: int) -> None:
        """Compute operation's communication group"""
        # NOTE: Explict super call due to bug
        super(BroadcastRequest, self).__post_init__([self.root], range(size))


@dataclass(slots=True, frozen=True)
class AllGatherRequest[T](OperationRequest):
    """All gather request"""
    obj: T

    def __post_init__(self, rank: Rank, size: int) -> None:
        """Compute operation's communication group"""
        # NOTE: Explict super call due to bug
        super(AllGatherRequest, self).__post_init__(range(size), range(size))


@dataclass(slots=True, frozen=True)
class AllReduceRequest[T](OperationRequest):
    """All reduce request"""
    obj: T
    op: ReduceOperation = ReduceOperation.SUM

    def __post_init__(self, rank: Rank, size: int) -> None:
        """Compute operation's communication group"""
        # NOTE: Explict super call due to bug
        super(AllReduceRequest, self).__post_init__(range(size), range(size))


@dataclass(slots=True, frozen=True)
class AllPhasedReduceRequest[T](OperationRequest):
    """All phased reduce request"""
    obj: T
    op: ReduceOperation = ReduceOperation.SUM
    phase: dataclasses.InitVar[int] = 0
    group: dataclasses.InitVar[int] = 2

    def __post_init__(self, rank: Rank, size: int, phase: int, group: int) -> None:
        """Compute operation's communication group"""
        phase_size = (group ** (phase + 1))
        start = (rank // phase_size) * phase_size
        step = (group ** (phase))
        stop = min(start + phase_size, size)

        # NOTE: Explict super call due to bug
        super(AllPhasedReduceRequest, self).__post_init__(range(start, stop, step), range(start, stop))


@dataclass(slots=True, frozen=True)
class OperationResponse[T]:
    """Operation response"""
    dst: RankGroup
    obj: T
