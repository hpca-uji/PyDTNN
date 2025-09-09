"""Shared client-server MPI code"""

# NOTE: Communications are lazily initialized to prevent module imports execution

# NOTE: Dataclasses must not use functools.cache, as it would add data to serialization

# FIXME: Check sends and recives are sent or as much as posible even on error,
# or if this is handled completely by the comunication layer.

# TODO: Check which lazy inizializations are actually required now, if not necessary
# inizialize eagerly and avoid complex handeling.

import os
import abc
import enum
import uuid
import typing
import operator
import functools
import dataclasses
from dataclasses import dataclass
from collections import abc as coll_abc
from traceback import TracebackException

from intbitset import intbitset


__all__ = (
    "Rank",
    "get_init",
    "get_addr",
    "get_port",
    "get_size",
    "get_rank",
    "CommmunicationGroup",
    "ReduceOperation",
    "StateRequest",
    "RankInit",
    "RankFinalize",
    "OperationRequest",
    "OperationResponse",
    "BroadcastContext",
    "AllGatherContext",
    "AllReduceContext",
)


type Rank = int


@functools.cache
def get_init() -> bool:
    """Should service auto initialize"""
    return bool(
        not os.environ.get("PYDTNN_MPI_ADDR")
    )


@functools.cache
def get_addr() -> str:
    """Service address"""
    return (
        os.environ.get("PYDTNN_MPI_ADDR")
        or "127.0.0.1"
    )


@functools.cache
def get_port() -> int:
    """Service port"""
    return int(
        os.environ.get("PYDTNN_MPI_PORT")
        or 61642
    )


@functools.cache
def get_size() -> int:
    """Communication size"""
    return int(
        os.environ.get("PYDTNN_MPI_SIZE")
        or os.environ.get("OMPI_COMM_WORLD_SIZE")
        or os.environ.get("PMI_SIZE")
        or os.environ.get("SLUM_NPROCS")
        or 1
    )


@functools.cache
def get_rank() -> Rank:
    """Communication identifier"""
    return int(
        os.environ.get("PYDTNN_MPI_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("SLUM_PROCID")
        or 0
    )


class RemoteException(RuntimeError):
    """Remote exception (serialization safe)"""

    @classmethod
    def from_exception(cls, exc: Exception):
        """Create message from exception"""
        traceback = TracebackException.from_exception(exc)
        message = "".join(traceback.format())
        return cls(message)


@dataclass(slots=True, frozen=True)
class CommmunicationGroup:
    """Communication group"""
    src: frozenset[Rank]
    dst: frozenset[Rank]

    def __init__(self, src: coll_abc.Iterable[Rank], dst: coll_abc.Iterable[Rank]) -> None:
        """Inizialize communication group"""
        # NOTE: Frozen dataclasess must use object.__setattr__ during __init__
        object.__setattr__(self, "src", intbitset(src))  # type: ignore
        object.__setattr__(self, "dst", intbitset(dst))  # type: ignore

    @property
    def root(self) -> Rank:
        """Root rank of communication group"""
        return min(self.src)


class ReduceOperation(enum.Enum):
    """Reduction operation."""
    SUM = enum.auto()


@dataclass(slots=True, frozen=True)
class StateRequest:
    """Status request"""


@dataclass(slots=True, frozen=True)
class RankInit(StateRequest):
    """Initialize rank"""
    rank: Rank


@dataclass(slots=True, frozen=True)
class RankFinalize(StateRequest):
    """Finalize rank"""


@dataclass(slots=True, frozen=True)
class StateResponse:
    """Status response"""
    size: int


@dataclass(slots=True, frozen=True)
class OperationContext[T](abc.ABC):
    """Operation context"""

    @abc.abstractmethod
    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(self, objs: coll_abc.Mapping[Rank, T]) -> typing.Any:
        """Apply operation over objects"""
        raise NotImplementedError()


@dataclass(slots=True, frozen=True)
class BroadcastContext[T](OperationContext[T]):
    """Broadcast operation"""
    root: Rank = 0

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup([self.root], range(size))

    def apply(self, objs: coll_abc.Mapping[Rank, T]) -> T:
        """Apply operation over objects"""
        return objs[self.root]


@dataclass(slots=True, frozen=True)
class AllGatherContext[T](OperationContext[T]):
    """All gather operation"""

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), range(size))

    def apply(self, objs: coll_abc.Mapping[Rank, T]) -> list[T]:
        """Apply operation over objects"""
        objs = dict(sorted(objs.items(), key=lambda item: item[0]))
        return list(objs.values())


@dataclass(slots=True, frozen=True)
class AllReduceContext[T](OperationContext[T]):
    """All reduce operation"""
    op: ReduceOperation = ReduceOperation.SUM

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), range(size))

    def apply(self, objs: coll_abc.Mapping[Rank, T]) -> T:
        """Apply operation over objects"""
        match self.op:
            case ReduceOperation.SUM:
                return functools.reduce(operator.add, objs.values())  # type: ignore (T should be addable)
            case _:
                raise NotImplementedError(f"op with not {self.op}")


@dataclass(slots=True, frozen=True)
class AllPhasedReduceContext(AllReduceContext):
    """All phased reduce operation"""

    def comm(self, rank: Rank, size: int, phase: int = 0, group: int = 2) -> CommmunicationGroup:
        """Compute operation's communication group"""
        phase_size = (group ** (phase + 1))
        start = (rank // phase_size) * phase_size
        step = (group ** (phase))
        stop = min(start + phase_size, size)
        return CommmunicationGroup(range(start, stop, step), range(start, stop))


@dataclass(slots=True, frozen=True)
class OperationRequest:
    """Operation request"""
    comm: CommmunicationGroup
    context: OperationContext | None = None
    obj: typing.Any | None = None
    id: uuid.UUID = dataclasses.field(init=False, default_factory=uuid.uuid4)


@dataclass(slots=True, frozen=True)
class OperationResponse:
    """Operation response"""
    id: uuid.UUID
    obj: typing.Any
