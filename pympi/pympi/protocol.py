"""Shared client-server MPI code"""

# NOTE: Communications are lazily initialized to prevent module imports execution

# NOTE: Dataclasses must not use functools.cache, as it would add data to serialization

# FIXME: Check sends and recives are sent or as much as posible even on error,
# or if this is handled completely by the comunication layer.

# TODO: Check which lazy inizializations are actually required now, if not necessary
# inizialize eagerly and avoid complex handeling.

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
    "Tag",
    "Rank",
    "SERIALIZABLE",
    "CommmunicationGroup",
    "ReduceOperation",
    "StateRequest",
    "StateResponse",
    "RankInit",
    "RankFinalize",
    "OperationRequest",
    "OperationResponse",
    "BroadcastContext",
    "AllGatherContext",
    "AllReduceContext",
    "GatherContext",
    "ReduceContext",
    "ScatterContext",
    "AllToAllContext",
    "SendRecvContext"
)

SERIALIZABLE = (
    "uuid.UUID",
    "intbitset_helper._",
    *(f"{__name__}.{name}" for name in __all__)
)


type Rank = int
type Tag = int


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
    MAX = enum.auto()
    MIN = enum.auto()
    SUM = enum.auto()
    PROD = enum.auto()
    LAND = enum.auto()
    BAND = enum.auto()
    LOR = enum.auto()
    BOR = enum.auto()
    LXOR = enum.auto()
    BXOR = enum.auto()
    MINLOC = enum.auto()
    MAXLOC = enum.auto()


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
    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, typing.Any]:
        """Apply operation over objects"""
        raise NotImplementedError()


@dataclass(slots=True, frozen=True)
class BroadcastContext[T](OperationContext[T]):
    """Broadcast operation"""
    root: Rank = 0

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup([self.root], range(size))

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, T]:
        """Apply operation over objects"""
        return dict.fromkeys(dst, src[self.root])


@dataclass(slots=True, frozen=True)
class AllGatherContext[T](OperationContext[T]):
    """All gather operation"""

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), range(size))

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, coll_abc.Sequence[T]]:
        """Apply operation over objects"""
        src = dict(sorted(src.items(), key=lambda item: item[0]))
        return dict.fromkeys(dst, list(src.values()))


@dataclass(slots=True, frozen=True)
class AllReduceContext[T](OperationContext[T]):
    """All reduce operation"""
    op: ReduceOperation = ReduceOperation.SUM

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), range(size))

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, T]:
        """Apply operation over objects"""
        match self.op:
            case ReduceOperation.MAX:
                result = max(src.values())  # type: ignore
            case ReduceOperation.MIN:
                result = min(src.values())  # type: ignore
            case ReduceOperation.SUM:
                result = functools.reduce(operator.add, src.values())
            case ReduceOperation.PROD:
                result = functools.reduce(operator.mul, src.values())
            case ReduceOperation.LAND:
                result = all(src.values())
            case ReduceOperation.BAND:
                result = functools.reduce(operator.and_, src.values())
            case ReduceOperation.LOR:
                result = any(src.values())
            case ReduceOperation.BOR:
                result = functools.reduce(operator.or_, src.values())
            case ReduceOperation.LXOR:
                result = functools.reduce(lambda a, b: bool(a) != bool(b), src.values())  # type: ignore
            case ReduceOperation.BXOR:
                result = functools.reduce(operator.xor, src.values())
            case ReduceOperation.MINLOC:
                rank = min(src, key=src.__getitem__)  # type: ignore
                result = (src[rank], rank)
            case ReduceOperation.MAXLOC:
                rank = max(src, key=src.__getitem__)  # type: ignore
                result = (src[rank], rank)
            case _:
                raise NotImplementedError(f"op with not {self.op}")
        return dict.fromkeys(dst, result)  # type: ignore


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
class ScatterContext[T](OperationContext[T]):
    """Scatter operation"""
    root: Rank = 0

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup([self.root], range(size))

    def apply(self, src: coll_abc.Mapping[Rank, coll_abc.Sequence[T]], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, T]:
        """Apply operation over objects"""
        return dict(zip(sorted(dst), src[self.root]))


@dataclass(slots=True, frozen=True)
class AllToAllContext[T](OperationContext[T]):
    """All to all operation"""

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), range(size))

    def apply(self, src: coll_abc.Mapping[Rank, coll_abc.Sequence[T]], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, coll_abc.Sequence[T]]:
        """Apply operation over objects"""
        src = dict(sorted(src.items(), key=lambda item: item[0]))
        return dict(zip(sorted(dst), map(list, zip(*src.values()))))


@dataclass(slots=True, frozen=True)
class GatherContext[T](OperationContext[T]):
    """Gather operation"""
    root: Rank = 0

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), [self.root])

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, coll_abc.Sequence[T]]:
        """Apply operation over objects"""
        src = dict(sorted(src.items(), key=lambda item: item[0]))
        return {self.root: list(src.values())}


@dataclass(slots=True, frozen=True)
class ReduceContext[T](OperationContext[T]):
    """Reduce operation"""
    op: ReduceOperation = ReduceOperation.SUM
    root: Rank = 0

    def comm(self, size: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup(range(size), [self.root])

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, T]:
        """Apply operation over objects"""
        match self.op:
            case ReduceOperation.MAX:
                result = max(src.values())  # type: ignore
            case ReduceOperation.MIN:
                result = min(src.values())  # type: ignore
            case ReduceOperation.SUM:
                result = functools.reduce(operator.add, src.values())
            case ReduceOperation.PROD:
                result = functools.reduce(operator.mul, src.values())
            case ReduceOperation.LAND:
                result = all(src.values())
            case ReduceOperation.BAND:
                result = functools.reduce(operator.and_, src.values())
            case ReduceOperation.LOR:
                result = any(src.values())
            case ReduceOperation.BOR:
                result = functools.reduce(operator.or_, src.values())
            case ReduceOperation.LXOR:
                result = functools.reduce(lambda a, b: bool(a) != bool(b), src.values())  # type: ignore
            case ReduceOperation.BXOR:
                result = functools.reduce(operator.xor, src.values())
            case ReduceOperation.MINLOC:
                rank = min(src, key=src.__getitem__)  # type: ignore
                result = (src[rank], rank)
            case ReduceOperation.MAXLOC:
                rank = max(src, key=src.__getitem__)  # type: ignore
                result = (src[rank], rank)
            case _:
                raise NotImplementedError(f"op with not {self.op}")
        return {self.root: result}  # type: ignore


@dataclass(slots=True, frozen=True)
class SendRecvContext[T](OperationContext[T]):
    """Send and receive operation"""
    tag: Tag = 0

    def comm(self, src: int, dst: int) -> CommmunicationGroup:
        """Compute operation's communication group"""
        return CommmunicationGroup([src], [dst])

    def apply(self, src: coll_abc.Mapping[Rank, T], dst: coll_abc.Set[Rank]) -> coll_abc.Mapping[Rank, T]:
        """Apply operation over objects"""
        rank_src = next(iter(src))
        rank_dst = next(iter(dst))
        return {rank_dst: src[rank_src]}


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
