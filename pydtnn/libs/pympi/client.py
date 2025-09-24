"""Message Passing Interface (client)"""

# NOTE: Due to how Python and external libraries handle threading, there is no
# reliable way to track when the MPI context should be automatically finalized.
#
# MPI for Python finalizes its context via a atexit handler, which waits for all
# non-daemon threads to finish before automatically finalizing (if not disabled).
#
# This implementation attempts to finalizes its context specifically when the main
# thread finishs. As other threads might be internal implementation details and
# might be required during the finalization stage. Waiting for a atexit handler
# might lead to those internal threads being reaped, therefor being unable to
# gracefully finalize.
#
# If this approach is not plausible, as it is Python implementation dependant,
# a classic atexit handler is also registered, but as mention before, this
# might lead to an undefined finaliztion behaviour.
#
# To ensure defined gracefull finalization the only reliable method is to manually
# call Finalize, and ensure it is always called, even when exceptions might be
# unhandled and lead to thread termination.

# TODO: Revise vector variants

# FIXME: Implement self-messaging

# FIXME: Move backgroud_server logic from communicator to module. Currently it is
# here because of import restrictions, but it will be an issue when multiple
# communicator are open.

import copy
import uuid
import enum
import typing
import atexit
import warnings
import functools
import threading
import itertools
from collections import abc
from concurrent import futures
from concurrent.futures import Future

from pydtnn.libs import net_queue as comms
from pydtnn.libs.net_queue import asynctools
from pydtnn.libs.net_queue.io_stream import byteview
from pydtnn.libs.net_queue.asynctools import thread_queue
from pydtnn.libs.pympi import protocol as mpi_comm, rc as mpi_rc, util as mpi_util


__all__ = (
    "Init",
    "Finalize",
    "IN_PLACE",
    "MAX",
    "MIN",
    "SUM",
    "PROD",
    "LAND",
    "BAND",
    "LOR",
    "BOR",
    "LXOR",
    "BXOR",
    "MINLOC",
    "MAXLOC",
    "COMM_WORLD",
)


ANY_TAG: mpi_comm.Tag = 0
ANY_SOURCE: mpi_comm.Rank = -1


def Init() -> None:
    """Initialize the MPI execution environment."""


def Finalize() -> None:
    """Terminate the MPI execution environment."""
    COMM_WORLD.Disconnect()


class InPlace(enum.Enum):
    """In-place buffer argument."""
    IN_PLACE = enum.auto()


class RequestState(enum.Enum):
    INI = enum.auto()
    ACK = enum.auto()
    RES = enum.auto()
    FIN = enum.auto()


class Request[T]:
    """Request handler."""

    def __init__(self) -> None:
        """Inizialize request"""
        self._state = RequestState.INI
        self._lock = threading.Lock()
        self._future = Future[T]()

    @staticmethod
    def _process[O](result: O) -> O:
        """Result post-processing"""
        if isinstance(result, mpi_comm.RemoteException):
            raise result
        return result

    def _resolve(self, future: Future[T]) -> None:
        """Resolve request according to future"""
        exc = future.exception()
        self._state = RequestState.FIN

        if exc is None:
            result = future.result()
            asynctools.future_set_result(self._future, result)
        else:
            asynctools.future_set_exception(self._future, exc)

    def wait(self, status=None) -> T:
        """Wait for a non-blocking operation to complete."""
        if status:
            raise ValueError("Status are not supported")

        with self._lock:
            if "_future" in self.__dict__:
                result = self._future.result()
                del self._future
            else:
                result = None
        return result  # type: ignore

    @classmethod
    def Waitall(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[int]:
        """Wait for all previously initiated requests to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        fs = [request._future for request in requests]
        done = futures.wait(fs=fs, return_when=futures.ALL_COMPLETED).done

        return list(map(fs.index, done))

    @classmethod
    def Waitsome(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[int]:
        """Wait for some previously initiated requests to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        fs = [request._future for request in requests]
        done = futures.wait(fs=fs, return_when=futures.FIRST_COMPLETED).done

        return list(map(fs.index, done))

    @classmethod
    def Waitany(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> int:
        """Wait for any previously initiated request to complete"""
        if statuses:
            raise ValueError("Status are not supported")

        return cls.Waitsome(requests, statuses)[0]

    @classmethod
    def waitall(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[T]:
        """Wait for all previously initiated requests to complete"""
        return [requests[i].wait() for i in cls.Waitall(requests, statuses)]

    @classmethod
    def waitsome(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> list[T]:
        """Wait for some previously initiated requests to complete"""
        return [requests[i].wait() for i in cls.Waitsome(requests, statuses)]

    @classmethod
    def waitany(cls, requests: abc.Sequence["Request[T]"], statuses=None) -> T:
        """Wait for any previously initiated request to complete"""
        return requests[cls.Waitany(requests, statuses)].wait()


class Comm:
    """Communicator."""

    def __init__(self, comm_options: comms.CommunicatorOptions = comms.CommunicatorOptions()) -> None:
        """Communicator initialization"""
        self._comm_options = copy.replace(mpi_util.comm_options(comm_options))

        self._comm_lock = threading.Lock()

        self._close_init = threading.Lock()
        self._close_done = threading.Event()

        self._requests = dict[uuid.UUID, Request]()
        self._responses = dict[uuid.UUID, typing.Any]()

        thread_prefix = f"{__name__}.{self.__class__.__qualname__}:{id(self)}"
        self._recv_queue = thread_queue(f"{thread_prefix}.recv")
        self._proc_queue = thread_queue(f"{thread_prefix}.proc")

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_init.locked()

    def _recive_response(self) -> None:
        """Recive one response from communication"""
        while self._requests:
            try:
                response = self._comm.get().obj
            except Exception as exc:
                warnings.warn(repr(exc), RuntimeWarning)
                continue

            match response:
                case mpi_comm.OperationResponse():
                    pass
                case _:
                    raise RuntimeError(f"Unknown response {response}")

            self._responses[response.id] = response.obj

            # Process pending requests
            self._handle_request(response.id)

    def _handle_request(self, id: uuid.UUID) -> None:
        """Handle a request"""
        # While matching request and response
        while id in self._requests and id in self._responses:
            request = self._requests.pop(id)
            response = self._responses.pop(id)

            match request._state:
                case RequestState.INI:
                    id = response
                    self._requests[id] = request
                    request._state = RequestState.ACK
                case RequestState.ACK:
                    if isinstance(response, mpi_comm.RemoteException):
                        request._process = Request._process  # Remove callback
                    request._state = RequestState.RES
                    future = self._proc_queue.submit(request._process, response)
                    future.add_done_callback(request._resolve)
                case _:
                    raise RuntimeError(f"Invalid request state {request._state}")

    def _shedule_operation(self, comm: mpi_comm.CommmunicationGroup, context: mpi_comm.OperationContext, obj: typing.Any = None, process: abc.Callable = Request._process) -> Request:
        """Schedule a new operation"""

        # Create appropriate request according to participation
        if self.rank == comm.root:
            operation = mpi_comm.OperationRequest(comm=comm, context=context, obj=obj)
        elif self.rank in comm.src:
            operation = mpi_comm.OperationRequest(comm=comm, obj=obj)
        elif self.rank in comm.dst:
            operation = mpi_comm.OperationRequest(comm=comm)
        else:
            raise RuntimeError("Tried to schedule operation without participating")

        request = Request()
        request._process = process
        asynctools.future_set_running(request._future)

        if self.rank in comm.dst:
            self._requests[operation.id] = request

        future = self._comm.put(operation)

        if self.rank in comm.dst:
            future.add_done_callback(lambda future: future.exception() and request._resolve(future))
            self._recv_queue.submit(self._recive_response).add_done_callback(asynctools.future_warn_exception)
        else:
            future.add_done_callback(request._resolve)

        return request

    @functools.cached_property
    def _comm(self) -> comms.Communicator:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if "_comm" in self.__dict__:
                comm = self.__dict__["_comm"]
            else:
                comm = self.__dict__["_comm"] = self._new_comm()
        return comm

    def _new_comm(self) -> comms.Communicator:
        """Create a new communication and inizialize it"""

        # If requested, start a local server
        if mpi_rc.init:
            if self.rank == 0:
                from pydtnn.libs.pympi.server import background_server
                self._server = background_server()

            # Allow some time for server startup
            from time import sleep
            sleep(mpi_rc.wait)

        state = mpi_comm.RankInit(rank=self.rank)
        try:
            assert mpi_rc.proto, "MPI comunication protocol not defined!"
            comm = comms.new(protocol=mpi_rc.proto, purpose=comms.Purpose.CLIENT, options=self._comm_options)
            comm.put(state)
            while True:
                response = comm.get().obj

                match response:
                    case mpi_comm.StateResponse():
                        pass
                    case _:
                        warnings.warn(f"Unknown response {response}", RuntimeWarning)
                        continue  # response lost

                if response.size == self.size:
                    break
        except Exception:
            try:
                comm.close()
            except:  # noqa: E722
                pass
            raise
        return comm

    def _close_comm(self, comm: comms.Communicator) -> None:
        """Fianlize a communication object"""
        state = mpi_comm.RankFinalize()
        try:
            comm.put(state)
            while True:
                response = comm.get().obj

                match response:
                    case mpi_comm.StateResponse():
                        pass
                    case _:
                        warnings.warn(f"Unknown response {response}", RuntimeWarning)
                        continue  # response lost

                if response.size == 0:
                    break
        finally:
            comm.close()

        # If requested, stop a local server
        if mpi_rc.init:
            if self.rank == 0:
                self._server.result()

            # Allow some time for server shutdown
            from time import sleep
            sleep(mpi_rc.wait)

    def _close(self) -> None:
        """Communicator finalizer"""
        # Close comunicator if initialized
        with self._comm_lock:
            if "_comm" in self.__dict__:
                self.barrier()

            self._recv_queue.shutdown()
            self._proc_queue.shutdown()

            if comm := self.__dict__.pop("_comm", None):
                self._close_comm(comm)

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        if self._close_init.acquire(blocking=False):
            self._close()
            self._close_done.set()
        self._close_done.wait()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    # Properties
    @property
    def size(self) -> int:
        """Communication size"""
        return mpi_rc.size

    @property
    def rank(self) -> mpi_comm.Rank:
        """Communication identifier"""
        return mpi_rc.rank

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    # Barrier synchronization
    def ibarrier(self) -> Request[None]:
        """Nonblocking Barrier synchronization."""
        return self.iallreduce(None, mpi_comm.ReduceOperation.LAND)

    def barrier(self) -> None:
        """Barrier synchronization."""
        return self.ibarrier().wait()

    def Ibarrier(self) -> Request[None]:
        """Nonblocking Barrier synchronization."""
        return self.ibarrier()

    def Barrier(self) -> None:
        """Barrier synchronization."""
        return self.Ibarrier().wait()

    # Broadcast
    def ibcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> Request[T]:
        """Broadcast."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=obj)

    def bcast[T](self, obj: T, root: mpi_comm.Rank = 0) -> T:
        """Broadcast."""
        return self.ibcast(obj=obj, root=root).wait()

    def Ibcast[T: abc.Buffer](self, buf: T, root: mpi_comm.Rank = 0) -> Request[None]:
        """Nonblocking Gather to All."""
        context = mpi_comm.BroadcastContext(root=root)
        comm = context.comm(size=self.size)

        def process(result: T) -> None:
            with byteview(result) as src, byteview(buf) as dst:
                dst[:] = src

        return self._shedule_operation(comm=comm, context=context, obj=buf, process=process)

    def Bcast[T: abc.Buffer](self, buf: T, root: mpi_comm.Rank = 0) -> None:
        """Gather to All."""
        return self.Ibcast(buf=buf, root=root).wait()

    # Gather to All
    def iallgather[T](self, sendobj: T) -> Request[list[T]]:
        """Nonblocking Gather to All."""
        context = mpi_comm.AllGatherContext()
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def allgather[T](self, sendobj: T) -> list[T]:
        """Gather to All."""
        return self.iallgather(sendobj=sendobj).wait()

    # Reduce to All
    def iallreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[T]:
        """Reduce to All."""
        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def allreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All."""
        return self.iallreduce(sendobj=sendobj, op=op).wait()

    def Iallreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        context = mpi_comm.AllReduceContext(op=op)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Allreduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> None:
        """Reduce to All."""
        self.Iallreduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op).wait()

    # Reduce to All (with steps)
    def _phased_allreduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM) -> T:
        """Reduce to All (with steps)."""
        context = mpi_comm.AllPhasedReduceContext(op=op)

        for phase in itertools.count():
            comm = context.comm(rank=self.rank, size=self.size, phase=phase)
            sendobj = self._shedule_operation(comm=comm, context=context, obj=sendobj).wait()
            if len(comm.dst) == self.size:
                break

        return sendobj

    # Scatter
    def iscatter[T](self, sendobj: abc.Sequence[T], root: mpi_comm.Rank = 0) -> Request[T]:
        """Nonblocking Scatter."""
        context = mpi_comm.ScatterContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def scatter[T](self, sendobj: abc.Sequence[T], root: mpi_comm.Rank = 0) -> T:
        """Scatter."""
        return self.iscatter(sendobj=sendobj, root=root).wait()

    # All to All Scatter/Gather
    def ialltoall[T](self, sendobj: abc.Sequence[T]) -> Request[list[T]]:
        """All to All Scatter/Gather."""
        context = mpi_comm.AllToAllContext()
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def alltoall[T](self, sendobj: abc.Sequence[T]) -> list[T]:
        """All to All Scatter/Gather."""
        return self.ialltoall(sendobj=sendobj).wait()

    # Gather to All
    def igather[T](self, sendobj: T, root: mpi_comm.Rank = 0) -> Request[list[T]]:
        """Nonblocking Gather."""
        context = mpi_comm.GatherContext(root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def gather[T](self, sendobj: T, root: mpi_comm.Rank = 0) -> list[T]:
        """Gather."""
        return self.igather(sendobj=sendobj, root=root).wait()

    # Reduce
    def ireduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> Request[T]:
        """Reduce to All."""
        context = mpi_comm.ReduceContext(op=op, root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendobj)

    def reduce[T](self, sendobj: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> T:
        """Reduce to All."""
        return self.ireduce(sendobj=sendobj, op=op, root=root).wait()

    def Ireduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> Request[None]:
        """Nonblocking Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf

        def process(result: T) -> None:
            with byteview(result) as src, byteview(recvbuf) as dst:
                dst[:] = src

        context = mpi_comm.ReduceContext(op=op, root=root)
        comm = context.comm(size=self.size)
        return self._shedule_operation(comm=comm, context=context, obj=sendbuf, process=process)

    def Reduce[T: abc.Buffer](self, sendbuf: T | typing.Literal[InPlace.IN_PLACE], recvbuf: T, op: mpi_comm.ReduceOperation = mpi_comm.ReduceOperation.SUM, root: mpi_comm.Rank = 0) -> None:
        """Reduce to All."""
        self.Ireduce(sendbuf=sendbuf, recvbuf=recvbuf, op=op, root=root).wait()

    # Send
    def isend[T](self, obj: T, dest: mpi_comm.Rank, tag: mpi_comm.Tag = 0) -> Request[None]:
        """Nonblocking Send in standard mode."""
        if dest == self.rank:
            raise ValueError("Self send not supported")

        if tag:
            raise ValueError("Tags are not supported")

        context = mpi_comm.SendRecvContext(tag=tag)
        comm = context.comm(src=self.rank, dst=dest)
        return self._shedule_operation(comm=comm, context=context, obj=obj)

    def send[T](self, obj: T, dest: mpi_comm.Rank, tag: mpi_comm.Tag = 0) -> None:
        """Send in standard mode."""
        return self.isend(obj=obj, dest=dest, tag=tag).wait()

    def Isend[T: abc.Buffer](self, buf: T, dest: mpi_comm.Rank, tag: mpi_comm.Tag = 0) -> Request[None]:
        """Nonblocking Send in standard mode."""
        return self.isend(obj=buf, dest=dest, tag=tag)

    def Send[T: abc.Buffer](self, buf: T, dest: mpi_comm.Rank, tag: mpi_comm.Tag = 0) -> None:
        """Send in standard mode."""
        return self.Isend(buf=buf, dest=dest, tag=tag).wait()

    # Receive
    def irecv[T: abc.Buffer](self, buf: T | None = None, source: mpi_comm.Rank = ANY_SOURCE, tag: mpi_comm.Tag = ANY_TAG, status=None) -> Request[T]:
        """Nonblocking Receive."""
        if source == self.rank:
            raise ValueError("Self source not supported")

        if source == ANY_SOURCE:
            raise ValueError("Any source is not supported")

        if tag:
            raise ValueError("Tags are not supported")

        if status:
            raise ValueError("Status are not supported")

        def process(result: T):
            if buf is None:
                return result
            with byteview(result) as src, byteview(buf) as dst:
                dst[:] = src

        context = mpi_comm.SendRecvContext(tag=tag)
        comm = context.comm(src=source, dst=self.rank)
        return self._shedule_operation(comm=comm, context=context, process=process)

    def recv[T: abc.Buffer](self, buf: T | None = None, source: mpi_comm.Rank = ANY_SOURCE, tag: mpi_comm.Tag = ANY_TAG, status=None) -> T:
        """Nonblocking Receive."""
        return self.irecv(buf=buf, source=source, tag=tag, status=status).wait()

    # Send and Recevie
    def isendrecv[S, R: abc.Buffer](self, sendobj: S, dest: mpi_comm.Rank, sendtag: mpi_comm.Rank = 0, recvbuf: R | None = None, source: mpi_comm.Rank = ANY_SOURCE, recvtag: mpi_comm.Tag = ANY_TAG, status=None) -> Request[R]:
        """Nonblocking Send and Receive."""
        send = self.isend(obj=sendobj, dest=dest, tag=sendtag)
        recv = self.irecv(buf=recvbuf, source=source, tag=recvtag, status=status)

        request = Request[R]()
        future = asynctools.merge_futures([send._future, recv._future])
        future.add_done_callback(lambda future: request._resolve(recv._future))

        return request

    def sendrecv[S, R: abc.Buffer](self, sendobj: S, dest: mpi_comm.Rank, sendtag: mpi_comm.Rank = 0, recvbuf: R | None = None, source: mpi_comm.Rank = ANY_SOURCE, recvtag: mpi_comm.Tag = ANY_TAG, status=None) -> R:
        """Nonblocking Send and Receive."""
        return self.isendrecv(sendobj=sendobj, dest=dest, sendtag=sendtag, recvbuf=recvbuf, source=source, recvtag=recvtag, status=status).wait()

    # Send and Recevie
    def Isendrecv[S: abc.Buffer, R: abc.Buffer](self, sendbuf: S, dest: mpi_comm.Rank, sendtag: mpi_comm.Rank = 0, recvbuf: R | None = None, source: mpi_comm.Rank = ANY_SOURCE, recvtag: mpi_comm.Tag = ANY_TAG, status=None) -> Request[R]:
        """Nonblocking Send and Receive."""
        request = Request[R]()
        send = self.Isend(buf=sendbuf, dest=dest, tag=sendtag)
        recv = self.irecv(buf=recvbuf, source=source, tag=recvtag, status=status)

        def process(future: Future[None]) -> None:
            if send._future.exception():
                request._resolve(send._future)  # type: ignore
            request._resolve(recv._future)

        future = asynctools.merge_futures([send._future, recv._future])
        future.add_done_callback(process)
        return request

    def Sendrecv[S: abc.Buffer, R: abc.Buffer](self, sendbuf: S, dest: mpi_comm.Rank, sendtag: mpi_comm.Rank = 0, recvbuf: R | None = None, source: mpi_comm.Rank = ANY_SOURCE, recvtag: mpi_comm.Tag = ANY_TAG, status=None) -> R:
        """Nonblocking Send and Receive."""
        return self.Isendrecv(sendbuf=sendbuf, dest=dest, sendtag=sendtag, recvbuf=recvbuf, source=source, recvtag=recvtag, status=status).wait()


# Exports
IN_PLACE = InPlace.IN_PLACE

MAX = mpi_comm.ReduceOperation.MAX
MIN = mpi_comm.ReduceOperation.MIN
SUM = mpi_comm.ReduceOperation.SUM
PROD = mpi_comm.ReduceOperation.PROD
LAND = mpi_comm.ReduceOperation.LAND
BAND = mpi_comm.ReduceOperation.BAND
LOR = mpi_comm.ReduceOperation.LOR
BOR = mpi_comm.ReduceOperation.BOR
LXOR = mpi_comm.ReduceOperation.LXOR
BXOR = mpi_comm.ReduceOperation.BXOR
MINLOC = mpi_comm.ReduceOperation.MINLOC
MAXLOC = mpi_comm.ReduceOperation.MAXLOC

COMM_WORLD = Comm()

# Best effort finalizer
try:
    threading._register_atexit(Finalize)  # type: ignore (private implementation dependant)
except AttributeError:
    atexit.register(Finalize)
