"""Message Passing Interface (server)"""

# NOTE: Communications are lazily initialized to prevent module imports execution

# FIXME: serve_until_... exit when no clients, not when no clients and no operations

import copy
import uuid
import typing
import warnings
import functools
import threading
from collections import defaultdict
from concurrent.futures import Future
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

import net_queue as nq
from net_queue import asynctools
from pympi import proto, rc, util


__all__ = (
    "Server",
    "background_server"
)


# Argument pasrser
arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI server"
)
arg_parser.add_argument("--oneshot", action="store_true")


class Operation:
    """MPI Operation"""

    def __init__(self, group: proto.CommmunicationGroup) -> None:
        """Initialize operation"""
        self.id = uuid.uuid4()
        self.group = group
        self.compute: Future[None] | None = None
        self.requests = dict[proto.Rank, proto.OperationRequest]()

    def put(self, rank: proto.Rank, request: proto.OperationRequest) -> bool:
        """Try to insert request into operation"""
        return request.group == self.group and request is self.requests.setdefault(rank, request)

    @property
    def ctx(self) -> proto.OperationContext:
        """Operation context"""
        request = self.requests[self.group.root]
        assert request.ctx is not None, "Root request has no context"
        return request.ctx

    @property
    def src_ready(self) -> bool:
        """Operation sources ready"""
        return set(self.group.src).issubset(self.requests)

    @property
    def dst_ready(self) -> bool:
        """Operation destinations ready"""
        return set(self.group.dst).issubset(self.requests)

    @property
    def objs(self) -> dict[proto.Rank, typing.Any]:
        """Operations data"""
        return {
            rank: self.requests[rank].data
            for rank in self.group.src
        }


class Server:
    """MPI server"""

    def __init__(self, thread_pool: ThreadPoolExecutor, comm_options: nq.CommunicatorOptions = nq.CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__()
        self._comm_options = copy.replace(util.comm_options(comm_options), workers=rc.size)

        # State
        self._shutdown = False
        self._pool = thread_pool
        self._comm_lock = threading.Lock()

        self._close_init = threading.Lock()
        self._close_done = threading.Event()

        self._state_lock = threading.Lock()
        self._state = list[Operation]()

        self._peers_lock = threading.Lock()
        self._peers = bidict[proto.Rank, uuid.UUID]()

    @property
    def _closed(self):
        """Is communicator closed"""
        return self._close_init.locked()

    @property
    def _size(self):
        """Get the approximate number of clients"""
        return len(self._peers)

    @functools.cached_property
    def _comm(self) -> nq.Communicator:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        with self._comm_lock:
            if comm := self.__dict__.get("_comm"):
                pass
            else:
                comm = self.__dict__["_comm"] = nq.new(protocol=rc.proto, purpose=nq.Purpose.SERVER, options=self._comm_options)
        return comm

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def serve_forever(self) -> None:
        """Handle until shutdown"""
        while not self._closed:
            self.serve_util_finalize()

    def serve_util_finalize(self) -> None:
        """Handle until finalized"""

        while not self._closed:
            try:
                message = self._comm.get()
            except Exception as exc:
                warnings.warn(repr(exc), RuntimeWarning)
                continue
            request = message.data

            # Handle request
            match request:
                case proto.StateRequest():
                    self._handle_state_request(message)
                case proto.OperationRequest():
                    self._handle_operation_request(message)
                case _:
                    warnings.warn(f"Unknown request type {request}", RuntimeWarning)
                    continue

            # Finish if idle
            if self._size == 0:
                break

    def _handle_state_request(self, message: nq.Message[proto.StateRequest]) -> None:
        """Handle an state request"""
        request = message.data

        match request:
            case proto.RankInit():
                self._handle_init(message)  # type: ignore (not inferred by typecheker)
            case proto.RankFinalize():
                self._handle_finalize(message)  # type: ignore (not inferred by typecheker)
            case _:
                warnings.warn(f"Unknown state type {request}", RuntimeWarning)
                return

    def _handle_init(self, message: nq.Message[proto.RankInit]) -> None:
        """Initialize."""
        # Request context
        peer = message.peer
        request = message.data
        rank = request.rank

        # Thread-safe client setup
        with self._peers_lock:
            self._peers[rank] = peer

        # Inform clients of state change
        self._comm.put(proto.StateResponse(size=self._size))

    def _handle_finalize(self, message: nq.Message[proto.RankFinalize]) -> None:
        """Terminate."""
        # Request context
        peer = message.peer
        rank = self._peers.inverse[peer]

        # Thread-safe client taredown
        with self._peers_lock:
            del self._peers[rank]

        # Inform clients of state change
        self._comm.put(proto.StateResponse(size=self._size))

    def _handle_operation_request(self, message: nq.Message[proto.OperationRequest]) -> None:
        """Handle an operation request"""
        # Operation context
        peer = message.peer
        request = message.data
        rank = self._peers.inverse[peer]

        for operation in self._state:
            if operation.put(rank, request):
                break
        else:
            operation = Operation(group=request.group)
            pushed = operation.put(rank, request)
            assert pushed, "Could not inset request into empty operation"
            self._state.append(operation)

        # Send proxy response
        if rank in request.group.dst:
            self._comm.put(proto.OperationResponse(data=operation.id, id=request.id), peer)

        # Start operation compute
        if operation.compute is None and operation.src_ready:
            operation.compute = self._pool.submit(self._handle_operation, operation)
            operation.compute.add_done_callback(asynctools.future_warn_exception)

        # Operation queuing finished
        if operation.src_ready and operation.dst_ready:
            self._state.remove(operation)

    def _handle_operation(self, operation: Operation) -> None:
        """Dispatch operation to relevant handler"""
        # Setup compute
        context = operation.ctx
        src = operation.objs

        # Compute result
        try:
            dst = context.apply(src, operation.group.dst)
        except Exception as exc:
            exc = proto.RemoteException.from_exception(exc)
            dst = dict.fromkeys(operation.group.dst, exc)

        # Group result
        results = {}
        ranks = defaultdict(set)
        for rank, result in dst.items():
            ranks[id(result)].add(rank)
            results[id(result)] = result

        # Send result
        for result_id, ranks in ranks.items():
            result = results[result_id]
            response = proto.OperationResponse(data=result, id=operation.id)
            self._comm.put(response, *(self._peers[rank] for rank in ranks))

    def _close(self) -> None:
        """Close communicator"""
        if comm := self.__dict__.pop("_comm", None):
            comm.close()

    def close(self) -> None:
        """Close the server"""
        if self._close_init.acquire(blocking=False):
            self._close()
            self._close_done.set()
        self._close_done.wait()


def background_server() -> Future:
    """Start a background server"""
    from net_queue.asynctools import thread_func

    def serve_oneshot():
        with ThreadPoolExecutor(thread_name_prefix=f"{__name__}") as pool:
            with Server(pool) as server:
                server.serve_util_finalize()

    future = thread_func(serve_oneshot)
    future.add_done_callback(asynctools.future_warn_exception)
    return future


def main(config: Namespace) -> None:
    """Application entrypoint"""
    with ThreadPoolExecutor(thread_name_prefix=f"{__name__}") as pool:
        with Server(pool, rc.comm) as server:
            if config.oneshot:
                server.serve_util_finalize()
            else:
                server.serve_forever()


if __name__ == "__main__":
    main(arg_parser.parse_args())
