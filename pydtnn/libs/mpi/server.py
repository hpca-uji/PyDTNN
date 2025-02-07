"""Message Passing Interface (server)"""

# NOTE: Module considerations
#
# Communications are lazily initialized to prevent module imports execution

# TODO: Optimize lock usage

import uuid
import functools
import threading
import dataclasses
from collections import deque
from dataclasses import dataclass
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn import comms
from pydtnn.libs.mpi import comm as mpi_comm


__all__ = (
    "Server",
)


# Argument pasrser
arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI server"
)
arg_parser.add_argument("-np", dest="size", type=int, default=4)
arg_parser.add_argument("-a", dest="addr", type=str, default=None)
arg_parser.add_argument("-p", dest="port", type=str, default=None)


@dataclass(slots=True, frozen=True)
class CommmunicationGroup:
    """Communication group"""
    src: frozenset[mpi_comm.Rank]
    dst: frozenset[mpi_comm.Rank]


@dataclass(slots=True)
class Operation[T: mpi_comm.OperationRequest]:
    """MPI Operation"""
    group: CommmunicationGroup = dataclasses.field()
    requests: dict[mpi_comm.Rank, T] = dataclasses.field(default_factory=dict)
    responses: list | None = None

    def put(self, rank: mpi_comm.Rank, request: T) -> bool:
        return request is self.requests.setdefault(rank, request)

    def full(self) -> bool:
        return len(self.requests) == len(self.group.src)


class Server:
    """MPI server"""

    def __init__(self) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._shutdown = False
        self._send_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4)

        self._state_lock = threading.Lock()
        self._state = dict[CommmunicationGroup, deque[Operation]]()

        self._peers_lock = threading.Lock()
        self._peers = bidict[mpi_comm.Rank, uuid.UUID]()

    @property
    def _size(self):
        """Get the approximate number of clients"""
        return len(self._peers)

    def serve_forever(self) -> None:
        """Handle requests forever"""
        while True:
            message = self._comm.get()
            request = message.obj

            match request:
                case mpi_comm.StateRequest():
                    self._handle_state_request(message)
                case mpi_comm.OperationRequest():
                    self._handle_operation_request(message)
                case _:
                    raise RuntimeError(f"Unknown request type {request}")

    def _handle_state_request(self, message: comms.Message[mpi_comm.StateRequest]) -> None:
        """Handle an state request"""
        request = message.obj

        match request:
            case mpi_comm.InitRequest():
                self._handle_init(message)  # type: ignore (not inferred my typecheker)
            case mpi_comm.FinalizeRequest():
                self._handle_finalize(message)  # type: ignore (not inferred my typecheker)
            case _:
                raise RuntimeError(f"Unknown state type {request}")

    def _handle_operation_request(self, message: comms.Message[mpi_comm.OperationRequest]) -> None:
        """Handle an operation request"""
        # Operation context
        peer = message.peer
        request = message.obj
        rank = self._peers.inverse[peer]
        src = request.request_requirements(size=self._size)
        dst = request.response_requirements(size=self._size)
        group = CommmunicationGroup(src=src, dst=dst)

        # Queue operation
        with self._state_lock:
            queue = self._state.setdefault(group, deque())

            for operation in queue:
                if operation.put(rank, request):
                    break
            else:
                operation = Operation(group=group)
                operation.put(rank, request)
                queue.append(operation)

        # Start operation
        if operation.full():
            future = self._pool.submit(self._handle_operation, operation)
            future.add_done_callback(lambda _: self._handle_operation_responses())

    def _pop_responded_operations(self) -> list[Operation]:
        """Get and remove responded operations from state"""
        assert self._send_lock.locked(), "Send lock not held, message order could be lost"
        operations = list[Operation]()

        with self._state_lock:
            groups = list[CommmunicationGroup]()

            for group, queue in self._state.items():
                for operation in queue:
                    if operation.responses is not None:
                        operations.append(operation)
                    else:
                        break

                for operation in operations:
                    queue.popleft()

                if len(queue) == 0:
                    groups.append(group)

            for group in groups:
                del self._state[group]

        return operations

    def _handle_operation_responses(self) -> None:
        """Cleanup responded operations from state"""
        # Try to send, if nobody else
        if not self._send_lock.acquire(blocking=False):
            return
        try:
            for operation in self._pop_responded_operations():
                self._send_operation(operation)
        finally:
            self._send_lock.release()

    def _send_operation(self, operation: Operation) -> None:
        """Send a operation response to the clients"""
        assert self._send_lock.locked(), "Send lock not held, message order could be lost"
        assert operation.responses is not None, f"Sending in progress operation {operation}"

        for obj in operation.responses:
            response = mpi_comm.OperationResponse(group=operation.group.dst, obj=obj)
            peers = [
                self._peers[rank]
                for rank in operation.group.dst
            ]
            self._comm.put(response, *peers)

    def _handle_operation(self, operation: Operation) -> None:
        """Dispatch operation to relevant handler"""
        context = operation.requests[0]

        match context:
            case mpi_comm.BroadcastRequest():
                handler = self._handle_broadcast
            case mpi_comm.AllGatherRequest():
                handler = self._handle_allgather
            case mpi_comm.AllReduceRequest():
                handler = self._handle_allreduce
            case _:
                raise RuntimeError(f"Unknown operation type {context}")

        handler(operation)

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.shutdown()

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # NOTE: Lazily initialized, prevent module imports execution
        return comms.Server()

    def shutdown(self) -> None:
        """Close the server"""
        if comm := self.__dict__.pop("_comm", None):
            comm.close()

        if self._shutdown:
            raise comms.ResourceClosed()
        self._shutdown = True

        self._pool.shutdown()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.shutdown()
        except:  # noqa: E722
            pass

    def _handle_init(self, message: comms.Message[mpi_comm.InitRequest]) -> None:
        """Initialize."""
        # Request context
        peer = message.peer
        request = message.obj
        rank = request.rank
        size = request.size

        # Thread-safe client setup
        with self._peers_lock:
            self._peers[rank] = peer

        # Syncronize clients when all ready
        if self._size == size:
            self._comm.put(None)

    def _handle_finalize(self, message: comms.Message[mpi_comm.FinalizeRequest]) -> None:
        """Terminate."""
        # Request context
        peer = message.peer
        rank = self._peers.inverse[peer]

        # Thread-safe client taredown
        with self._peers_lock:
            del self._peers[rank]

        # Syncronize clients when all ready
        if self._size == 0:
            self._comm.put(None)

    def _handle_broadcast(self, operation: Operation[mpi_comm.BroadcastRequest]) -> None:
        """Broadcast."""
        response = operation.requests[0].obj
        operation.responses = [response]

    def _handle_allgather(self, operation: Operation[mpi_comm.AllGatherRequest]) -> None:
        """Gather to All."""
        rank_requests = sorted(
            operation.requests.items(),
            key=lambda rank_request: rank_request[0]
        )

        operation.responses = [
            request.obj
            for _, request in rank_requests
        ]

    def _handle_allreduce(self, operation: Operation[mpi_comm.AllReduceRequest]) -> None:
        """Reduce to All."""
        response = sum(msg.obj for msg in operation.requests.values())
        operation.responses = [response]


def main(config: Namespace) -> None:
    """Application entrypoint"""
    if config.addr:
        comms.Server._addr = config.addr

    if config.port:
        comms.Server._port = config.port

    with Server() as server:
        server.serve_forever()


if __name__ == "__main__":
    main(arg_parser.parse_args())
