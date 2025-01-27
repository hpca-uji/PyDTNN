"""Message Passing Interface (server)"""

import typing
import threading
import functools
from pydtnn import comms
from queue import SimpleQueue
from pydtnn.libs.mpi import comm as mpi_comm
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor


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


class Server:
    """MPI server"""

    def __init__(self, thead_pool: ThreadPoolExecutor) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._pool = thead_pool
        self._lock = threading.Lock()
        self._queue = SimpleQueue[mpi_comm.Request]()

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.close()

    @functools.cached_property
    def _comm(self) -> comms.Communication:
        """Communication connection"""
        # Lazily initialized to prevent module imports execution
        return comms.Server()

    def _get(self) -> mpi_comm.Request:
        """Get request from clients"""
        return self._queue.get()

    def _put(self, obj) -> None:
        """Publish response to clients"""
        response = mpi_comm.Response(obj)
        self._comm.put(response)

    def _handle(self, request: mpi_comm.Request) -> None:
        """Client request handler"""
        # Store data
        self._queue.put(request)

        # Compute (if nobody else)
        if not self._lock.acquire(blocking=False):
            return
        try:
            name = request.operation.name.lower()
            handler = getattr(self, name)
            handler(request)
        finally:
            self._lock.release()

    def serve_forever(self) -> None:
        """Serve requests forever using worker pool"""
        try:
            while True:
                request = self._comm.get()
                self._pool.submit(self._handle, request)
        finally:
            self.close()

    def close(self) -> None:
        """Close the server"""
        if "_comm" in self.__dict__:
            self._comm.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def broadcast(self, context: mpi_comm.Request) -> None:
        """Broadcast."""
        self._put(self._get().obj)

    def gather(self, context: mpi_comm.Request) -> None:
        """Gather to All."""
        reqs = sorted(
            (self._get() for _ in range(context.size)),
            key=lambda req: req.rank
        )

        for req in reqs:
            self._put(req.obj)

    def reduce(self, context: mpi_comm.Request) -> None:
        """Reduce to All."""
        self._put(sum(self._get().obj for _ in range(context.size)))


def main(*args: str) -> None:
    """Application entrypoint"""
    config = arg_parser.parse_args(args)
    config = typing.cast(Namespace, config)
    if config.addr:
        comms.Server._addr = config.addr
    if config.port:
        comms.Server._port = config.port
    with ThreadPoolExecutor(max_workers=config.size) as pool:
        with Server(pool) as server:
            server.serve_forever()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
