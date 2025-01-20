import grpc
import pickle
import typing
import threading
from concurrent import futures
from collections import defaultdict
from queue import SimpleQueue, Empty
from pydtnn.comms.grpc import mpi_pb2
from pydtnn.comms.grpc import mpi_pb2_grpc
from argparse import ArgumentParser, Namespace


StreamResponse = list[mpi_pb2.RecvResponse]


arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI gRPC server"
)

arg_parser.add_argument("-np", dest="size", type=int, default=4)
arg_parser.add_argument("-a", dest="addr", type=str, default="localhost")
arg_parser.add_argument("-p", dest="port", type=int, default="50051")


class MPIServicer(mpi_pb2_grpc.MPIServicer):
    _pickle_protocol = 5

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = SimpleQueue[mpi_pb2.SendRequest]()
        self.responses = defaultdict[int, SimpleQueue[StreamResponse]](SimpleQueue)

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object for comunication"""
        return pickle.loads(data)

    def send(self, request: mpi_pb2.SendRequest, context):
        """Recive data from clients"""
        self.requests.put(request)
        return mpi_pb2.SendResponse()

    def recv(self, request: mpi_pb2.RecvRequest, context):
        """Send data to clients"""
        match request.op:
            case mpi_pb2.Op.BCAST:
                yield from self.dispatch(request, context, self.bcast)

            case mpi_pb2.ALLGATHER:
                yield from self.dispatch(request, context, self.allgather)

            case mpi_pb2.ALLREDUCE:
                yield from self.dispatch(request, context, self.allreduce)

    def dispatch(self, request: mpi_pb2.RecvRequest, context: grpc.ServicerContext, handler):
        """Handle operation synchronization"""
        queue = self.responses[request.rank]

        try:  # LOCAL (RESPONSE)
            return queue.get_nowait()
        except Empty:  # LOCAL-GLOBAL (RACE)
            if self.lock.acquire(blocking=False):
                try:  # LOCAL (RESPONSE)
                    return queue.get_nowait()
                except Empty:  # GLOBAL (COMPUTE)
                    res = StreamResponse(handler(request))
                    for rank in range(request.size):
                        if rank != request.rank:
                            self.responses[rank].put(res)
                    return res
                finally:
                    self.lock.release()
            else:
                # GLOBAL (BUSY)
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                return []

    def bcast(self, request: mpi_pb2.RecvRequest):
        """Broadcast."""
        req = self.requests.get()
        yield mpi_pb2.RecvResponse(data=req.data)

    def allgather(self, request: mpi_pb2.RecvRequest):
        """Gather to All."""
        reqs = sorted((self.requests.get() for _ in range(request.size)), key=lambda req: req.rank)
        for req in reqs:
            yield mpi_pb2.RecvResponse(data=req.data)

    def allreduce(self, request: mpi_pb2.RecvRequest):
        """Reduce to All."""
        data = self._serialize(sum(self._deserialize(self.requests.get().data) for _ in range(request.size)))
        yield mpi_pb2.RecvResponse(data=data)


def main(*args: str) -> None:
    """Application entrypoint"""
    config = arg_parser.parse_args(args)
    config = typing.cast(Namespace, config)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.size))
    mpi_pb2_grpc.add_MPIServicer_to_server(MPIServicer(), server)
    server.add_insecure_port(f"{config.addr}:{config.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
