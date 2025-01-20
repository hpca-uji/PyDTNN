import os
import typing
import pickle
import threading
from concurrent import futures
from collections import defaultdict
from pydtnn.comms.mqtt import mpi_dc
from queue import SimpleQueue, Empty
import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.subscribeoptions as mqtt_subscribe
from argparse import ArgumentParser, Namespace

# from queue import _PySimpleQueue as SimpleQueue

StreamResponse = list[mpi_dc.RecvResponse]


arg_parser = ArgumentParser(
    prog="mpi_server",
    description="MPI gRPC server"
)

arg_parser.add_argument("-np", dest="size", type=int, default=4)
arg_parser.add_argument("-a", dest="addr", type=str, default="localhost")
arg_parser.add_argument("-p", dest="port", type=int, default="50051")


class MPIServer:
    _pickle_protocol = 5
    _mqtt_protocol = mqtt_client.MQTTv5

    def __init__(self, thead_pool: ThreadPoolExecutor) -> None:
        self.pool = thead_pool
        self.lock = threading.Lock()
        self.requests = SimpleQueue[mpi_dc.SendRequest]()
        self.responses = defaultdict[int, SimpleQueue[StreamResponse]](SimpleQueue)

    def debug(self):
        """Show internal server state"""
        assert hasattr(self.requests, "_queue"), "Use _PySimpleQueue as SimpleQueue to allow inspection"
        print("=" * 50)
        print(f"compute={self.lock}")
        print(f"requests={list(self.requests._queue)}") # type: ignore
        for rank, queue in sorted(self.responses.copy().items()):
            print(f"responses[{rank}]={list(queue._queue)}") # type: ignore
        print("=" * 50)

    def _mqtt_init(self) -> None:
        """Inizialize MQTT context"""
        self._mqtt = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._mqtt_protocol
        )

        # Setup environment
        self._mqtt_host = os.environ.get("MQTT_HOST", "localhost")
        self._mqtt_queue = SimpleQueue[mqtt_client.MQTTMessage]()

        # Client inizialization
        self._mqtt.connect(self._mqtt_host)
        self._mqtt.on_message = self._mqtt_handle_message
        self._mqtt.subscribe(topic="/server", options=mqtt_subscribe.SubscribeOptions(qos=2))

        self._mqtt.loop_start()

    def _mqtt_handle_message(self, client: mqtt_client.Client, userdata, msg: mqtt_client.MQTTMessage) -> None:
        """MQTT message handler"""
        self._mqtt_queue.put(msg)

    def _mqtt_finalize(self):
        """Finalize MQTT context"""
        self._mqtt.loop_stop()
        del self._mqtt_queue

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object for comunication"""
        return pickle.loads(data)

    def serve_forever(self):
        """Serve requests forever using worker pool"""
        try:
            self._mqtt_init()
            while True:
                msg = self._mqtt_queue.get()
                self.pool.submit(self.handle, msg)
        finally:
            self._mqtt_finalize()

    def handle(self, msg: mqtt_client.MQTTMessage):
        """Handle one request"""
        req = self._deserialize(msg.payload)
        match req:
            case mpi_dc.SendRequest():
                _ = self.send(req)
            case mpi_dc.RecvRequest():
                try:
                    res_steam = self.recv(req)
                    for res in res_steam:
                        self._mqtt.publish(topic=f"/client/{req.rank}", payload=self._serialize(res))
                    self._mqtt.publish(topic=f"/client/{req.rank}", payload=self._serialize(mpi_dc.SteamEnd()))
                except mpi_dc.UnavailableError as exc:
                    self._mqtt.publish(topic=f"/client/{req.rank}", payload=self._serialize(exc))
            case _:
                raise mpi_dc.CommunicationError(f"Unknown request type {type(req)}")

    def send(self, request: mpi_dc.SendRequest):
        """Recive data from clients"""
        self.requests.put(request)
        return mpi_dc.SendResponse()

    def recv(self, request: mpi_dc.RecvRequest):
        """Send data to clients"""
        match request.op:
            case mpi_dc.Op.BCAST:
                yield from self.dispatch(request, self.bcast)

            case mpi_dc.Op.ALLGATHER:
                yield from self.dispatch(request, self.allgather)

            case mpi_dc.Op.ALLREDUCE:
                yield from self.dispatch(request, self.allreduce)

    def dispatch(self, request: mpi_dc.RecvRequest, handler):
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
                raise mpi_dc.UnavailableError()

    def bcast(self, request: mpi_dc.RecvRequest):
        """Broadcast."""
        req = self.requests.get()
        yield mpi_dc.RecvResponse(data=req.data)

    def allgather(self, request: mpi_dc.RecvRequest):
        """Gather to All."""
        reqs = sorted((self.requests.get() for _ in range(request.size)), key=lambda req: req.rank)
        for req in reqs:
            yield mpi_dc.RecvResponse(data=req.data)

    def allreduce(self, request: mpi_dc.RecvRequest):
        """Reduce to All."""
        data = self._serialize(sum(self._deserialize(self.requests.get().data) for _ in range(request.size)))
        yield mpi_dc.RecvResponse(data=data)


def main(*args: str) -> None:
    """Application entrypoint"""
    config = arg_parser.parse_args(args)
    config = typing.cast(Namespace, config)

    server = MPIServer(futures.ThreadPoolExecutor(max_workers=config.size))
    server.serve_forever()


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
