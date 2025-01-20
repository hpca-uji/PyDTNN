"""Message Passing Interface."""

import os
import enum
import queue
import pickle
import numpy as np
from pydtnn.comms.mqtt import mpi_dc
import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client
import paho.mqtt.subscribeoptions as mqtt_subscribe

__all__ = (
    "Finalize",
    "IN_PLACE",
    "SUM",
    "COMM_WORLD",
)


StreamResponse = list[mpi_dc.RecvResponse]


def Finalize():
    """Terminate the MPI execution environment."""
    COMM_WORLD.Disconnect()


class InPlace(enum.Enum):
    """In-place buffer argument."""
    IN_PLACE = enum.auto()


class Op(enum.Enum):
    """Reduction operation."""
    SUM = enum.auto()


class Request:
    """Request handler."""

    def wait(self) -> None:
        """Wait for a non-blocking operation to complete."""


class Comm:
    """Communication context."""
    _pickle_protocol = 5
    _mqtt_protocol = mqtt_client.MQTTv5

    def __init__(self):
        """Inizialize comunication context"""
        self.size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self._mqtt_init()

    def _mqtt_init(self) -> None:
        """Inizialize MQTT context"""
        self._mqtt = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._mqtt_protocol
        )

        # Setup environment
        self._mqtt_host = os.environ.get("MQTT_HOST", "localhost")
        self._mqtt_queue = queue.SimpleQueue[mqtt_client.MQTTMessage]()

        # Client inizialization
        self._mqtt.connect(self._mqtt_host)
        self._mqtt.on_message = self._mqtt_handle_message
        self._mqtt.subscribe(f"/client/{self.rank}", options=mqtt_subscribe.SubscribeOptions(qos=2))

        self._mqtt.loop_start()

    def _mqtt_finalize(self):
        """Finalize MQTT context"""
        self._mqtt.loop_stop()
        del self._mqtt_queue

    def _mqtt_handle_message(self, client: mqtt_client.Client, userdata, msg: mqtt_client.MQTTMessage) -> None:
        """MQTT message handler"""
        self._mqtt_queue.put(msg)

    def _serialize(self, obj) -> bytes:
        """Serialize object for comunication"""
        return pickle.dumps(obj, protocol=self._pickle_protocol)

    def _deserialize(self, data: bytes):
        """Deserialize object for comunication"""
        return pickle.loads(data)

    def _send(self, obj):
        """Send object to server"""
        req = mpi_dc.SendRequest(rank=self.rank, data=self._serialize(obj))
        self._mqtt.publish(topic="/server", payload=self._serialize(req))

    def _recv_many(self, op: mpi_dc.Op):
        """Recive objects to server"""
        req = mpi_dc.RecvRequest(rank=self.rank, size=self.size, op=op)
        while True:
            self._mqtt.publish(topic="/server", payload=self._serialize(req))
            while True:
                msg = self._mqtt_queue.get()
                res = self._deserialize(msg.payload)
                match res:
                    case mpi_dc.RecvResponse():
                        yield self._deserialize(res.data)
                    case mpi_dc.SteamEnd():
                        return
                    case mpi_dc.UnavailableError():
                        break
                    case _:
                        raise mpi_dc.CommunicationError(f"Unknown response type {type(res)}")

    def _recv(self, op: mpi_dc.Op):
        """Recive object to server"""
        return next(self._recv_many(op))

    def Disconnect(self) -> None:
        """Disconnect from a communicator."""
        self._mqtt_finalize()

    def __del__(self):
        """Best effort finalizer"""
        try:
            self.Disconnect()
        except:  # noqa: E722
            pass

    def Get_rank(self) -> int:
        """Return the rank of this process in a communicator."""
        return self.rank

    def Get_size(self) -> int:
        """Return the number of processes in a communicator."""
        return self.size

    def bcast(self, obj, rank=0):
        """Broadcast."""
        if rank == self.rank:
            self._send(obj)
        return self._recv(op=mpi_dc.Op.BCAST)

    def Barrier(self) -> None:
        """Barrier synchronization."""
        self.allgather(None)

    def allgather(self, obj):
        """Gather to All."""
        self._send(obj)
        return list(self._recv_many(op=mpi_dc.Op.ALLGATHER))

    def Allreduce(self, sendbuf, recvbuf, op=Op.SUM) -> None:
        """Reduce to All."""
        if sendbuf is InPlace.IN_PLACE:
            sendbuf = recvbuf
        else:
            raise NotImplementedError("sendbuf with not IN_PLACE")

        if not isinstance(recvbuf, np.ndarray):
            raise NotImplementedError("recvbuf with not np.ndarray")

        if op is not Op.SUM:
            raise NotImplementedError("op with not SUM")

        self._send(sendbuf)
        recvbuf[:] = self._recv(op=mpi_dc.Op.ALLREDUCE)

    def Iallreduce(self, sendbuf, recvbuf, op=Op.SUM):
        """Nonblocking Reduce to All."""
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = Op.SUM

COMM_WORLD = Comm()
