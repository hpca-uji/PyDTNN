"""Message Passing Interface."""

import os
import enum
import queue
import pickle
import numpy as np
import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client
import paho.mqtt.subscribeoptions as mqtt_subscribe

__all__ = (
    "Finalize",
    "IN_PLACE",
    "SUM",
    "COMM_WORLD",
)


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
        self._mqtt_topic = "/"
        self._mqtt_queue = queue.SimpleQueue()

        # Client inizialization
        self._mqtt.connect(self._mqtt_host)
        self._mqtt.on_message = self._mqtt_handle_message
        self._mqtt.subscribe(self._mqtt_topic, options=mqtt_subscribe.SubscribeOptions(qos=2))

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
        self._mqtt.publish(self._mqtt_topic, self._serialize(obj))

    def _recv(self):
        """Recive object to server"""
        return next(self._recv_many(1))

    def _recv_many(self, size=None):
        """Recive objects to server"""
        if size is None:
            size = self.size
        for _ in range(size):
            msg = self._mqtt_queue.get()
            yield self._deserialize(msg.payload)

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
        return self._recv()

    def Barrier(self) -> None:
        """Barrier synchronization."""
        self.allgather(None)

    def allgather(self, obj):
        """Gather to All."""
        rank_obj = (self.rank, obj)
        self._send(rank_obj)
        rank_objs = sorted(self._recv_many())
        return list(zip(*rank_objs))[1]

    def Allreduce(self, sendbuf, recvbuf, op=Op.SUM):
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
        recvbuf[:] = sum(self._recv_many())

    def Iallreduce(self, sendbuf, recvbuf, op=Op.SUM):
        """Nonblocking Reduce to All."""
        self.Allreduce(sendbuf, recvbuf, op)
        return Request()


# Exports
IN_PLACE = InPlace.IN_PLACE

SUM = Op.SUM

COMM_WORLD = Comm()
