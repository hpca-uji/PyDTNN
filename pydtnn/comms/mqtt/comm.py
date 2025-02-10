"""MQTT communication"""

# NOTE: Module considerations
#
# Low level comunications are handled single-threaded and are limited to pushing
# or pulling data to queues without blocking, so all operations are minimal
# and fast.
#
# Expensive operations, such as serialization and blocking, are done at at the
# public's API callers thread.

# TODO: Peer-specific and global comunications are topic optimized, but peer groups
# are not. This could be implemented using grouping requests that generate new
# UUID per group. This would reduce also reduce load on the broker.

import uuid
import threading
from queue import Empty, SimpleQueue

import paho.mqtt.client as mqtt_client

from pydtnn.comms.mqtt import Protocol
from pydtnn.comms import ResourceClosed, Message


__all__ = (
    "Server",
    "Client"
)


# Sentinel objects
END_COMM = b""


class Server(Protocol):
    """MQTT server"""

    def __init__(self) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._lock = threading.Lock()
        self._request_count = threading.Semaphore(value=0)
        self._requests = dict[uuid.UUID, SimpleQueue[bytes]]()

        # MQTT
        self._register_handler(topic="syc/+", handler=self._syc)
        self._register_handler(topic="fin/+", handler=self._fin)
        self._register_handler(topic="c2s/+", handler=self._c2s)

    def _syc(self, client: mqtt_client.Client, userdata, mqtt_message: mqtt_client.MQTTMessage) -> None:
        """Client connection startup"""
        # NOTE: communication thead
        peer = self._peer(mqtt_message)

        # Thread-safe client setup
        with self._lock:
            self._requests[peer] = SimpleQueue()

        # Send server identification
        data = self._serialize(self.id)
        self._publish(topic=f"s2c/{peer}", data=data)

    def _fin(self, client: mqtt_client.Client, userdata, mqtt_message: mqtt_client.MQTTMessage) -> None:
        """Client connection finalizer"""
        # NOTE: communication thead
        peer = self._peer(mqtt_message)

        # Thread-safe client taredown
        with self._lock:
            requests = self._requests.pop(peer)

        # Drain queues and update counts
        for _ in range(requests.qsize()):
            self._request_count.acquire()

    def _c2s(self, client: mqtt_client.Client, userdata, mqtt_message: mqtt_client.MQTTMessage) -> None:
        """Client message handler"""
        # NOTE: communication thead
        peer = self._peer(mqtt_message)
        data = mqtt_message.payload

        self._requests[peer].put(data)
        self._request_count.release()

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        super().get(*peers)

        while True:
            # Wait for a request
            # FIXME: if peers is defined and other peers have messages this is a busy-wait
            self._request_count.acquire()

            # Acquire peers
            if peers:
                _peers = peers
            else:
                with self._lock:
                    _peers = tuple(self._requests)

            # Search for a request
            for peer in _peers:
                try:
                    data = self._requests[peer].get_nowait()
                except (KeyError, Empty):
                    continue
                else:
                    break

            # Request not found, revert notification and retry
            else:
                self._request_count.release()
                continue

            # Request found, continue
            break

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=peer, obj=obj)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to clients"""
        super().put(obj, *peers)
        data = self._serialize(obj)

        if not peers:
            self._publish(topic="s2c", data=data)
        else:
            # TODO: Optimize peer groups
            for peer in peers:
                self._publish(topic=f"s2c/{peer}", data=data)

    def close(self) -> None:
        """Close the server"""
        super().close()
        for queue in self._requests.values():
            queue.put(END_COMM)


class Client(Protocol):
    """MQTT client"""

    def __init__(self) -> None:
        """Client initialization"""
        super().__init__()

        # State
        self._responses = SimpleQueue[bytes]()

        # MQTT
        self._register_handler(topic="s2c", handler=self._s2c)
        self._register_handler(topic=f"s2c/{self.id}", handler=self._s2c)
        self._publish(topic=f"syc/{self.id}")
        data = self._responses.get()
        self._server = self._deserialize(data)

    def _s2c(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        """Server message handler"""
        data = message.payload
        self._responses.put(data)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        data = self._serialize(obj)
        self._publish(topic=f"c2s/{self.id}", data=data)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get server data"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"
        data = self._responses.get()

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=self._server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if not self.closed:
            self._publish(topic=f"fin/{self.id}")
        super().close()
        self._responses.put(END_COMM)
