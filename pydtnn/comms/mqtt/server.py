"""MQTT server"""

import uuid
import threading
from queue import Empty, SimpleQueue

import paho.mqtt.client as mqtt_client

from pydtnn.comms.mqtt import Protocol
from pydtnn.comms import ResourceClosed, Message


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = b""


class Server(Protocol):
    """MQTT server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

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

        if not peers or len(peers) == len(self._requests):
            self._publish(topic="s2c", data=data)
        else:
            # TODO: Optimize peer groups
            for peer in peers:
                self._publish(topic=f"s2c/{peer}", data=data)

    def close(self) -> None:
        """Close the server"""
        if self.closed:
            return
        super().close()
        with self._lock:
            for queue in self._requests.values():
                queue.put(END_COMM)
