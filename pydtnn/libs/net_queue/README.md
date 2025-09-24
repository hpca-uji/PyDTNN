# net-queue
Easy & fast network communications using queues

## Example
```python
# server.py
import net_queue as nq

with nq.new(purpose=nq.Purpose.SERVER) as queue:
    message = queue.get()
    queue.put("Hello, Client!")
```

```python
# client.py
import net_queue as nq

with nq.new(purpose=nq.Purpose.CLIENT) as queue:
    queue.put("Hello, Server!")
    message = queue.get()
```

## Documentation


## Implementation
### gRPC
gRPC does not conform well to a async send & async receive model, it expects remote procedure calls to be called, processed and responded. To simulate this model we created a bidirectional streaming procedure. Sent data is queued at the server, recived data is polled until available.

Polling is implemented with a exponential backoff time and a limit. The gRPC library queues requests, so requests would always be replyed in a timely maner, but we do not want to hogh the CPU or network with usesless requests.

It is important to not hold the prodedures indefinitely, since this could starve the server of threads. Additionaly, if a streaming direction was already closed, messages could end up queued forever if not restarted.

### MQTT
MQTT broker implementations are not common, so the server provided here is actually another client. Therefore the address and port provided to both, the client and server, should be the one of the actual broker, not where the server is running.

The MQTT library handles comunications single-threaded, therefore operations on related callbacks are limited to pushing or pulling data from queues without blocking, so all operations are minimal and fast.

Peer-groups and global comunications are not optimized. First, chunked message ordering must be resolved. Single chunk order it is guaranteed by the protocol, even on with diferent topics. Second, peer-groups could be implemented using grouping requests that generate new UUID per group. This would reduce also reduce load on the broker.
