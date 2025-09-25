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


## Notes
### Communication conventions
- ini: connection start (identify)
- fin: connection stop  (flush)
- com: message exchange (generic)
- c2s: message exchange (client -> server)
- s2c: message exchange (server -> client)

### Communication handshakes
Ini:
- Server & client sends ID
- Server & client wait for ID
- Server create session or continues session

Fin:
- Server & client flushes message queue
- Server & client sends empty message
- Server & client wait for empty message

### Communication persistency
Ini:
- Must be done on first or changing connection

Fin:
- Must be done on session end (not connection)

### Communication contract
Constructor
- Never blocks
- Only one communicator per ID
- Reusing ID retain server queues

Put
- Never blocks
- Communication will not modify object
- Consumer must not modify object util future resolved
- Resolved futures acknowledge peer reception
- Cancelled futures indicates peer diconnected

Get
- Always block
- Returns a message or raises ResourceClosed
- Once closed it continues working until exhausted then it raises ResourceClosed

Close
- Always block
- Server waits for peers to disconnect

### gRPC
gRPC does not conform well to a async send & async receive model, it expects remote procedure calls to be called, processed and responded. To simulate this model we created a bidirectional streaming procedure. Sent data is queued at the server, recived data is polled until available.

Polling is implemented with a exponential backoff time and a limit. The gRPC library queues requests, so requests would always be replyed in a timely maner, but we do not want to hogh the CPU or network with usesless requests.

It is important to not hold the prodedures indefinitely, since this could starve the server of threads. Additionaly, if a streaming direction was already closed, messages could end up queued forever if not restarted.

### MQTT
MQTT broker implementations are not common, so the server provided here is actually another client. Therefore the address and port provided to both, the client and server, should be the one of the actual broker, not where the server is running.

The MQTT library handles comunications single-threaded, therefore operations on related callbacks are limited to pushing or pulling data from queues without blocking, so all operations are minimal and fast.

Peer-groups and global comunications are not optimized.

First, chunked message ordering must be resolved. Single chunk order it is guaranteed by the protocol, even on with diferent topics. Second, peer-groups could be implemented using grouping requests that generate new UUID per group. This would reduce also reduce load on the broker.

## Planned
Implement reconnection support. The protocol already has support for it, server support is done, clients can reconnect but can not yet disconnect without flushing.

Implement two-way connection expiration and keep-alives. There is no reliable way to track connection drops between communication implementations. Most of them end up with memory leaks. If desired expiration periods could be long and automatic client reconnections could be allowed, enabling MQTT-like reliability without the cost.