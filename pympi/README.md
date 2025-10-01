# pympi
Python-based MPI implementation for reasearch and experimentation

## Example
```python
# example.py
from pympi import MPI
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# Synchronization
comm.barrier()

# Collective
message = comm.bcast("Hello, World!")

ranks = comm.allgather(rank)

# Point-to-point
match rank:
    case 0:
        comm.send("Hello, 1!", dest=1)
    case 1:
        comm.recv(source=1)

# Nonblocking
req = comm.ireduce(rank)

result = req.wait()

# Buffer-based
buffer = bytearray(10)

comm.Allreduce(MPI.IN_PLACE, buffer)
```

Execute with 2 processes:
```bash
python -m pympi -np 2 python example.py
```

Note: `mpirun`, `srun` and other job managers can also be used*

## Limitations
Note: all the features mentioned here are planned for implementation,
however they are not currently available.

For comunicators, only `COMM_WORLD` is supported.
`COMM_SELF`, `COMM_NULL`, subcommunicators and others can not be created.

Buffer-based operations are limited to the ones which the send and receive buffer are equally sized.

For point-to-point comuncations, only `send`, and its asynchronous and buffer-based variants, is supported.
`ssend`, `bsend` and `rsend` can not be used. Additionally, self-messaging and message tagging are also not supported.

## Implementation
For most cases `pympi` is a drop-in replacement for `mpi4py`,
therefore only mayor diferences will be documented.

Documentation for `mpi4py` available here: <https://mpi4py.readthedocs.io/en/stable/>

However, unlike `mpi4py`, asynchronous respones will be available without needing to call `wait` on them.

---

For communications `net-queue` is used,
options can be customized via the `rc` module and enviroment variables.

Documentation for `net-queue` available here: _Publish pending_

Following it's memory handeling methodology, no internal buffering is currently preformed,
so requests that typically block until a local copy is done, will block until the data is fully transmitted.
This is only meaningfully observed on the point-to-point `send`, elsewhere this difference is negligible.

---

`pympi` follows a client-server architecture, by default rank `0` lauches the server on a separed thread.

The server can also be started externally by:
```bash
python -m pympi.server
```

## Personalized operations
Unlike traditional MPI and `mpi4py`,
with `pympi` you can define fully personalized operations,
not just reduce functions.

For example, on this operation we group ranks with its neighbour in pairs,
apply a custom reduce function, and return the result to the lower rank.

```python
# pair_reduce.py
from pympi import MPI, proto
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

# Define operation
@dataclass(slots=True, frozen=True)
class PairReduce(proto.OperationContext):
    reducer: Callable

    def group(self, size: int) -> proto.CommmunicationGroup:
        src = range(size)
        dst = range(0, size, 2) # half of ranks
        return proto.CommmunicationGroup(src, dst)

    def apply(self, src: Mapping[Rank, int], dst: Set[Rank]) -> Mapping[Rank, int]:
        values = [src[rank] for rank in sorted(src)]  # sort values by rank
        results = map(self.reducer, itertools.batched(values, 2))
        return dict(zip(dst, results))

# Inputs
value = rank
print(f"R{rank}: {value}")
# R0: 0
# R1: 1
# R2: 2
# R3: 3

# Execute operation
ctx = PairReduce(reducer=sum)
group = ctx.group(size=comm.size)
op = proto.OperationRequest(group, ctx, value)
result = comm.submit(op).wait()

# Outputs
print(f"R{rank}: {result}")
# R0: 1
# R1: None
# R2: 5
# R3: None
```

## Documentation
### Constatns
- `rc.init: bool = True`

  Should server auto initialize

  Environment variables checked for defaults:
  - `PYMPI_ADDR` (if not defined)

- `rc.wait: float = 0.5` (seconds)

  Wait for auto server transitions

  If `init` is `False`, `wait` is ignored.

  Environment variables checked for defaults:
  - `PYMPI_WAIT`

- `rc.addr: str = "127.0.0.1"`

  Server address

  Environment variables checked for defaults:
  - `PYMPI_ADDR`

- `rc.port: int = 61642`

  Server port

  Environment variables checked for defaults:
  - `PYMPI_PORT`

- `rc.size: int = 1`

  Communication size

  Environment variables checked for defaults:
  - `PYMPI_SIZE`
  - `OMPI_COMM_WORLD_SIZE`
  - `PMI_SIZE`
  - `SLUM_NPROCS`

- `rc.rank: int = 0`

  Communication identifier

  Environment variables checked for defaults:
  - `PYMPI_RANK`
  - `OMPI_COMM_WORLD_RANK`
  - `PMI_RANK`
  - `SLUM_PROCID`

- `rc.serial: Iterable[str] = []`

  Serializable global names

  Environment variables checked for defaults:
  - `PYMPI_SERIAL` (comma separted list of global names)

  See `nq.io_stream.Serializer(restrict)` for more information.

- `rc.proto: nq.Protocol = Protocol.TCP`

  Communication protocol

  Environment variables checked for defaults:
  - `PYMPI_PROTO`

- `rc.ssl: bool = False`

  Use secure communications

  Environment variables checked for defaults:
  - `PYMPI_SSL`

- `rc.ssl_key: Path | None = None`

  Secure communications private key

  Environment variables checked for defaults:
  - `PYMPI_SSL_KEY`

- `rc.ssl_cert: Path | None = None`

  Secure communications certificate chain

  Environment variables checked for defaults:
  - `PYMPI_SSL_CERT`

- `rc.comm: nq.CommunicatorOptions = nq.CommunicatorOptions()`

  Additional net-queue communicator options

### Structures
- `proto.CommmunicationGroup(...)`

  Communication group

  - `src: frozenset[Rank]`
  - `dst: frozenset[Rank]`

- `OperationContext(...)`

  Abstract dataclass base for operation contexts

  - `group(...) -> CommmunicationGroup`

    Compute operation's communication group

  - `apply(src: Mapping[Rank, Any], dst: Set[Rank]) -> Mapping[Rank, Any]`

    Apply operation over objects

- `proto.OperationRequest(...)`

  Operation request

  - `group: CommmunicationGroup`
  - `ctx: OperationContext | None = None`
  - `data: typing.Any | None = None`

  ---

  - `id: uuid.UUID = uuid.uuid4()` (random)

### Classes
- `Comm(comm_options)`

  Communicator

  - `comm_options: nq.ComunicatorOptions = nq.ComunicatorOptions()`

  ---

  - `submit(op, callback) -> Request`

    Schedule a new operation

    Operation can be a user defined instance, allowing for custom operations.

    - `op: proto.OperationRequest`
    - `callback: abc.Callable = lambda x: x`

      Called with the result of the operation prior to resolving the request, the return value of this function is the one provided to `wait`.

- `server.Server(thread_pool, comm_options)`

  MPI server

  - `thread_pool: concurrent.futures.ThreadPoolExecutor`
  - `comm_options: nq.ComunicatorOptions = nq.ComunicatorOptions()`

## Notes
Due to how Python and external libraries handle threading, there is no reliable way to track when the MPI context should be automatically finalized.

MPI for Python finalizes its context via a atexit handler, which waits for all non-daemon threads to finish before automatically finalizing (if not disabled).

This implementation attempts to finalizes its context specifically when the main thread finishs. As other threads might be internal implementation details and might be required during the finalization stage. Waiting for a atexit handler might lead to those internal threads being reaped, therefor being unable to gracefully finalize.

If this approach is not plausible, as it is Python implementation dependant, a classic atexit handler is also registered, but as mention before, this might lead to an undefined finaliztion behaviour.

To ensure defined gracefull finalization the only reliable method is to manually call Finalize, and ensure it is always called, even when exceptions might be unhandled and lead to thread termination.