# pympi
Simple Python-based MPI implementation

## Example
```python
from pympi import MPI
comm = MPI.COMM_WORLD

message = comm.bcast("Hello, World!")

ranks = comm.allgather(comm.rank)
```

## Documentation

## Implementation
Due to how Python and external libraries handle threading, there is no reliable way to track when the MPI context should be automatically finalized.

MPI for Python finalizes its context via a atexit handler, which waits for all non-daemon threads to finish before automatically finalizing (if not disabled).

This implementation attempts to finalizes its context specifically when the main thread finishs. As other threads might be internal implementation details and might be required during the finalization stage. Waiting for a atexit handler might lead to those internal threads being reaped, therefor being unable to gracefully finalize.

If this approach is not plausible, as it is Python implementation dependant, a classic atexit handler is also registered, but as mention before, this might lead to an undefined finaliztion behaviour.

To ensure defined gracefull finalization the only reliable method is to manually call Finalize, and ensure it is always called, even when exceptions might be unhandled and lead to thread termination.