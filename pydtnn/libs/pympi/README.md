# pympi
Simple Python-based MPI implementation

## Example
```python
from pympi import MPI
comm = MPI.COMM_WORLD

message = comm.bcast("Hello, World!")

ranks = comm.allgather(comm.rank)
```