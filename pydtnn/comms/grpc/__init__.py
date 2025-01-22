"""gRPC communications"""

import sys
from pydtnn import comms

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import grpc  # noqa: F401
finally:
    sys.path.insert(0, _pkg)


class Base(comms.Communication):
    """Shared base gRPC implementation"""
    _protocol_port = 8080
    _compression = grpc.Compression.NoCompression
