import copy

import net_queue as comms
from net_queue import CommunicatorOptions
from net_queue.io_stream import Serializer
from pympi import rc as mpi_rc, protocol as mpi_comm


def comm_options(base: CommunicatorOptions = CommunicatorOptions()):
    """Generate MPI specific comunicator options"""
    netloc = comms.NetworkLocation(host=mpi_rc.addr, port=mpi_rc.port)
    serialization_restrict = (*mpi_comm.SERIALIZABLE, *mpi_rc.serial) if mpi_rc.serial else None
    serialization = comms.SerializationOptions(load=Serializer(restrict=serialization_restrict).load)
    security = comms.SecurityOptions(key=mpi_rc.ssl_key, cert=mpi_rc.ssl_cert) if mpi_rc.ssl else None
    return copy.replace(base, netloc=netloc, serialization=serialization, security=security)
