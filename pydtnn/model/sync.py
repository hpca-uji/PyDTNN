import logging

import numpy as np

from pydtnn import MPI
from pydtnn.model.init import Init
from pydtnn.utils.constants import Array

__all__ = ("Sync",)

logger = logging.getLogger(__name__)


class Sync[T: Array](Init[T]):
    def _layer_reduce_encode(self, data: np.ndarray):
        data *= self.rank_weight

        if self.model_sync_quantize:
            data = np.astype(data, self.model_sync_dtype)

        if self.crypt:
            data = self.crypt.encrypt(data)  # type: ignore

        return data

    def _layer_reduce_decode(self, data) -> np.ndarray:

        if self.crypt:
            data = self.crypt.decrypt(data)

        if self.model_sync_quantize:
            data = np.astype(data, self.dtype)

        return data

    def _layer_reduce_sync(self, data: np.ndarray) -> np.ndarray:
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            data = self.comm.allreduce(data, op=MPI.SUM)
        return data

    def _layer_reduce_async(self, data):
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            req = self.comm.Iallreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            req = self.comm.iallreduce(data, op=MPI.SUM)
        return req

    def _layer_reduce_wait(self, data, request):
        if (response := request.wait()) is not None:
            data = response
        return data
