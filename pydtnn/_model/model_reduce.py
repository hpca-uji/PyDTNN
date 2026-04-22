from typing import TYPE_CHECKING
from warnings import warn

from pydtnn._model.model_base import Model_Base as Model
from pydtnn.utils.constants import Array
import numpy as np
from pydtnn import MPI

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import polyhe
else:
    try:
        import polyhe
    except Exception:
        polyhe = None

class Model_Reduce[T: Array](Model[T]):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Encryption
        if self.encryption_name:
            self.crypt = self._crypt_init(self.encryption_name)
        else:
            self.crypt = None
    
    def _crypt_init(self, encryption_name: str) -> "polyhe.Context":
        """Initialize encryption context"""
        if polyhe is None:
            raise RuntimeError("uHE is not avaliable, but is requiested!")

        backend = polyhe.Backend(encryption_name)
        options = polyhe.Options(
            slots=self.encryption_slots,
            scale=self.encryption_scale,
            security=self.encryption_security
        )

        if self.comm_rank == 0:
            crypt = polyhe.new(backend, options)

        if self.comm:
            crypt = self.comm.bcast(crypt if self.comm_rank == 0 else None)

        assert crypt is not None
        if self.enable_nccl:
            warn_text = "If NCCL is active, encryption is disabled"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)

        return crypt
    # -----

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