"""
Module for handling distributed synchronization operations in PyDTNN.
"""

import logging

import numpy as np

from pydtnn import MPI
from pydtnn.model.init import Init
from pydtnn.utils.constants import Array

__all__ = ("Sync",)

logger = logging.getLogger(__name__)


class Sync[T: Array](Init[T]):
    """
    Base class for distributed synchronization operations, providing methods for
    encoding, decoding, and performing collective communication reductions.
    """

    def _layer_reduce_encode(self, data: np.ndarray):
        """
        Prepares data for reduction by applying weights, quantization, and encryption.

        Args:
            data: The input array to be encoded.

        Returns:
            The processed array ready for synchronization.
        """
        data *= self.rank_weight

        if self.model_sync_quantize:
            data = np.astype(data, self.model_sync_dtype)

        if self.crypt:
            data = self.crypt.encrypt(data)  # type: ignore

        return data

    def _layer_reduce_decode(self, data) -> np.ndarray:
        """
        Decodes data after reduction by performing decryption and dequantization.

        Args:
            data: The reduced data to be decoded.

        Returns:
            The decoded numpy array.
        """

        if self.crypt:
            data = self.crypt.decrypt(data)

        if self.model_sync_quantize:
            data = np.astype(data, self.dtype)

        return data

    def _layer_reduce_sync(self, data: np.ndarray) -> np.ndarray:
        """
        Performs a synchronous all-reduce operation across the communicator.

        Args:
            data: The data array to reduce.

        Returns:
            The reduced data array.
        """
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            data = self.comm.allreduce(data, op=MPI.SUM)
        return data

    def _layer_reduce_async(self, data):
        """
        Initiates an asynchronous all-reduce operation.

        Args:
            data: The data array to reduce.

        Returns:
            The MPI request object for the operation.
        """
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            req = self.comm.Iallreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            req = self.comm.iallreduce(data, op=MPI.SUM)
        return req

    def _layer_reduce_wait(self, data, request):
        """
        Waits for the completion of an asynchronous reduction operation.

        Args:
            data: The original data buffer.
            request: The MPI request object to wait for.

        Returns:
            The reduced data array.
        """
        if (response := request.wait()) is not None:
            data = response
        return data
