"""
Module for handling distributed synchronization operations in PyDTNN.
"""

import logging

import numpy as np

from pydtnn import MPI
from pydtnn.datasets.dataset import Dataset
from pydtnn.model.base import Base
from pydtnn.model.init import Init
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_MDL_EVENT_enum
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

    def _model_reduce_sync(self, gradient=True):
        """Performs a synchronous all-reduce operation on model weights or gradients."""
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
            layer.reduce_weights_sync(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_async(self, gradient=True):
        """Initiates an asynchronous all-reduce operation on model weights or gradients."""
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
            layer.reduce_weights_async(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_wait(self, gradient=True):
        """Waits for completion of pending asynchronous all-reduce operations."""
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.WAIT_DW)
            layer.wait_allreduce_async(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    # TODO: Modify the method's name.
    def _weight_update(self, gradient=True, blocking=True, pipeline=False):
        """Updates model weights or gradients based on the configured synchronization strategy."""
        if blocking:
            self._model_reduce_sync(gradient)
        elif pipeline:
            self._model_reduce_wait(gradient)
            self._model_reduce_async(gradient)
        else:
            self._model_reduce_async(gradient)
            self._model_reduce_wait(gradient)

    def _compute_rank_weight(self, mask: list[int], part: Dataset.Part) -> float:
        """Calculates the weight contribution of the current rank based on dataset participation."""
        match self.model_sync_participation:
            case Base.SyncParticipation.ALL:
                comm_nsamples = self.comm_nsamples[part]
            case Base.SyncParticipation.AVAIL2ALL:
                if mask[self.comm_rank]:
                    comm_nsamples = [nsamples for nsamples, mask in zip(self.comm_nsamples[part], mask) if mask]
                else:
                    return 0.0
            case _:
                raise ValueError(f"Model synchronization participation option '{self.model_sync_participation}' not recognized. Only recognized: {list(Base.SyncParticipation)}")

        min_nsamples, max_nsamples, total_nsamples = min(comm_nsamples), max(comm_nsamples), sum(comm_nsamples)
        comm_size = len(comm_nsamples)

        match self.model_sync_algo:
            case Base.SyncAlgorithm.AVG:
                return 1.0 / comm_size
            case Base.SyncAlgorithm.WAVG:
                return self.dataset._nsamples[part] / total_nsamples
            case Base.SyncAlgorithm.INVAVG:
                inverse_nsamples = min_nsamples + (max_nsamples - self.dataset._nsamples[part])
                return inverse_nsamples / total_nsamples
            case _:
                raise ValueError(f"Model synchronization algorithm option '{self.model_sync_algo}' not recognized. Only recognized: {list(Base.SyncAlgorithm)}")
