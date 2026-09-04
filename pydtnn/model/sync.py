"""Module for handling distributed synchronization operations in PyDTNN."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from pydtnn import MPI
from pydtnn.datasets.abstract import Dataset
from pydtnn.model.base import SyncAlgorithm, SyncParticipation
from pydtnn.model.state import State
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT,
                                   PYDTNN_MDL_EVENTS, MdlEventEnum)
from pydtnn.utils.constants import Array, SyncMode

if TYPE_CHECKING:
    from pympi.MPI import Request

__all__ = ("Sync",)

logger = logging.getLogger(__name__)


class Sync[T: Array](State[T]):  # noqa: D101 (generics not detected)
    """
    Base class for distributed synchronization operations, providing methods for
    encoding, decoding, and performing collective communication reductions.
    """

    def _model_init(self) -> None:
        super()._model_init()
        if self.use_blocking_mpi:
            self._model_sync = self._model_sync_blocking_mpi
        elif self.parallel_pipeline:
            self._model_sync = self._model_sync_parallel_pipeline
        else:
            self._model_sync = self._model_sync_non_blocking

    def _layer_reduce_encode(self, data: np.ndarray) -> np.ndarray:
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
            data = self.crypt.encrypt(data)  # pyright: ignore[reportAssignmentType]

        return data

    def _layer_reduce_decode(self, data: np.ndarray) -> np.ndarray:
        """
        Decodes data after reduction by performing decryption and dequantization.

        Args:
            data: The reduced data to be decoded.

        Returns:
            The decoded numpy array.
        """

        if self.crypt:
            data = self.crypt.decrypt(data)  # pyright: ignore[reportArgumentType]

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

    def _layer_reduce_async(self, data: np.ndarray) -> Request:
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

    def _layer_reduce_wait(self, data: np.ndarray, request: Request) -> np.ndarray:
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

    def _model_reduce_sync(self, parameters: SyncMode) -> None:
        """Performs a synchronous all-reduce operation on model weights or gradients."""
        for layer in self.layers:
            self.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.ALLREDUCE_DW
            )
            layer.state_reduce_sync(mode=parameters)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_async(self, parameters: SyncMode) -> None:
        """Initiates an asynchronous all-reduce operation on model weights or gradients."""
        for layer in self.layers:
            self.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.ALLREDUCE_DW
            )
            layer.state_reduce_async(mode=parameters)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_wait(self, mode: SyncMode) -> None:
        """Waits for completion of pending asynchronous all-reduce operations."""
        for layer in self.layers:
            self.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.WAIT_DW
            )
            layer.state_reduce_wait(mode=mode)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_sync(self, mode: SyncMode) -> None:
        """Updates model weights or gradients based on the configured synchronization strategy."""
        raise NotImplementedError("This is a fake method, use an actual _model_sync_* method")

    def _model_sync_blocking_mpi(self, mode: SyncMode) -> None:
        """Updates the weights or gradients with a blocking MPI reduction."""
        self._model_reduce_sync(mode)

    def _model_sync_parallel_pipeline(self, mode: SyncMode) -> None:
        """Updates the weights or gradients using a parallel pipeline."""
        self._model_reduce_wait(mode)
        self._model_reduce_async(mode)

    def _model_sync_non_blocking(self, mode: SyncMode) -> None:
        """Updates the weights or gradients using a non-blocking parallel pipeline."""
        self._model_reduce_async(mode)
        self._model_reduce_wait(mode)

    def _compute_rank_weight(self, mask: list[int], part: Dataset.Part) -> float:
        """Calculates the weight contribution of the current rank based on dataset participation."""
        match self.model_sync_participation:
            case SyncParticipation.ALL:
                comm_nsamples = self.comm_nsamples[part]
            case SyncParticipation.AVAIL2ALL:
                if mask[self.comm_rank]:
                    comm_nsamples = [
                        nsamples for nsamples, mask in zip(self.comm_nsamples[part], mask) if mask
                    ]
                else:
                    return 0.0
            case _:
                raise ValueError(
                    f"Model synchronization participation option {self.model_sync_participation} not recognized."
                    f" Only recognized: {list(SyncParticipation)}"
                )

        min_nsamples, max_nsamples, total_nsamples = (
            min(comm_nsamples),
            max(comm_nsamples),
            sum(comm_nsamples),
        )
        comm_size = len(comm_nsamples)

        match self.model_sync_algo:
            case SyncAlgorithm.AVG:
                return 1.0 / comm_size
            case SyncAlgorithm.WAVG:
                return self.dataset._nsamples[part] / total_nsamples
            case SyncAlgorithm.INVAVG:
                inverse_nsamples = min_nsamples + (max_nsamples - self.dataset._nsamples[part])
                return inverse_nsamples / total_nsamples
            case _:
                raise ValueError(
                    f"Model synchronization algorithm option {self.model_sync_algo} not recognized."
                    f" Only recognized: {list(SyncAlgorithm)}"
                )
