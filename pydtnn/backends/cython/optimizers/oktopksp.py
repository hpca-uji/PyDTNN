"""Cython-accelerated implementation of the OkTopkSP optimizer for PyDTNN."""

import logging
from typing import TYPE_CHECKING

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.cython.optimizers.optimizer import OptimizerCython
from pydtnn.backends.cython.utils.oktopk_cython import (compute_dense_acc_cython,
                                                        intersect_1d_indexes_cython,
                                                        reset_residuals_cython,
                                                        update_sparsed_weights_cython,
                                                        update_sparsed_weights_mv_cython)
from pydtnn.backends.numpy.optimizers.oktopksp import OkTopkSPNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.sparse.sparse import SparseMatrixFlat

__all__ = ("OkTopkSPNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class OkTopkSPCython(OkTopkSPNumpy, OptimizerCython):
    """
    Cython-optimized version of the OkTopkSP optimizer.

    Inheriting from both the NumPy implementation and the Cython optimizer base class.
    """

    def _compute_acc(
        self, residuals: np.ndarray, dw: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """
        Compute acc, where: acc = residuals + (learning_rate * dw)

        Parameters:
            residuals (np.array): 2D dense matrix with the current layer residuals
            dw (np.array): 2D dense matrix with the current layer gradients
            learning_rate (float): learning rate float value

        Warning:
            'cython' method does not provide the same exact accuracy as 'numpy'.

        Returns:
            acc (np.array): 2D dense matrix with the updated residuals
        """

        self._show_message_only_once(
            "\n\nIn '_compute_acc', the method that it is being used is 'cython'"
        )
        acc = np.empty_like(dw)  # TODO: move to model init.
        compute_dense_acc_cython(residuals, dw, acc, learning_rate)

        return acc

    def _reset_residuals(self, acc: np.ndarray, indexes: np.ndarray) -> np.ndarray:
        """
        Update residuals.

        Set zero value if it is in indexes, else acc value is set.
        If density is 100% and some gradients are zero, scipy will
        be removing those indexes even if no sparsity is applied.
        Thus, to simulate 100% density, residuals must be always zero.
        This means that a slightly sparse factor will may remove more
        values because the gradients are already zero.

        Parameters:
            acc (np.array): 2D dense matrix
            indexes (tuple(np.array, np.array)): a tuple with rows and cols
            method (string, optional): The method to use for updating the weights.
            It can be 'cython' or 'numpy'. Default is 'cython'.

        Returns:
            residuals (np.array): which is the same as acc with the values in indexes set to zero.
        """

        self._show_message_only_once(
            "In '_reset_residuals', the method that it is being used is 'cython'"
        )

        if self.density == 1:
            return np.zeros_like(acc)
        else:
            # TODO: CHECK THIS!!! (Must have always a cannonical format)
            assert self._has_canonical_format(indexes)
            reset_residuals_cython(acc, indexes)
            return acc

    def _update_weights(
        self, layer: Layerable, w_type: str, w: np.ndarray, coo_u: SparseMatrixFlat
    ) -> None:
        """
        Perform weight updates using Cython-accelerated sparse operations.

        Args:
            layer: The layer object to update.
            w_type: The attribute name of the weights.
            w: The dense weight matrix.
            coo_u: The sparse gradient update matrix.
        """

        self._show_message_only_once(
            f"In '_update_weights', the method that it is being used is '{
                self._update_weights_method
            }'"
        )

        if len(self.dw_original_shape) != 1:
            w = w.reshape(-1)
        update_sparsed_weights_cython(w, self.model.nprocs, coo_u.data, coo_u.indexes)
        if len(self.dw_original_shape) != 1:
            w = w.reshape(self.dw_original_shape)
        setattr(layer, w_type, w)

    def _update_weights_with_momentum(
        self, layer: Layerable, w_type: str, w: np.ndarray, coo_u: SparseMatrixFlat
    ) -> None:
        """
        Perform weight updates with velocity and momentum using Cython-accelerated operations.

        Args:
            layer: The layer object to update.
            w_type: The attribute name of the weights.
            w: The dense weight matrix.
            coo_u: The sparse gradient update matrix.
        """

        self._show_message_only_once(
            f"In '_update_weights', the method that it is being used is '{
                self._update_weights_method
            }'"
        )

        if self.momentum == 0:
            logger.warning(
                "If momentum is 0 use 'cython' method, it produces the same output but it is faster"
            )

        if len(self.dw_original_shape) != 1:
            w = w.reshape(-1)
        velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
        update_sparsed_weights_mv_cython(w, self.model.nprocs, coo_u.data, coo_u.indexes, velocity, self.momentum)
        if len(self.dw_original_shape) != 1:
            w = w.reshape(self.dw_original_shape)
        setattr(layer, w_type, w)
        setattr(layer, "velocity_%s" % w_type, velocity)

    def _intersect_indexes(
        self,
        local_indexes: np.ndarray,
        global_indexes: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the intersection of two sets of indices of 2D.

        The assertion statement is only executed when the script
        is not run in optimized mode (python3 -O script.py).
        Remember that '_has_canonical_format' should only be
        used for debugging/development purposes to assert that
        indexes are correct. Indexes in scipy are usually in
        canonical format, so it should not be necessary to evaluate
        the indexes format. When optimized mode is enabled
        (python3 -O script.py), the assert sentences are not computed.

        Parameters:
            local_indexes (np.array):
                a numpy array representing the indexes
            global_indexes (np.array):
                a numpy array representing the indexes

        Returns:
            intersected_indexes (np.array):
                A numpy array representing the common indices.

        Example:
            - local_indexes  = np.array([0, 1, 2, 3, 5, 8]
            - global_indexes = np.array([1, 5, 8, 13, 21]
            - output: array([1, 5, 9])
        """

        max_size = min(len(local_indexes), len(global_indexes))
        intersected_indexes = np.zeros(max_size, dtype=np.int32)

        return intersect_1d_indexes_cython(local_indexes, global_indexes, intersected_indexes)
