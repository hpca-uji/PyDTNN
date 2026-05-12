"""
Cython-accelerated implementation of the OkTopk optimizer for PyDTNN.
"""

import logging
import warnings
from typing import TYPE_CHECKING

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.cython.optimizers.optimizer import OptimizerCython
from pydtnn.backends.cython.utils.oktopk_utils_cython import compute_dense_acc_cython, reset_residuals_cython, update_sparsed_weights_cython, update_sparsed_weights_mv_cython
from pydtnn.backends.numpy.optimizers.oktopk import OkTopkNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.sparse.sparse import SparseMatrixCOO

__all__ = ("OkTopkNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class OkTopkCython(OkTopkNumpy, OptimizerCython):
    """
    Cython-optimized version of the OkTopk optimizer, inheriting from both
    the NumPy implementation and the Cython optimizer base class.
    """

    def _model_init(self, list_layers: list[Layerable]) -> None:
        """
        Initialize the model layers and configure the Cython weight update method.

        Args:
            list_layers: List of layers to be optimized.
        """
        super()._model_init(list_layers)
        self._update_weights_method = "cython"
        # method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        match self._update_weights_method:
            case "cython":
                self._update_weights = self._update_weights_cython
            case "cython_with_vel_and_momentum":
                self._update_weights = self._update_weights_with_vel_and_momentum
            case _:
                NotImplementedError(f'The weights update is not implemented for "{self._update_weights_method}" method.')

    def _compute_acc(self, residuals: np.ndarray, dw: np.ndarray, learning_rate: float) -> np.ndarray:
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

        self._show_message_only_once("\n\nIn '_compute_acc', the method that it is being used is 'cython'")

        return compute_dense_acc_cython(residuals, dw, learning_rate)

    def _reset_residuals(self, acc: np.ndarray, indexes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Update residuals: set zero value if it is in indexes, else acc value is set.
        If density is 100% and some gradients are zero, scipy will be removing those indexes even if no sparsity is applied.
        Thus, to simulate 100% density, residuals must be always zero.
        This means that a slightly sparse factor will may remove more values because the gradients are already zero.

        Parameters:
            acc (np.array): 2D dense matrix
            indexes (tuple(np.array, np.array)): a tuple with rows and cols
            method (string, optional): The method to use for updating the weights. It can be 'cython' or 'numpy'. Default is 'cython'.

        Returns:
            residuals (np.array): which is the same as acc with the values in indexes set to zero.
        """

        self._show_message_only_once("In '_reset_residuals', the method that it is being used is 'cython'")

        if self.density == 1:
            return np.zeros_like(acc)
        else:
            assert self._has_canonical_format(indexes)
            return reset_residuals_cython(acc, indexes[0], indexes[1])

    def _update_weights(self, layer: Layerable, w_type: str, w: np.ndarray, coo_u: SparseMatrixCOO) -> None:
        """
        Update weights: w -= (u / self.model.nprocs)
        and set to weight layer attribute: setattr(layer, w_type, w)

        Parameters:
            layer (int): layer id
            w_type (string): weight param type (bias, weight, ...)
            w (np.array): N dimensional dense weights matrix/tensor
            coo_u (SparseMatrixCOO): Sparse 2D gradient matrix in COO format to update w

        Returns:
            (void): instead it directly applies the result to the weight layer attribute
        """
        raise NotImplementedError("This is a fake method that must be replaced with the right one.")

    def _update_weights_cython(self, layer: Layerable, w_type: str, w: np.ndarray, coo_u: SparseMatrixCOO) -> None:
        """
        Perform weight updates using Cython-accelerated sparse operations.

        Args:
            layer: The layer object to update.
            w_type: The attribute name of the weights.
            w: The dense weight matrix.
            coo_u: The sparse gradient update matrix.
        """

        self._show_message_only_once(f"In '_update_weights', the method that it is being used is '{self._update_weights_method}'")

        if len(self.dw_original_shape) != 2:
            w = w.reshape(w.shape[0], -1)
        w = update_sparsed_weights_cython(w, coo_u.data, coo_u.row, coo_u.col)
        if len(self.dw_original_shape) != 2:
            w = w.reshape(self.dw_original_shape)
        setattr(layer, w_type, w)

    def _update_weights_with_vel_and_momentum(self, layer: Layerable, w_type: str, w: np.ndarray, coo_u: SparseMatrixCOO) -> None:
        """
        Perform weight updates with velocity and momentum using Cython-accelerated operations.

        Args:
            layer: The layer object to update.
            w_type: The attribute name of the weights.
            w: The dense weight matrix.
            coo_u: The sparse gradient update matrix.
        """

        self._show_message_only_once(f"In '_update_weights', the method that it is being used is '{self._update_weights_method}'")

        if self.momentum == 0:
            logger.warning("If momentum is 0 use 'cython' method, it produces the same output but it is faster")
            warnings.warn("If momentum is 0 use 'cython' method, it produces the same output but it is faster", RuntimeWarning)

        if len(self.dw_original_shape) != 2:
            w = w.reshape(w.shape[0], -1)
        velocity = getattr(layer, "velocity_%s" % w_type, np.zeros_like(w, dtype=layer.model.dtype))
        w, velocity = update_sparsed_weights_mv_cython(w, coo_u.data, coo_u.row, coo_u.col, velocity, self.momentum)
        if len(self.dw_original_shape) != 2:
            w = w.reshape(self.dw_original_shape)
        setattr(layer, w_type, w)
        setattr(layer, "velocity_%s" % w_type, velocity)
