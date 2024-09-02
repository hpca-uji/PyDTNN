#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from scipy.sparse import coo_matrix

from pydtnn.cython_modules import top_threshold_selection_cython, flattened_top_threshold_selection_cython
from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import SGD_OkTopk

try:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from mpi4py import MPI
except (ImportError, ModuleNotFoundError):
    pass

class SGD_OkTopkCPU(OptimizerCPU, SGD_OkTopk):


    def update(self, layer, **kwargs):
        current_batch = kwargs.get("current_batch", None)

        if layer.id not in self.residuals:
            self.residuals[layer.id] = {dw_: None for dw_ in layer.grad_vars.values()}

        # TODO: This variable and the if else in the for, should be removed. Use only for testing.
        #       "weights", "biases", "beta", "gamma" should always be trained using oktopk, not SGD.
        
        oktopk_trainable_params = ["weights"] 
         
        for w_, dw_ in layer.grad_vars.items():
            if w_ in oktopk_trainable_params:
                w, dw = getattr(layer, w_), getattr(layer, dw_)
                if self.residuals[layer.id][dw_] is None:
                    self.residuals[layer.id][dw_] = np.zeros_like(w, dtype=layer.model.dtype)

                acc = self.residuals[layer.id][dw_] + (self.learning_rate * dw)
                u, indexes = self._ok_sparse_allreduce(acc, current_batch, self.k)
                self.residuals[layer.id][dw_] = self._update_residuals(acc, indexes)
                w = w - u / self.nprocs
                # w[indexes] = u[indexes] / self.nprocs
                # w[indexes] = u / self.nprocs
                setattr(layer, w_, w)
            else:
                lr = self.learning_rate
                w, dw = getattr(layer, w_), getattr(layer, dw_)
                velocity = getattr(layer, "velocity_%s" % w_, np.zeros_like(w, dtype=layer.model.dtype))

                velocity = self.momentum * velocity + dw
                if self.nesterov:
                    w -= lr * (self.decay * w + dw + self.momentum * velocity)
                else:
                    w -= lr * (self.decay * w + velocity)

                setattr(layer, w_, w)
                setattr(layer, "velocity_%s" % w_, velocity)


    def _update_residuals(self, acc, indexes, method="numpy"):
        """
        Returns the residuals: set zero value if it is in indexes, else acc value is set.

        Parameters:
            - acc: gradient accumulation values (tensor)
            - indexes: set of tuples representing multi-dimensional topk indexes

        Returns:
            - residuals
        
        Example:
            - acc = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            - indexes = (array([0, 2]), array([2, 1]))
            - output: [[1, 2, 0], [4, 5, 6], [7, 0, 9]]
        """

        if method == "numpy":
            residuals = np.array(acc)
            residuals[indexes] = 0
            return residuals

    
    def _ok_sparse_allreduce(self, acc, t, k, space_repartition_t=64, thresholds_re_evaluation_t=32):
        """
        Performs the Ok-Topk sparse allreduce operation. 
        This method executes the Ok-Topk sparse allreduce algorithm, which 
        optimizes communication by only exchanging the most significant 
        gradient values (top-k) across distributed processes. The method 
        periodically re-evaluates the thresholds and repartitions the 
        gradient space to maintain efficiency and accuracy.

        Parameters:
            - acc: Gradient matrix accumulation values.
            - t: Current iteration number.
            - k: Number of top-k gradient values to select.
            - space_repartition_t: Interval of iterations for space repartitioning.
            - thresholds_re_evaluation_t: Interval of iterations for threshold re-evaluation.

        Returns:
            - u: The updated gradient values after sparse allreduce.
            - indexes: The indices of the top-k gradient values that were updated.
        """

        if t % thresholds_re_evaluation_t == 0:
            self.local_th = self._th_re_evaluate(acc, k)
        
        if t % space_repartition_t == 0:
            self.boundaries = self._space_repartition(acc, self.local_th)

        reduced_topk, local_topk_indexes = self._split_and_reduce(acc, self.local_th, self.boundaries)
        
        if t % thresholds_re_evaluation_t == 0:
            all_reduced_topk = self._allgather(reduced_topk)
            self.global_th = self._th_re_evaluate(all_reduced_topk, k)

        u, global_topk_indexes = self._balance_and_allgather(reduced_topk, self.global_th)
        indexes = self.intersect_indexes(local_topk_indexes, global_topk_indexes, acc.shape)
        return u, indexes


    def _th_re_evaluate(self, acc, k, method="numpy"):
        """
        Return the absolute gradient threshold of acc tensor

        Parameters:
            - acc: gradient dense tensor
            - k: selection values hyperparameter

        Returns:
            - threshold: absolute gradient threshold k 
        """

        if method == "numpy":
            sorted_acc = np.sort(np.abs(acc).flatten())
            threshold = sorted_acc[max(-k, -len(sorted_acc))]
            return threshold


    def _space_repartition(self, acc, local_th, method="naive"):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        Parameters:
            - acc: gradient matrix values
            - local_th: local process gradient threshold

        Returns:
            - boundaries: [(row_start, row_end), ...]
                where row_start is included and row_end is excluded.
        """
    
        boundaries = []

        if method == "naive":
            total_rows = acc.shape[0]
            region_size = total_rows // self.nprocs
            for proc in range(self.nprocs):
                row_start = proc * region_size
                row_end = row_start + region_size
                if proc == self.nprocs - 1:  
                    row_end = total_rows
                boundaries.append((row_start, row_end))

        elif method == "balanced":
            # TODO 
            pass

        return boundaries


    def _split_and_reduce(self, acc, local_th, boundaries, method="allreduce"):
        """
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts
        the reduction locally. 

        Parameters:

            - acc: Gradient matrix accumulation values.
            - local_th: Local threshold for selecting top-k values.
            - boundaries: Boundaries for partitioning the gradient space: { proc_id : (row_start, row_end) }.

         Returns:
            - reduced_topk: The reduced top-k gradient values.
            - local_topk_indexes: The indices of the top-k gradient values selected locally.
        """


        if method == "allreduce":
            local_topk, local_topk_indexes = self._top_threshold_selection(acc, local_th)
            reduced_topk = self._allreduce(local_topk)
            return reduced_topk, local_topk_indexes


        elif method == "boundaries":
            local_topk, local_topk_indexes = self._top_threshold_selection(acc, local_th)
            reduced_topk = self._p2p_reduce_topk(local_topk, boundaries)
            return reduced_topk, local_topk_indexes


    def _balance_and_allgather(self, reduced_topk, global_th, method="allreduce"):
        
        if method == "allreduce":
            # If split_and_reduce phase was performed with allreduce:
            global_topk, global_topk_indexes = self._top_threshold_selection(reduced_topk, global_th)
            return global_topk, global_topk_indexes

        elif method == "boundaries":
            # 1. Global topk selection
            global_topk, global_topk_indexes = self._top_threshold_selection(reduced_topk, global_th)

            # 2. Data packaging
            # TODO

            # 3. Data balancing
            # TODO

            # 4. Allgatherv using recursive doubling
            self._allgather(data)
            return global_topk, global_topk_indexes


    def intersect_indexes(self, local_indexes_tuple, global_indexes_tuple, shape, method="numpy"):
        """
        Calculates the intersection of two sets of indices of any dimension.

        Parameters:
            - local_indexes_tuple: a tuple of N arrays of shape M. 
            - global_indexes_tuple: a tuple of N arrays of shape M. 
                where N is the dimensions and M the number of indexes.
        
        Returns:
            - Set of tuples representing the common indices in all dimensions.
        
        Example:
            - local_indexes_tuple = (np.array([0, 3, 2]), np.array([2, 4, 1]), np.array([5, 1, 5]))
            - global_indexes_tuple = (np.array([1, 2, 3]), np.array([3, 1, 4]), np.array([5, 6, 1]))
            - output: (array([3]), array([4]), array([1]))
        """

        if method == "numpy":
            local_flattened_indexes = np.ravel_multi_index(local_indexes_tuple, dims=shape)
            global_flattened_indexes = np.ravel_multi_index(global_indexes_tuple, dims=shape)
            flattened_intersection = np.intersect1d(local_flattened_indexes, global_flattened_indexes)
            unravel_intersection = np.unravel_index(flattened_intersection, shape=shape)
            return unravel_intersection
            

    def _top_threshold_selection(self, tensor, threshold, method="numpy"):
        """
        Selects top-k elements from the tensor that are greater than or equal to the threshold.
        
        Parameters:
            - tensor: The input tensor from which to select elements.
            - threshold (float): The threshold value to compare against the absolute values of the tensor elements.
        
        Returns:
            - topk: An array with the elements that meet the threshold condition (coo)
                    A sparse tensor with the elements thth meet the threshold condition (not coo)
            - topk_indexes: a tuple of np.arrays with the indexes, e.g (array([0, 1]), array([1, 1]))
        """

        topk, topk_indexes = None, None

        if method == "numpy":
            topk = np.zeros_like(tensor, dtype=tensor.dtype)
            topk_indexes = np.where(np.abs(tensor) >= threshold)
            topk[topk_indexes] = tensor[topk_indexes]

        elif method == "cython":
            topk, topk_indexes = top_threshold_selection_cython(tensor, threshold)

        elif method == "cython_flattening":
            topk, topk_indexes = flattened_top_threshold_selection_cython(tensor.flatten(), threshold)
            topk = np.reshape(topk, tensor.shape)
            topk_indexes = np.unravel_index(topk_indexes, tensor.shape)
        
        elif method == "numpy_coo":
            topk_indexes = np.where(np.abs(tensor) >= threshold)
            topk = tensor[topk_indexes]

        return topk, topk_indexes


    def _p2p_reduce_topk(self, topk, boundaries, method="dense"):
        """
        TODO: To simplify the implementation, destination rotation and bucketing,
        is not yet implemented.

        """
        if self.nprocs == 1:
            return topk
        
        if method == "dense":
            reduced_topk = None
            for region, (row_start, row_end) in enumerate(boundaries):
                if self.rank == region:
                    reduced_topk = self.comm.reduce(topk[row_start:row_end], op=MPI.SUM, root=region)
                else:
                    self.comm.reduce(topk[row_start:row_end], op=MPI.SUM, root=region)
            return reduced_topk

        elif method == "sparse":
            reduced_topk = None
            #TODO
            return reduced_topk 


    def _allreduce(self, data, op=MPI.SUM):
        if self.nprocs == 1:
            return data

        data = self.comm.allreduce(data, op=op)
        return data


    def _allgather(self, data):
        if self.nprocs == 1:
            return data
        
        data = self.comm.allgather(data)
        return data

