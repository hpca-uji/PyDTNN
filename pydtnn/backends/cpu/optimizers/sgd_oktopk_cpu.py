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

        for w_, dw_ in layer.grad_vars.items():
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


    def _update_residuals(self, acc, indexes):
        """
        Returns the residuals: set zero value if it is in indexes, else acc value is set.

        Parameters:
            - acc: gradient matrix accumalation values
            - indexes: topk indexes

        Returns:
            - residuals
        
        Example:
            - acc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            - indexes = [2, 5, 9]
            - output: [1 2 0 4 5 0 7 8 9 0]
        """
        residuals = np.zeros_like(acc, dtype=acc.dtype)
        if len(indexes) > 0:
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
        local_topk_indexes_set, global_topk_indexes_set = set(local_topk_indexes), set(global_topk_indexes)
        indexes = np.array(list(local_topk_indexes_set.intersection(global_topk_indexes_set)))
        return u, indexes


    def _th_re_evaluate(self, acc, k):
        """
        Return the absolute gradient threshold of acc matrix

        Parameters:
            - acc: gradient dense matrix
            - k: selection values hyperparameter

        Returns:
            - threshold: absolute gradient threshold k 
        """

        sorted_acc = np.sort(np.abs(acc).flatten())
        if len(sorted_acc) > k:
            threshold = sorted_acc[-k]
        else:
            threshold = sorted_acc[0]
        return threshold


    def _space_repartition(self, acc, local_th):
        """
        Returns the boundaries of the regions of the gradient matrix for the split and reduce phase.
        
        TODO: Currently, the distribution of space is not balanced, 
        but static: the matrix is divided into P equal regions.
        The regions should not be static, they should be distributed 
        in a balance way regarding to topk values locations.

        Parameters:
            - acc: gradient matrix values
            - local_th: local process gradient threshold

        Returns:
            - boundaries: { proc_id : (row_start, row_end) }
                where row_start is included and row_end is excluded.
        """

        boundaries = {}
        total_rows = acc.shape[0]
        region_size = total_rows // self.nprocs

        for proc in range(self.nprocs):
            row_start = proc * region_size
            row_end = row_start + region_size
            if proc == self.nprocs - 1:  
                row_end = total_rows
            boundaries[proc] = (row_start, row_end)
        return boundaries


    def _split_and_reduce(self, acc, local_th, boundaries):
        """
        Split the gradients into partitions and reduce them by selecting top-k values.
        Each worker receives sparse regions from the other workers and and then conducts
        the reduction locally. 

        TODO: To simplify the implementation, boundaries are not used, instead: allreduce
        TODO: To simplify the implementation, destination rotation and bucketing,
        is not yet implemented.

        Parameters:

            - acc: Gradient matrix accumulation values.
            - local_th: Local threshold for selecting top-k values.
            - boundaries: Boundaries for partitioning the gradient space: { proc_id : (row_start, row_end) }.

         Returns:
            - reduced_topk: The reduced top-k gradient values.
            - local_topk_indexes: The indices of the top-k gradient values selected locally.
        """
        # 1. Local topk values selection: 
        # All procs perform topk selection in all acc
        local_topk, local_topk_indexes = self._topk_selection(acc, local_th)

        # 2. Balance split and reduce
        # TODO: Para simplificar de momento: Allreduce
        # TODO: Convert to COO for allreduce
        #  local_topk_coo = self._convert_to_coo_format(local_topk, "offset")
        reduced_topk = self._allreduce(local_topk)
        return reduced_topk, local_topk_indexes


    def _balance_and_allgather(self, reduced_topk, global_th):
        # 1. Global topk selection
        global_topk, global_topk_indexes = self._topk_selection(reduced_topk, global_th)

        # 2. Data packaging
        # TODO

        # 3. Data balancing
        # TODO

        # 4. Allgatherv using recursive doubling
        # Como antes hemos hecho un allreduce, creo que no hace falta allgather
        # TODO 

        return global_topk, global_topk_indexes

    def _topk(self, tensor, k):
        """
            Implementation of Li and Hoefler
        """
        indexes = np.abs(tensor).argsort()[-k:]
        return indexes, tensor[indexes]


    def _topk_selection(self, data, threshold):
        """
        Selects top-k elements from the data array that are greater than or equal to the threshold.
        
        Parameters:
            - data: The input array from which to select elements.
            - threshold (float): The threshold value to compare against the absolute values of the data elements.
        
        Returns:
            - topk: An array of the same shape as `data`, where elements that meet the threshold condition
                        are retained, and all other elements are set to zero.
            - topk_indexes: 
        """
        topk = np.zeros_like(data, dtype=data.dtype)
        topk_indexes = np.where(np.abs(data) >= threshold)[0]
        topk[topk_indexes] = data[topk_indexes]
        return topk, topk_indexes
        

    def _convert_to_coo_format(self, data, storage_format="offset"):
        """
        Returns a sparse array in coo format 

        Parameters: 
            - data: dense matrix 
            - storage_format: "offset" (default) or "coordinate" 
        
        Returns:
            - if "offset" storage format: { offset: non-zero value }
            - if "coordinate" storage format: { (row, col): non-zero value}
        """

        coo_storage_formats = {"offset", "coordinate"} 

        if storage_format not in coo_storage_formats:
            print(f"Storage format '{storage_format}' not known. Try with: {coo_storage_formats}")
            return None

        if storage_format == "coordinate":
            rows, cols = np.nonzero(data)
            values = data[rows, cols]
            coo_data = {(row, col): value for row, col, value in zip(rows, cols, values)}

        elif storage_format == "offset":
            flattened_data = data.flatten()
            indexes = np.nonzero(flattened_data)[0]
            values = flattened_data[indexes]
            coo_data = {index: value for index, value in zip(indexes, values)}

        return coo_data


    def _convert_from_coo_format(self, coo_data, shape, storage_format="offset"):
        """
        Returns a dense array from sparse data 

        Parameters: 
            - coo_data: depending on the storage_format, it is either:
                - if "offset" storage format, coo_data: { offset: non-zero value }
                - if "coordinate" storage format, coo_data: { (row, col): non-zero value}
            - coo_data: sparse matrix [[values], [coordinates/offset]]
            - shape: dense matrix expected shape 
            - storage_format: "offset" (default) or "coordinate" 
        
        Returns:
            - dense_matrix: dense data with the provided shape
        """

        coo_storage_formats = {"offset", "coordinate"} 

        if storage_format not in coo_storage_formats:
            print(f"Storage format '{storage_format}' not known. Try with: {coo_storage_formats}")
            return None

        dense_matrix = np.zeros(shape, dtype=np.float64)

        if storage_format == "coordinate":
            for (row, col), value in coo_data.items():
                dense_matrix[row, col] = value

        elif storage_format == "offset":
            rows, cols = shape
            for offset, value in coo_data.items():
                row = offset // cols
                col = offset % cols
                dense_matrix[row, col] = value

        return dense_matrix
        

    def _allgather(self, data):
        if self.nprocs == 1:
            return data
        self.comm.Allgather(MPI.IN_PLACE, data)
        return data


    def _allreduce(self, data):
        if self.nprocs == 1:
            return data
        self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        return data
