import warnings
import numpy as np


class SparseMatrixCOO:
    """
    Represents a sparse matrix in COO format.
    This format stores the matrix using three arrays:
        - data: the nonzero values.
        - row: the row indices corresponding to each value.
        - col: the column indices corresponding to each value.
    The matrix is assumed to be in canonical format: indices sorted by row and then by column, and no duplicate entries are present.
    This class is not designed to store explict zeros so, len(self.data) should always be equal to nnz. 
    """

    def __init__(self, data: np.ndarray, row:np.ndarray, col:np.ndarray, shape:tuple, has_canonical_format:bool):
        """
        Primary initializer for SparseMatrixCOO.
        
        Parameters:
            data (np.ndarray): Array with the nonzero values.
            row (np.ndarray): Array with the row indices.
            col (np.ndarray): Array with the column indices.
            shape (tuple): Shape of the original matrix.
        """

        if len(data) != len(row) or len(data) != len(col):
            raise AssertionError("Data, row, and col arrays must have the same shape")
        
        if has_canonical_format:
            assert(self._has_canonical_format())
            self.data = data
            self.row = row
            self.col = col
            self.shape = shape
            self.nnz = len(self.data)
            self.has_canonical_format = True

        else:
            # TODO: order arrays in canonical format
            raise NotImplementedError("Not yet implemented constructor with unordered rows and cols")


    @classmethod
    def from_dense(cls, dense_array: np.ndarray):
        """
        Alternative constructor to create a SparseMatrixCOO from a dense 2D array.
        
        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.
        
        Returns:
            SparseMatrixCOO: The sparse matrix in COO format.
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")
        
        # TODO: Implement conversion from dense to COO
        raise NotImplementedError("Conversion from dense array is not yet implemented.")


    @classmethod
    def top_threshold_selection(cls, dense_array: np.ndarray, threshold: np.float32):
        """
        Alternative constructor to create a SparseMatrixCOO from a dense array,
        considering only elements with absolute value greater than or equal to the threshold.
        
        Parameters:
            dense_array (np.ndarray): A 2D dense matrix.
            threshold (np.float32): Threshold for including an element.
        
        Returns:
            SparseMatrixCOO: The sparse matrix in COO format, containing only significant elements.
        """

        if len(dense_array.shape) != 2:
            raise AssertionError("Dense array must be 2D.")
        
        topk_row, topk_col = np.where(np.abs(dense_array) >= threshold)
        topk = dense_array[topk_row, topk_col]
        return cls(topk, topk_row, topk_col, dense_array.shape, has_canonical_format=True)
    

    def __add__(self, other):
        """
        Adds two SparseMatrixCOO matrices that are in canonical format.
        
        Parameters:
            other (SparseMatrixCOO): Another SparseMatrixCOO instance.
        
        Returns:
            SparseMatrixCOO: A new instance representing the sum of both matrices.
        """

        if type(other) != SparseMatrixCOO:
            raise AssertionError("Operand must be a SparseMatrixCOO instance.")
        if self.shape != other.shape:
            raise AssertionError("Matrices must have the same shape.")
        if not self.has_canonical_format or not other.has_canonical_format:
            raise AssertionError("Both matrices must be in canonical format.")

        count = 0
        i_self, i_other = 0, 0

        max_size = self.nnz + other.nnz
        summ_val = np.empty(max_size, dtype=np.float32)
        summ_row = np.empty(max_size, dtype=np.int32)
        summ_col = np.empty(max_size, dtype=np.int32)
        while i_self < self.nnz and i_other < other.nnz:
            row_self = self.row[i_self]
            row_other = other.row[i_other]
            if row_self < row_other:
                # Set self data, row, col
                summ_val[count] = self.data[i_self]
                summ_row[count] = self.row[i_self]
                summ_col[count] = self.col[i_self]
                i_self += 1
            elif row_self > row_other:
                # Set other data, row, col
                summ_val[count] = other.data[i_other]
                summ_row[count] = other.row[i_other]
                summ_col[count] = other.col[i_other]
                i_other += 1
            else:
                # Same row, let's see the column
                col_self = self.col[i_self]
                col_other = other.col[i_other]
                if col_self < col_other:
                    # Set self data, row, col
                    summ_val[count] = self.data[i_self]
                    summ_row[count] = self.row[i_self]
                    summ_col[count] = self.col[i_self]
                    i_self += 1
                elif col_self > col_other:
                    # Set other data, row, col
                    summ_val[count] = other.data[i_other]
                    summ_row[count] = other.row[i_other]
                    summ_col[count] = other.col[i_other]
                    i_other += 1
                else:
                    # Set self + other data, row, col
                    summ_val[count] = self.data[i_self] + other.data[i_other]
                    summ_row[count] = self.row[i_self]
                    summ_col[count] = self.col[i_self]
                    i_other += 1                    
                    i_self += 1
            count += 1

        return SparseMatrixCOO(summ_val[:count], summ_row[:count], summ_col[:count], self.shape, has_canonical_format=True)


    def intersect_indexes(self, other):
        count = 0
        i_self, i_other = 0, 0
        max_size = min(self.nnz, other.nnz)
        intersected_rows = np.empty(max_size, dtype=np.int32)
        intersected_cols = np.empty(max_size, dtype=np.int32)
        while i_self < self.nnz and i_other < other.nnz:
            row_self = self.row[i_self]
            row_other = other.row[i_other]
            if row_self < row_other:
                i_self += 1
            elif row_self > row_other:
                i_other += 1
            else:
                col_self = self.col[i_self]
                col_other = other.col[i_other]
                if col_self < col_other:
                    i_self += 1
                elif col_self > col_other:
                    i_other += 1
                else:
                    intersected_rows[count] = row_self
                    intersected_cols[count] = col_self
                    i_other += 1                    
                    i_self += 1
                    count += 1
        return intersected_rows[:count], intersected_cols[:count]


    def top_threshold_selection(self, threshold, inplace=True):
        mask = np.abs(self.data) >= threshold
        topk = self.data[mask]
        topk_row, topk_col = (self.row[mask], self.col[mask])

        if inplace:
            self.data = topk
            self.row = topk_row
            self.col = topk_col
            self.nnz = len(self.data)
            # self.shape remains equal
            # self.has_canonical_format remains equal
        else:
            return SparseMatrixCOO(topk, topk_row, topk_col, self.shape, self.has_canonical_format)
        

    def get_triplet(self):
        return self.data, self.row, self.col
    

    def _has_canonical_format(self):
        """
        Check if SparseMatrixCOO follows canonical format: 
            - Indexes are sorted by row and then by column 
            - There are no duplicate entries
            - There may have explicit zero elements
        This function is computationally expensive and therefore should only be used for developing/debugging purposes.
        This function should only be used in developement to assert that sparse matrices have canonical format. 

        Returns:
            has_canonical_format (boolean): True if indexes are in canonical format, False if not. 
        """
        
        warnings.warn("This function ('has_canonical_format') should be used only in case of debugging for performance reasons.")
        
        if self.nnz == 0:
            return True

        if not np.all(self.row[:-1] <= self.row[1:]):
            return False

        for i in range(self.nnz - 1):
            if self.row[i] == self.row[i + 1] and self.col[i] >= self.col[i + 1]:
                return False
        return True
    
