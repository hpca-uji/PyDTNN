#define TYPE "TYPE"

#define TRUE_POSITIVE  {0,0}
#define TRUE_NEGATIVE  {1,1}
#define FALSE_NEGATIVE {0,1}
#define FALSE_POSITIVE {1,0}

#define SHIFT_POINTER_CM(label, i, j, num_rows, num_columns) (((label * num_rows + i) * num_columns) + j)
#define SHIFT_POINTER_LOCAL_CM(idx, label, i, j, num_labels, num_rows, num_columns) ((((idx * num_labels + label) * num_rows + i) * num_columns) + j)
#define SHIFT_POINTER_Y(i, j, dim_j) (i * dim_j + j)

__constant__ const short indexes[2][2][2] = {
    {TRUE_POSITIVE, TRUE_NEGATIVE},
    {FALSE_NEGATIVE, FALSE_POSITIVE}
};


__global__ void binary_confusion_matrix(TYPE *y_targ, TYPE *y_pred, int *cm, int *local_cm, const int num_classes, const int n)
{
    int label, i, j, is_pred_correct, idx;
    short index_0, index_1;
    TYPE value_targ, value_pred;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < n; idx += workers)
    {
        for(label = 0; label < num_classes; label++)
        {
            value_targ = *(y_targ + SHIFT_POINTER_Y(idx, label, num_classes));
            value_pred = *(y_pred + SHIFT_POINTER_Y(idx, label, num_classes));

            // NOTE: y_pred[idx][label]' only possible values are 0 or 1.
            is_pred_correct = (value_targ == value_pred);
            index_0 = indexes[is_pred_correct][((int) value_pred)][0];
            index_1 = indexes[is_pred_correct][((int) value_pred)][1];

            *(local_cm + SHIFT_POINTER_LOCAL_CM(idx, label, index_0, index_1, num_classes, 2, 2)) += 1;
        }
    }
    // Accumulating the local values
    if (base_idx == 0)
    {
        for(idx = blockDim.x/2; idx > 0; idx >>= 1)
        {
            if(base_idx < idx)
                for(label = 0; label < num_classes; label++)
                    for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                        *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx, label, i, j, num_classes, 2, 2)) += *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx + idx, label, i, j, num_classes, 2, 2));
        }
    }
    __syncthreads();

    // Accumulating the local values into the output's tensor.
    if (base_idx == 0)
    {
        for(label = 0; label < num_classes; label++)
            for(i = 0; i < 2; i++) for(j = 0; j < 2; j++)
                *(cm + SHIFT_POINTER_CM(label, i, j, 2, 2)) = *(local_cm + SHIFT_POINTER_LOCAL_CM(base_idx, label, i, j, num_classes, 2, 2));
    }
}