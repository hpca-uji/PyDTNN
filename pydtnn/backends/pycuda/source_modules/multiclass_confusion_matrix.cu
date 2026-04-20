#define TYPE "TYPE"

#define SHIFT_Y(p, i, dim_j) p + (i * dim_j)
#define INDEX_FIRST_ONE_ON(y, var_class) for(i = 0; (i < num_classes) && ((*(y + i)) != 0); i++); var_class = i;
#define SHIFT_POINTER_CM(p, i, j, num_classes) p + (i * num_classes + j)
#define SHIFT_POINTER_LOCAL_CM(p, idx, i, j, num_i, num_j) p + ((idx * num_i + i) * num_j + j)

__global__ void multiclass_confusion_matrix(TYPE *y_targ, TYPE *y_pred, int *cm, int *local_cm, const int num_classes, const int n)
{
    int idx, idx_i, i, j, target_class, predicted_class;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < n; idx += workers)
    {
        INDEX_FIRST_ONE_ON(SHIFT_Y(y_targ, idx, num_classes), target_class)
        INDEX_FIRST_ONE_ON(SHIFT_Y(y_pred, idx, num_classes), predicted_class)

        (*(SHIFT_POINTER_LOCAL_CM(local_cm, idx, target_class, predicted_class, num_classes, num_classes))) += 1;
    }

    // Accumulating the local values
    //for(idx_i = blockDim.x/2; idx_i > 0; idx_i >>= 1)
    //{
    //    if(threadIdx.x < idx_i)
    //    {
    //        for(i = 0; i < num_classes; i++) for(j = 0; j < num_classes; j++)
    //        {
    //            (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx, i, j, num_classes, num_classes))) += (*(SHIFT_POINTER_LOCAL_CM(local_cm, base_idx + idx_i, i, j, num_classes, num_classes)));
    //        }
    //    }
    //    __syncthreads();
    //}

    __syncthreads();

    // Accumulating the local values into the output's tensor.
    if (base_idx == 0)
    {
        for(idx_i = 1; idx < n; idx ++)
            for(i = 0; i < num_classes; i++)
                for(j = 0; j < num_classes; j++)
        {
            (*(SHIFT_POINTER_CM(cm, i, j, num_classes))) += (*(SHIFT_POINTER_LOCAL_CM(local_cm, idx_i, i, j, num_classes, num_classes)));
        }
    }
}