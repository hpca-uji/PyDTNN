#define TYPE "TYPE"

#define SHIFT_2D_AR(p, i, j, dim_j) (p + ((i * dim_j) + j))

__global__ void regression_mse(TYPE *y_targ, TYPE *y_pred, TYPE *res, TYPE *local_res, int n, int labels)
{
    int i, idx;
    TYPE diff, val_targ, val_pred;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < n; idx += workers)
    {
        *(local_res + idx) = (TYPE) 0.0;

        for(i = 0; i < labels; i++)
        {
            // val_targ = y_targ[idx][i];
            val_targ = (*SHIFT_2D_AR(y_targ, idx, i, labels));

            // val_pred = y_pred[idx][i];
            val_pred = (*SHIFT_2D_AR(y_pred, idx, i, labels));

            diff = val_targ - val_pred;
            *(local_res + idx) += (diff * diff);
        }

    }

    if(base_idx == 0)
    {
        (*res) = (*local_res);
        for(idx = 1; (idx < n); idx++)
            (*res) += (*(local_res + idx));

        (*res) /= (n * labels);
    }
}