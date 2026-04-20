#define TYPE "TYPE"
#define SHIFT_2D_AR(p, i, j, dim_j) (p + ((i * dim_j) + j))

__global__ void categorical_mae(TYPE *y_targ, TYPE *y_pred, TYPE *res, TYPE *local_res, int n, int labels)
{
    int i, idx;
    TYPE val_targ, val_pred, error;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < n; idx += workers)
    {
        for(i = 0; i < labels; i++)
        {
            // val_targ = y_targ[idx][i];
            val_targ = (*SHIFT_2D_AR(y_targ, idx, i, labels));

            // val_pred = y_pred[idx][i];
            val_pred = (*SHIFT_2D_AR(y_pred, idx, i, labels));

            error = (TYPE) (val_targ - val_pred);
            error = error > 0 ? error : (-1) * error; // absolute error
            *(local_res + idx) += error;
        }
    }

    // Getting the mean and accumulating it on the output's buffer.
    if(base_idx == 0)
    {
        for(idx = 1; idx < n; idx++)
            *(local_res) += *(local_res + idx);

        *(res) = (TYPE) (*(local_res) / (n * labels));
    }
}