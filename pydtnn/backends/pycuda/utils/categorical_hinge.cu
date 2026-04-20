#define TYPE "TYPE"
#define SHIFT_2D_AR(p, i, j, dim_j) (p + ((i * dim_j) + j))

__global__ void categorical_hinge(TYPE *y_targ, TYPE *y_pred, TYPE *res,
                                  TYPE *local_res, int n, int labels)
{
    int i, idx;
    TYPE pos, neg, max_v, val_targ, val_pred;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < n; idx += workers)
    {
        for(i = 0, max_v = (TYPE) 0.0; i < labels; i++)
        {
            // val_targ = y_targ[idx][i];
            val_targ = (*SHIFT_2D_AR(y_targ, idx, i, labels));

            // val_pred = y_pred[idx][i];
            val_pred = (*SHIFT_2D_AR(y_pred, idx, i, labels));

            pos += (TYPE) (val_targ * val_pred);
            neg = (TYPE) (-1 * val_targ) + 1;
            if ( (i == 0) || (max_v < neg))
                max_v = neg;
        }
        max_v = (TYPE) ((max_v - pos) + 1);
        *(local_res + idx) = (TYPE) (max_v > 0 ? max_v : 0);
    }

    if(base_idx == 0)
    {
        for(idx = 1; idx < n; idx++)
            *(local_res) += *(local_res + idx);

        *(res) = (TYPE) (*(local_res) / (n * labels));
    }
}