#define TYPE "TYPE"

__global__ void categorical_cross_entropy(TYPE *y_targ, TYPE *y_pred, TYPE *res,
                                          TYPE *dx, int b, int n, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b)
    {
        int i = 0, max = 0;
        TYPE max_value = y_targ[idx * n];
        dx[idx * n] = y_targ[idx * n];
        for ( i = 1; i < n; i++ )
        {
            dx[idx * n + i] = y_targ[idx * n + i];
            if ( y_targ[idx * n + i] > max_value )
            {
                max = i;
                max_value = y_targ[idx * n + i];
            }
        }

        TYPE pred = y_pred[idx * n + max];
        if ( pred < eps )          pred = eps;
        else if ( pred > (1-eps) ) pred = (1-eps);

        res[idx] = logf(pred);
        dx[idx * n + max] /= -(pred * b);
    }
    return;
}