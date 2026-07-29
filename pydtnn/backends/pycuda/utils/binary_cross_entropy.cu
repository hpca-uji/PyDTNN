#define TYPE "TYPE"

__global__ void binary_cross_entropy(TYPE *y_targ, TYPE *y_pred, TYPE *res,
                                     TYPE *dx, int b, int n, TYPE eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b) {
        int i = 0, max = 0;
        TYPE pred;
        res[idx] = 0;
        for ( i = 0; i < n; i++ )
        {
            res[idx] += (1 - y_targ[idx * n + i] ) * logf(fmaxf(1 - y_pred[idx * n + i], eps));
            pred = y_pred[idx * n + max];
            if ( pred < eps )          pred = eps;
            else if ( pred > (1-eps) ) pred = (1-eps);
            dx[idx * n + i] = (-(y_targ[idx * n + i]  / pred) + ((1 - y_targ[idx * n + i]) / pred) ) / b;
        }
    }
    return;
}