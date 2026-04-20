#define TYPE "TYPE"

__global__ void categorical_accuracy(TYPE *y_targ, TYPE *y_pred, TYPE *res, int b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    for(; idx < b; idx += workers)
    {
        int i = 0, max = 0;
        TYPE max_value = y_pred[idx * n];
        for ( i = 1; i < n; i++ )
        {
            if ( y_pred[idx * n + i] > max_value )
            {
                max = i;
                max_value = y_pred[idx * n + i];
            }
        }
        res[idx] = y_targ[idx * n + max];
    }
    return;
}
