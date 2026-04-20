#define TYPE "TYPE"
 
__global__ void kl_divergence_metric(TYPE *y_targ, TYPE *y_pred, TYPE *res, int b, int n, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b) {
        int i = 0;
        res[idx * n] = y_targ[idx * n];
        for ( i = 1; i < n; i++ ) {
            res[idx * n + i] = fabs(y_pred[idx * n + i] * logf(fabs(y_pred[idx * n + i] / (y_targ[idx * n + i] + eps)) + eps));
        }
    }
    return;
}