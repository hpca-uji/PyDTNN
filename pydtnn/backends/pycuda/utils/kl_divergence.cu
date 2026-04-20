#define TYPE "TYPE"


__global__ void kl_divergence(TYPE *y_targ, TYPE *y_pred, TYPE *res,
                              TYPE *dx, int b, int bs, int n, TYPE eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b) {
        int i = 0;
        double partial = 0;
        double loss = 0;
        for ( i = 0; i < n; i++ ) {
            partial = logf(fabs(y_targ[idx * n + i] / (y_pred[idx * n + i] + eps)) + 1.0) / bs;
            loss += fabs(y_targ[idx * n + i] * partial);
            dx[idx * n + i] = (T) partial;
        }
        res[idx] = (T) loss;
    }
    return;
}