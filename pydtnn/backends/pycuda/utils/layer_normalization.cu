#define TYPE "TYPE"

__global__ void layer_normalization_fwd(TYPE *x, TYPE *y, TYPE *xn, TYPE *std, TYPE *gamma,
                                            TYPE *beta, float epsilon, int batch, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        int i = 0;
        TYPE mu = 0;
        TYPE var = 0;
        TYPE xc = 0;
        // Mean
        for ( i = 0; i < n; i++ ) {
            mu += x[idx * n + i] / n;
        }

        // Var
        for ( i = 0; i < n; i++ ){
            xc = x[idx * n + i] - mu;
            var += (xc * xc) / n;
            xn[idx * n + i] = xc;
        }
        var = sqrtf(var + epsilon);
        std[idx] = var;
        // Normalization and Scaling
        for ( i = 0; i < n; i++ ){
            xn[idx * n + i] /= (var + epsilon);
            y[idx * n + i] = gamma[i] * xn[idx * n +i] + beta[i];
        }
    }
    return;
}

///////

__global__ void layer_normalization_bwd(TYPE *dy, TYPE *dx, TYPE *xn, TYPE *std, TYPE *gamma, float epsilon, int batch, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        int i = 0;
        TYPE mean1 = 0;
        TYPE mean2 = 0;

        // Means
        for ( i = 0; i < n; i++ ) {
            mean1 += gamma[i] * xn[idx * n + i] * (dy[idx * n + i] / n);
            mean2 += gamma[i] * (dy[idx * n + i] / n);
        }

        // dx
        for ( i = 0; i < n; i++ ) {
            dx[idx * n + i] = (dy[idx * n + i] - xn[idx * n + i] * mean1 - mean2) / (std[idx] + epsilon);
        }
    }
    return;
}

//////////////////


__global__ void layer_normalization_backward_weights(TYPE *dy, TYPE *xn, TYPE *dgamma, TYPE *dbeta, float epsilon, int batch, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int i = 0;
        TYPE mean1 = 0;
        TYPE mean2 = 0;

        // Means
        for ( i = 0; i < batch; i++ ) {
            mean1 += xn[i * n + idx] * (dy[i * n + idx] / batch);
            mean2 += dy[i * n + idx] / batch;
        }
        dgamma[idx] = (fabs(mean1) < epsilon) ? 0.0 : mean1;
        dbeta[idx] = (fabs(mean2) < epsilon) ? 0.0 : mean2;
    }
    return;
}