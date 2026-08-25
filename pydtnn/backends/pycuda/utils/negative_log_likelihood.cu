#define TYPE "TYPE"

__global__ void negative_log_likelihood(TYPE *y_targ, TYPE *y_pred,
                                        TYPE *loss, TYPE *weights,
                                        TYPE *dx, TYPE *argmax,
                                        int b, int n)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;
    int idx, i, max;
    TYPE max_value, pred;

    // Getting target's class
    for(idx = base_idx; idx < b; idx += workers)
    {
        max = 0;
        max_value = y_targ[idx * n];

        for (i = 1; i < n; i++)
        {
            if ( y_targ[idx * n + i] > max_value )
            {
                max = i;
                max_value = y_targ[idx * n + i];
            }
        }
        argmax[idx] = max;

        // Calculating the Loss and "DX"

        // Common
        pred = (TYPE) (y_pred[idx * n + max] / sum_y_targ);
        if ( pred < eps )          pred = eps;
        else if ( pred > (1-eps) ) pred = (1-eps);

        // Loss
        loss[idx] = (TYPE) logf(pred);
        loss[idx] = (TYPE) (weights[max] * loss[idx]);
        // The rest of the loss's operations will be done in the python's code.
        
        // DX
        dx[idx * n + max] /= -(pred * weights[max]);
    }
    return;
}