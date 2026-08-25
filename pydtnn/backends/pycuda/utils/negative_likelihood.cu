#define TYPE "TYPE"

__global__ void negative_likelihood(TYPE *y_targ, TYPE *y_pred,
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

        // Calculating the Loss and "dx"

        // Common
        max = argmax[idx];

        // Loss
        pred = y_pred[idx * n + max];
        loss[idx] = (TYPE) (weights[max] * pred);
        // NOTE: The rest of the loss' calculation will be done outside.

        // DX
        dx[idx * n + max] = (TYPE) (-1 * weights[max]);
    }
    return;
}