#define TYPE "TYPE"

__global__ void binary_cross_entropy(TYPE *y_targ, TYPE *y_pred, TYPE *loss,
                                     TYPE *dx, TYPE *weights, TYPE *argmax,
                                     int b, int n, TYPE eps)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;
    int idx, i, max;
    TYPE pred, max_value;

    // Getting the weight's max values
    for(idx = base_idx; idx < b; idx += workers)
    {
        i = 0, max = 0;
        max_value = y_targ[idx * n];
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
        argmax[idx] = weights[max];
    }


    // Getting the loss and the gradient.
    for (idx = base_idx; idx < b; idx += workers){
        i = 0, max = 0;
        loss[idx] = 0;

        for ( i = 0; i < n; i++ )
        {
            loss[idx] += (1 - y_targ[idx * n + i] ) * logf(fmaxf(1 - y_pred[idx * n + i], eps));
            pred = y_pred[idx * n + max];

            if ( pred < eps )          pred = eps;
            else if ( pred > (1-eps) ) pred = (1-eps);

            dx[idx * n + i] = (-(y_targ[idx * n + i]  / pred) + ((1 - y_targ[idx * n + i]) / pred) );

            // The remaining operation will be done in the python's code
        }
    }
    return;
}