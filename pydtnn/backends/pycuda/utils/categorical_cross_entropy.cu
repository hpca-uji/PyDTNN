#define TYPE "TYPE"

__global__ void categorical_cross_entropy(TYPE *y_targ, TYPE *y_pred, TYPE *loss,
                                          TYPE *weights, TYPE *dx, TYPE *argmax,
                                          int b, int n, float eps, TYPE sum_y_targ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;
    TYPE max_value, pred;
    int i, max;

    for(/*idx=idx*/; idx < b; idx += workers)
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
        argmax[idx] = max;

        pred = (TYPE) (y_pred[idx * n + max] / sum_y_targ);
        if ( pred < eps )          pred = eps;
        else if ( pred > (1-eps) ) pred = (1-eps);
        
        loss[idx] = logf(pred);
        loss[idx] = (TYPE) (weights[max] * loss[idx]);

        dx[idx * n + max] /= -pred;
        // The rest of the operations will be done in the python's code
    }
    return;
}