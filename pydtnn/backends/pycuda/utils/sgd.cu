#define TYPE "TYPE"
#define NESTEROV_OPS "Remove this line to unset nesterov" /*True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])", False: "w[i] -= lr * (decay * w[i] + v[i])"*/

__global__ void sgd_gpudirect(TYPE *w, TYPE *dw, TYPE *v,
                              float lr, float decay, float momentum, int N)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;
    int i; 

    for(i = base_idx; i < N; i+= workers)
    {
        dw[i] = (TYPE) dw[i] + decay * w[i];

        v[i] = momentum * v[i] + dw[i];
        #ifdef NESTEROV_OPS
            w[i] -= lr * (dw[i] + momentum * v[i]);
        #else
            w[i] -= lr * v[i];
        #endif
    }
}