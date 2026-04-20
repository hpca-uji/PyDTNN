#define TYPE "TYPE"
#define NESTEROV_OPS "Remove this line to unset nesterov" /*True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])", False: "w[i] -= lr * (decay * w[i] + v[i])"*/

__global__ void sgd_gpudirect(TYPE *w, TYPE *dw, TYPE *v,
                              float lr, float decay, float momentum, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        v[i] = momentum * v[i] + dw[i];
    #ifdef NESTEROV_OPS
        w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i]);
    #else
        w[i] -= lr * (decay * w[i] + v[i]);
    #endif
    }
}