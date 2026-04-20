#define TYPE "TYPE"

// FWD

__global__ void leaky_relu_fwd(TYPE* x, TYPE* max, TYPE* mask,
                               float negative_slope, int num_workers, int N)
{
    int idx, i;
    TYPE elem;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {
        elem = x[i];

        if (elem > 0)
        {
            max[i] = elem;
            mask[i] = 1;
        }
        else if(elem < 0)
        {
            max[i] = (TYPE) (elem * negative_slope);
            mask[i] = negative_slope;
        }
        else
        {
            max[i] = 0;
            mask[i] = 0;
        }
    }
}

// BWD

__global__ void leaky_relu_bwd(TYPE* dx, TYPE* dy, TYPE* mask,
                              int num_workers, int N)
{
    int i;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
        dx[i] = dy[i] * mask[i];
}

