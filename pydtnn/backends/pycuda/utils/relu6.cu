#define TYPE "TYPE"

// FWD

__global__ void relu6_fwd(TYPE* x, TYPE* max, TYPE* mask,
                          float cap, int num_workers, int N)
{
    int i;
    TYPE elem;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
    {
        elem = x[i];

        if(elem >= cap)
        {
            max[i] = (TYPE) cap;
            mask[i] = 1;
        }
        else if (elem > 0)
        {
            max[i] = elem;
            mask[i] = 1;
        }
        else
        {
            max[i] = 0;
            mask[i] = 0;
        }
    }
}

// BWD

__global__ void relu6_bwd(TYPE* dx, TYPE* dy, TYPE* mask,
                          int num_workers, int N)
{
    int i;

    for(i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += num_workers)
        dx[i] = dy[i] * mask[i];
}
