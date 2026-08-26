#define TYPE "TYPE"

// FWD

__global__ void log_fwd(TYPE* x, TYPE* y,
                        int num_workers, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = i; i < N; i += num_workers)
    {
        y[i] = log(x[i]);
    }
}


// BWD

__global__ void log_bwd(TYPE* dx, TYPE* dy, TYPE* y,
                        int num_workers, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = i; i < N; i += num_workers)
    {
        dx[i] = dy[i] * exp(-y[i]);
    }
}