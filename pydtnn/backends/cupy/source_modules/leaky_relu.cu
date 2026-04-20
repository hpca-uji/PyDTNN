extern "C"

#define TYPE "TYPE"

/////////
// FWD //
/////////

__global__ void leaky_relu_fwd(TYPE* x, TYPE* max, TYPE* mask,
                               float negative_slope, int N)
{
    int i;
    TYPE elem;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }
    else
    {
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }
    end_offset = n_offset + n_samples;

    for(i = n_offset; i < end_offset; i++)
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

/////////
// BWD //
/////////

__global__ void leaky_relu_bwd(TYPE* dx, TYPE* dy, TYPE* mask, int N)
{
    int i;
    TYPE elem;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }
    else
    {
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }
    end_offset = n_offset + n_samples;

    for(i = n_offset; i < end_offset; i++)
        dx[i] = dy[i] * mask[i];
}
