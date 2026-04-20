extern "C"

#define TYPE "TYPE"

#define GET_J(idx, dim_j) (idx % dim_j)
#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void batch_normalization_bwd(TYPE* dx, TYPE* dy, TYPE* xn,
                                        TYPE* std, TYPE* gamma,
                                        TYPE* dgamma, TYPE* dbeta,
                                        int dim_i, int dim_j, int N)
{
    int j;
    const int n = (const int) dim_i;
    TYPE _gamma, _std, _dy, _xn, _dgamma, _dbeta;

    // BLOCK DISTRIBUTION
    int idx;
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
    // BLOCK DISTRIBUTION

    for(idx = n_offset; idx < end_offset; idx++)
    {
        j = GET_J(idx, dim_j);

        _gamma = *(gamma + j);      // gamma[j]
        _std = *(std + j);          // std[j]
        _dy = *(dy + idx);          // dy[idx] = dy[i][j]
        _xn = *(xn + idx);          // xn[idx] = xn[i][j]
        _dgamma = *(dgamma + j);    // dgamma[j]
        _dbeta = *(dbeta + j);      // dbeta[j]

        *(dx + idx) = (TYPE) ( _gamma / ( _std * n) ) * (n * _dy - _xn * _dgamma - _dbeta);
    }
}