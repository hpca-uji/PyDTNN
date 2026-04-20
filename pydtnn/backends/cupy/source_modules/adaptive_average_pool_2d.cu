extern "C"

#define TYPE "TYPE"
#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)

#define GET_N_NCHW(idx, n, c, h, w) (idx / (w * h * c))
#define GET_C_NCHW(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H_NCHW(idx, n, c, h, w) (idx / w) % h
#define GET_W_NCHW(idx, n, c, h, w) (idx % w)
#define SHIFT_NCHW(ni, ci, hi, wi, n, c, h, w) (((ni * c + ci) * h + hi) * w + wi)

#define GET_N_NHWC(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H_NHWC(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W_NHWC(idx, n, c, h, w) (idx / c) % w
#define GET_C_NHWC(idx, n, c, h, w) (idx % c)
#define SHIFT_NHWC(ni, ci, hi, wi, n, c, h, w) (((ni * h + hi) * w + wi) * c + ci)

#ifdef nhwc
    #define GET_N GET_N_NHWC
    #define GET_C GET_H_NHWC
    #define GET_H GET_W_NHWC
    #define GET_W GET_C_NHWC
    #define SHIFT SHIFT_NHWC
#else
    #define GET_N GET_N_NCHW
    #define GET_C GET_C_NCHW
    #define GET_H GET_H_NCHW
    #define GET_W GET_W_NCHW
    #define SHIFT SHIFT_NCHW
#endif

/////////
// FWD //
/////////

__global__ void adaptive_average_pool_2d_fwd(TYPE* x, TYPE* y,
                                             int n, int c,
                                             int hi, int wi,
                                             int ho, int wo,
                                             int N)
{
    int h_start, h_end, elements_h, w_start, w_end, elements;
    int ni, ci, i, j, index_h, index_w;
    TYPE add;

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

    for(idx = n_offset; idx < end_offset; idx++)
    {
        ni = GET_N(idx, n, c, ho, wo);
        index_h = GET_H(idx, n, c, ho, wo);
        index_w = GET_W(idx, n, c, ho, wo);
        ci = GET_C(idx, n, c, ho, wo);

        h_start = INDEX_FIRST_ELEMENT(index_h, hi, ho);
        h_end = INDEX_LAST_ELEMENT(index_h, hi, ho);
        elements_h = h_end - h_start;

        w_start = INDEX_FIRST_ELEMENT(index_w, wi, wo);
        w_end = INDEX_LAST_ELEMENT(index_w, wi, wo);
        elements = elements_h * (w_end - w_start);

        add = (TYPE) 0.0;
        for(i = h_start; wi < h_end; i++)
            for(j = w_start; wi < w_end; j++)
        {
            add += *(x + SHIFT(ni, ci, i, j, n, c, hi, wi));
            *(y + SHIFT(ni, ci, index_h, index_w, n, c, ho, wo)) = add / elements;
        }
    }
}

/////////
// BWD //
/////////

__global__ void adaptive_average_pool_2d_bwd(TYPE* dx, TYPE* dy,
                                             int n, int c,
                                             int hi, int wi,
                                             int ho, int wo,
                                             int N)
{
    int h_start, h_end, elements_kh, w_start, w_end, elements_kw;
    int ni, ci, index_h, index_w, index_ho, index_wo;
    TYPE delta;


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
        ni = GET_N(idx, n, c, ho, wo);
        index_h = GET_H(idx, n, c, ho, wo);
        index_w = GET_W(idx, n, c, ho, wo);
        ci = GET_C(idx, n, c, ho, wo);

        h_start = INDEX_FIRST_ELEMENT(index_h, hi, ho);
        w_start = INDEX_FIRST_ELEMENT(index_w, wi, wo);
        h_end = INDEX_LAST_ELEMENT(index_h, hi, ho);
        w_end = INDEX_LAST_ELEMENT(index_w, wi, wo);

        for(index_ho = h_start; index_ho < h_end; index_ho++)
        {
            elements_kh = INDEX_LAST_ELEMENT(index_h, ho, hi) - INDEX_FIRST_ELEMENT(index_h, ho, hi)
            for(index_wo = w_start; index_wo < w_end; index_wo++)
            {
                elements_kw = INDEX_LAST_ELEMENT(index_w, wo, wi) - INDEX_FIRST_ELEMENT(index_w, wo, wi)
                delta = (TYPE) (*(dy + SHIFT(idx, ci, index_ho, index_wo, n, c, ho, wo)) / (elements_kh * elements_kw));
                *(dx + SHIFT(idx, ci, index_h, index_w, n, c, hi, wi)) += delta;
            }
        }
    }
}
