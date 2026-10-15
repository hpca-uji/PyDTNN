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

#define GET_N_BACKWARDS_NCHW(idx, n, c) idx / c
#define GET_C_BACKWARDS_NCHW(idx, n, c) idx % c

#define GET_N_NHWC(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H_NHWC(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W_NHWC(idx, n, c, h, w) (idx / c) % w
#define GET_C_NHWC(idx, n, c, h, w) (idx % c)
#define SHIFT_NHWC(ni, ci, hi, wi, n, c, h, w) (((ni * h + hi) * w + wi) * c + ci)

#define GET_N_BACKWARDS_NHWC(idx, n, c) idx
#define GET_C_BACKWARDS_NHWC(idx, n, c) "In the current implementation, it's obtained through the for loop."


#ifdef nhwc
    #define GET_N GET_N_NHWC
    #define GET_C GET_H_NHWC
    #define GET_H GET_W_NHWC
    #define GET_W GET_C_NHWC
    #define SHIFT SHIFT_NHWC
    #define GET_N_BACKWARDS GET_N_BACKWARDS_NHWCS
    #define GET_C_BACKWARDS GET_C_BACKWARDS_NHWCS
#else
    #define GET_N GET_N_NCHW
    #define GET_C GET_C_NCHW
    #define GET_H GET_H_NCHW
    #define GET_W GET_W_NCHW
    #define SHIFT SHIFT_NCHW
    #define GET_N_BACKWARDS GET_N_BACKWARDS_NCHW
    #define GET_C_BACKWARDS GET_C_BACKWARDS_NCHW
#endif

/////////
// FWD //
/////////

__global__ void adaptive_average_pool_2d_fwd(TYPE* x, TYPE* y,
                                             int n, int c,
                                             int hi, int wi,
                                             int ho, int wo,
                                             const int N)
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
        for(i = h_start; i < h_end; i++)
            for(j = w_start; j < w_end; j++)
                add += *(x + SHIFT(ni, ci, i, j, n, c, hi, wi));

        *(y + SHIFT(ni, ci, index_h, index_w, n, c, ho, wo)) = add / elements;
    }
}

/////////
// BWD //
/////////

__global__ void adaptive_average_pool_2d_bwd(TYPE* dx, TYPE* dy,
                                             int n, int c,
                                             int h, int w,
                                             int new_h, int new_w)
{
    int h_start, h_end, elements_kh, w_start, w_end, elements_kw;
    int ni, ci, index_h, index_w, ho, wo;
    TYPE delta;

    #ifdef nhwc
    // NHWC (we are iterating over N)
    const int N = n;
    #else
    //NCHW (we are iterating over NC)
    const int N = n * c;
    #endif

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
        ni = GET_N_BACKWARDS(idx, n, c);
        #ifndef nhwc
        // Case NCHW: the threads will iterate over NC.
        ci = GET_C_BACKWARDS(idx, n, c);
        #endif // The other case logic is placed later in the code.

        for(ho = 0; ho < h; ho++)
        {
            h_start = INDEX_FIRST_ELEMENT(ho, new_h, h);
            h_end = INDEX_LAST_ELEMENT(ho, new_h, h);
            elements_kh = h_end - h_start;

            for(wo = 0; wo < w; wo++)
            {
                w_start = INDEX_FIRST_ELEMENT(wo, new_w, w);
                w_end = INDEX_LAST_ELEMENT(wo, new_w, w);
                elements_kw = w_end - w_start;
                #ifdef nhwc
                // Case NHWC: the threads will iterate over N, so it's necessary to iterate over C too.
                for(ci = 0; ci < c; ci++)
                #endif
                {
                    delta = (TYPE) (*(dy + SHIFT(ni, ci, ho, wo, n, c, h, w)) / (elements_kh * elements_kw));
                    for(index_h = h_start; index_h < h_end; index_h++)
                        for(index_w = w_start; index_w < w_end; index_w++)
                            *(dx + SHIFT(ni, ci, index_h, index_w, n, c, new_h, new_w)) += delta;
                }
            }
        }
    }
}
