extern "C"

#define TYPE "TYPE"
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

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

__global__ void average_pool_2d_fwd(TYPE* x, TYPE* y,
                                    int n, int c, int h, int w,
                                    int kh, int kw, int ho, int wo,
                                    int hpadding, int wpadding,
                                    int hstride, int wstride,
                                    int hdilation, int wdilation)
{
    int ni, ci, hoi, woi, khi, kwi, items, hi, wi;
    TYPE accum;

    int idx;

    const int N = n * ho * wo * c;
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
        hoi = GET_H(idx, n, c, ho, wo);
        woi = GET_W(idx, n, c, ho, wo);
        ci = GET_C(idx, n, c, ho, wo);

        accum = (TYPE) 0.0;
        items = 0;
        
        for(khi = 0; khi < kh; khi++)
            for(kwi = 0; kwi < kw; kwi++)
        {
            hi = wstride * hoi + hdilation * khi - hpadding;
            wi = wstride * woi + wdilation * kwi - wpadding;

            if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            {
                accum += (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w)));
                items += 1;
            }
        }
        *(y + idx) = (TYPE) (accum / items);
    }
}

/////////
// BWD //
/////////

__global__ void average_pool_2d_bwd(TYPE* dx, TYPE* dy,
                                    int n, int c, int h, int w,
                                    int kh, int kw, int ho, int wo,
                                    int hpadding, int wpadding,
                                    int hstride, int wstride,
                                    int hdilation, int wdilation)
{
    int ni, ci, hi, wi, hoi, woi, khi, kwi, items;
    int idx, _xx, xx, _yy, yy;
    TYPE accum;

    const int N_avg = n * ho * wo * c;
    const int N_pool = n * h * w * c;
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    // Getting the average

    overworkers = N_avg % num_workers;
    samples_worker = N_avg / num_workers;
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

    // NOTE: This one iterates over dy (n, c, ho, wo || n, ho, wo, c)
    for(idx = n_offset; idx < end_offset; idx++)
    {
        // ni = GET_N(idx, n, c, ho, wo);
        hoi = GET_H(idx, n, c, ho, wo);
        woi = GET_W(idx, n, c, ho, wo);
        // ci = GET_C(idx, n, c, ho, wo);
        
        for(khi = 0; khi < kh; khi++)
        {
            hi = wstride * hoi + hdilation * khi - hpadding;
            for(kwi = 0; (kwi < kw) && (IS_BETWEEN(0, hi, h)); kwi++)
            {
                wi = wstride * woi + wdilation * kwi - wpadding;
                if(IS_BETWEEN(0, wi, w))
                    items += 1;
            }
        }
        *(dy + idx) = (TYPE) ( *(dy + idx) / items);
    }

    // Making the (average) pool
    __syncthreads();

    overworkers = N_pool % num_workers;
    samples_worker = N_pool / num_workers;
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

    // NOTE: This one iterates over dy (n, c, hi, wi || n, hi, wi, c)
    for(idx = n_offset; idx < end_offset; idx++)
    {
        ni = GET_N(idx, n, c, hi, wi);
        ci = GET_C(idx, n, c, hi, wi);
        hi = GET_H(idx, n, c, hi, wi);
        wi = GET_W(idx, n, c, hi, wi);
        
        for(khi = 0; khi < kh; khi++)
        {
            _xx = hi + hpadding - hdilation * khi;
            xx = _xx / hstride;
            _xx %= hstride;

            for(kwi = 0; (kwi < kw) && ((_xx == 0) && IS_BETWEEN(0, xx, ho)); kwi++)
            {
                _yy = wi + wpadding - wdilation * kwi;
                yy = _yy / wstride;
                _yy %= wstride;

                if((_yy == 0) && IS_BETWEEN(0, yy, wo))
                {
                    *(dx + idx) += (*(dy + SHIFT(ni, ci, xx, yy, n, c, ho, wo)));
                }
            }
        }
    }
}
