
extern "C"

#define TYPE "TYPE"

#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

#define SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw) ((ci * kh + khi) * kw + kwi)
#define GET_C_WEIGHTS(idx, c, kh, kw) (idx / (kw * kh))
#define GET_H_WEIGHTS(idx, c, kh, kw) (idx / kw) % kh
#define GET_W_WEIGHTS(idx, c, kh, kw) (idx % kw)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

#define SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw) ((ci * kh + khi) * kw + kwi)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

#define GET_N_NCHW(idx, n, c, h, w) (idx / (w * h * c))
#define GET_C_NCHW(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H_NCHW(idx, n, c, h, w) (idx / w) %h
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

__global__ void conv_depthwise_fwd(TYPE* x, TYPE* weights, TYPE* y,
                                   int n, int c, int h, int w,
                                   int ho, int wo, int kh, int kw,
                                   int hpadding, int wpadding,
                                   int hstride, int wstride,
                                   int hdilation, int wdilation)
{
    int ni, ci, khi, kwi, hoi, woi;
    int idx, hi, wi;

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
        ci = GET_CI(idx, n, c, ho, wo);
        
        for(khi = 0; khi < kh; khi++)
        {
            hi = wstride * hoi + hdilation * khi - hpadding;
            for(kwi = 0; (kwi < kw) && IS_BETWEEN(0, hi, h); kwi++)
            {
                wi = wstride * woi + wdilation * kwi - wpadding;
                if(IS_BETWEEN(0, wi, w))
                    *(y + idx) += (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w))) * (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw)));
            }
        }
    }
}

/////////
// BWD //
/////////

__global__ void conv_depthwise_bwd(TYPE* dx, TYPE* dy, TYPE* x,
                                   TYPE* dw, TYPE* weights,
                                   int n, int c, int h, int w,
                                   int ho, int wo, int kh, int kw,
                                   int vpadding, int wpadding,
                                   int hstride, int wstride,
                                   int hdilation, int wdilation)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * h * w * c;
    const int N_W = c * kh * kw;

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi, _hoi, _woi;
    int n_samples, n_offset, end_offset;
    int samples_worker, samples_overworker, overworkers;

    // Input gradient (dx)

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
        ni = GET_N(idx, n, c, h, w);
        hi = GET_H(idx, n, c, h, w);
        wi = GET_W(idx, n, c, h, w);
        ci = GET_C(idx, n, c, h, w);

        for(khi = 0; khi < kh; khi++)
        {
            _hoi = (hi + vpadding - hdilation * khi);
            hoi = _hoi / wstride;
            _hoi = _hoi % wstride;

            for(kwi = 0; (kwi < kw) && ((_hoi == 0) && IS_BETWEEN(0, hoi, ho)); kwi++)
            {
                _woi = (wi + wpadding - wdilation * kwi);
                woi = _woi / wstride;
                _woi = _woi % wstride;

                if((_woi == 0) && IS_BETWEEN(0, woi, wo))
                    *(dx + idx) += (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, n, c, kh, kw))) * (*(dy + SHIFT(ni, ci, hi, wi, n, c, h, w)));
            }
        }
    }

    // Weights gradient (dw)

    overworkers = N_W % num_workers;
    samples_worker = N_W / num_workers;
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
        ci = GET_C_WEIGHTS(idx, c, kh, kw);
        khi = GET_H_WEIGHTS(idx, c, kh, kw);
        kwi = GET_W_WEIGHTS(idx, c, kh, kw);

        for(ni = 0; ni < n; ni++)
        {
            for(hoi = 0; hoi < ho; hoi++)
            {
                hi = hstride * hoi + hdilation * khi - hpadding;
                for(woi = 0; (woi < wo) && (IS_BETWEEN(0, hi, h)); woi++)
                {
                    
                    wi = wstride * woi + wdilation * kwi - wpadding;
                    if(IS_BETWEEN(0, wi, w))
                        *(dw + idx) += (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw))) * (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w)));
                }
            }
        }
    }
}
