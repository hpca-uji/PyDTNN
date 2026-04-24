extern "C"

//
#define TYPE "TYPE"
#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
/// im2row (NHWC)
#define SHIFT_IM2ROW(ni, ci, hi, wi, n, c, h, w) ((ni * h + hi) * w + wi) * c + ci
#define GET_N_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh * wo * ho))
#define GET_HO_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh * wo)) % ho
#define GET_WO_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh)) % wo
#define GET_CI_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh)) % ci
#define GET_KH_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx / kw) % kh
#define GET_KW_IM2ROW(idx, n, ci, ho, wo, kh, kw) (idx % kw)
#define NAME_FUNC_FWD_NHWC im2row

/// im2col (NCHW)
#define SHIFT_IM2COL(ni, ci, hi, wi, n, c, h, w) ((ni * c + ci) * h + hi) * w + wi
#define GET_N_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo * ho * ci))
#define GET_HO_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo * ho)) % ci
#define GET_WO_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo)) % ho
#define GET_CI_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh)) % wo
#define GET_KH_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx / kw) % kh
#define GET_KW_IM2COL(idx, n, ci, ho, wo, kh, kw) (idx % kw)
#define NAME_FUNC_FWD_NCHW im2col

#ifdef nhwc
    #define SHIFT SHIFT_IM2ROW
    #define GET_N GET_N_IM2ROW
    #define GET_HO GET_HO_IM2ROW
    #define GET_WO GET_WO_IM2ROW
    #define GET_CI GET_CI_IM2ROW
    #define GET_KH GET_KH_IM2ROW
    #define GET_KW GET_KW_IM2ROW
    #define NAME_FUNC_FWD NAME_FUNC_FWD_NHWC
#else
    #define SHIFT SHIFT_IM2COL
    #define GET_N GET_N_IM2COL
    #define GET_HO GET_HO_IM2COL
    #define GET_WO GET_WO_IM2COL
    #define GET_CI GET_CI_IM2COL
    #define GET_KH GET_KH_IM2COL
    #define GET_KW GET_KW_IM2COL
    #define NAME_FUNC_FWD NAME_FUNC_FWD_NCHW
#endif


__global__ void im2_row_col(const TYPE *const x,
                            TYPE* rows,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int hstride, int wstride,
                            int vdilation, int hdilation)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo * c * kh * kw;

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi;
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
        ni = GET_N(idx, n, c, ho, wo, kh, kw);
        hoi = GET_HO(idx, n, c, ho, wo, kh, kw);
        woi = GET_WO(idx, n, c, ho, wo, kh, kw);
        ci = GET_CI(idx, n, c, ho, wo, kh, kw);
        khi = GET_KH(idx, n, c, ho, wo, kh, kw);
        kwi = GET_KW(idx, n, c, ho, wo, kh, kw);

        hi = hstride * hoi + vdilation * khi - vpadding;
        wi = wstride * woi + hdilation * kwi - hpadding;

        if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            *(rows + idx) = *(x + SHIFT(ni, ci, hi, wi, n, c, h, w));
        else
            *(rows + idx) = (TYPE) 0.0;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SHIFT_BWD(i, j, dim_j) (i * dim_j) + j
#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw

// row2im

#define GET_N_ROW2IM(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H_ROW2IM(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W_ROW2IM(idx, n, c, h, w) (idx / c) % w
#define GET_C_ROW2IM(idx, n, c, h, w) idx % c
#define NAME_FUNC_BWD_NHWC row2im

// col2im

#define GET_N_COL2IM(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H_COL2IM(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W_COL2IM(idx, n, c, h, w) (idx / c) % w
#define GET_C_COL2IM(idx, n, c, h, w) idx % c
#define NAME_FUNC_BWD_NCHW col2im


#ifdef nchw
    #define GET_N_BWD GET_N_ROW2IM
    #define GET_H GET_H_ROW2IM
    #define GET_W GET_W_ROW2IM
    #define GET_C GET_C_ROW2IM
    #define NAME_FUNC_BWD NAME_FUNC_BWD_NHWC
#else
    #define GET_N_BWD GET_N_COL2IM
    #define GET_H GET_H_COL2IM
    #define GET_W GET_W_COL2IM
    #define GET_C GET_C_COL2IM
    #define NAME_FUNC_BWD NAME_FUNC_BWD_NCHW
#endif


#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void row_col_2im(const TYPE *const rows,
                            TYPE* dx,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int hstride, int wstride,
                            int vdilation, int hdilation)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * h * w * c;
    const int num_cols = GET_COLS(c, kh, kw);

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi, _hoi, _woi, row, col;
    int n_samples, n_offset, end_offset;
    int samples_worker, samples_overworker, overworkers;

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
        ni = GET_N_BWD(idx, n, c, h, w);
        hi = GET_H(idx, n, c, h, w);
        wi = GET_W(idx, n, c, h, w);
        ci = GET_C(idx, n, c, h, w);

        for(khi = 0; khi < kh; khi++)
        {
            _hoi = (hi + vpadding - vdilation * khi);
            hoi = _hoi / hstride;
            _hoi = _hoi % hstride;

            for(kwi = 0; (kwi < kw) && ((_hoi == 0) && IS_BETWEEN(0, hoi, ho)); kwi++)
            {
                _woi = (wi + hpadding - hdilation * kwi);
                woi = _woi / wstride;
                _woi = _woi % wstride;

                if((_woi == 0) && IS_BETWEEN(0, woi, wo))
                {
                    row = GET_ROW(ni, hoi, woi, ho, wo);
                    col = GET_COL(ci, khi, kwi, kh, kw);
                    *(dx + idx) += *(rows + SHIFT_BWD(row, col, num_cols));
                }
            }
        }
    }
}