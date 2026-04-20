#define TYPE "TYPE"

#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

#define SHIFT_POINTER_NHWC(p, c, h, w, ni, ci, hi, wi) p + ((ni * h + hi) * w + wi) * c + ci
#define SHIFT_POINTER_K_NHWC(p, c, yc, ci, yci) p + (ci * yc + yci)
#define INDEX_N_NHWC(idx, N, n) idx * n / N
#define INDEX_H_NHWC(idx, h, w, c) (idx / (w * c)) % h
#define INDEX_W_NHWC(idx, h, w, c) (idx / c) % w
#define INDEX_C_NHWC(idx, h, w, c) idx % c

#define SHIFT_POINTER_NCHW(p, c, h, w, ni, ci, hi, wi) p + ((ni * c + ci) * h + hi) * w + wi
#define SHIFT_POINTER_K_NCHW(p, c, yc, ci, yci) p + (yci * c + ci)
#define INDEX_N_NCHW(idx, N, n) idx * n / N
#define INDEX_C_NCHW(idx, c, h, w) (idx / (h * w)) % c
#define INDEX_H_NCHW(idx, c, h, w) (idx / w) % h
#define INDEX_W_NCHW(idx, c, h, w) idx % w

#ifdef nhwc
    #define SHIFT_POINTER SHIFT_POINTER_NHWC
    #define SHIFT_POINTER_K SHIFT_POINTER_K_NHWC
    #define INDEX_N INDEX_N_NHWC
    #define INDEX_C INDEX_C_NHWC
    #define INDEX_H INDEX_H_NHWC
    #define INDEX_W INDEX_W_NHWC
#else
    #define SHIFT_POINTER SHIFT_POINTER_NCHW
    #define SHIFT_POINTER_K SHIFT_POINTER_K_NCHW
    #define INDEX_N INDEX_N_NCHW
    #define INDEX_C INDEX_C_NCHW
    #define INDEX_H INDEX_H_NCHW
    #define INDEX_W INDEX_W_NCHW
#endif

__global__ void cuda_depthwise_conv_2d_fwd(TYPE* x, TYPE* k, TYPE* res,
                                           int vpadding, int hpadding,
                                           int vstride, int hstride,
                                           int vdilation, int hdilation,
                                           int n, int c, int h, int w,
                                           int kh, int kw, int ho, int wo,
                                           int num_workers)
{
    int idx, cc, hi, wi, yy, xx, nn, x_x, x_y;
    int N = n * c * ho * wo;
    TYPE val_k, val_x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        for (hi = 0; hi < kh; hi++)
        {
            for (wi = 0; wi < kw; wi++)
            {
                x_x = vstride * xx + vdilation * hi - vpadding;
                x_y = hstride * yy + hdilation * wi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w))
                {
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, hi, wi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(res, c, h, w, nn, cc, xx, yy)) += (TYPE) (val_k * val_x);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////

__global__ void cuda_depthwise_conv_2d_bwd(TYPE* dy, TYPE* x, TYPE* k,
                                           TYPE* dx, TYPE* dw,
                                           int vpadding, int hpadding,
                                           int vstride, int hstride,
                                           int vdilation, int hdilation,
                                           int n, int c, int h, int w,
                                           int kh, int kw, int ho, int wo,
                                           int num_workers)
{
    int idx, cc, khi, kwi, yy, xx, nn, x_x, x_y;
    TYPE val_k, val_dy, val_x;
    int N = n * c * ho * wo;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        val_dy = (TYPE) *(SHIFT_POINTER(dy, c, h, w, nn, cc, xx, yy));
        for (khi = 0; khi < kh; khi++)
        {
            for (kwi = 0; kwi < kw; kwi++)
            {
                x_x = vstride * xx + vdilation * khi - vpadding;
                x_y = hstride * yy + hdilation * kwi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w)){
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, khi, kwi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(dw, c, h, w, 0, cc, khi, kwi)) = (TYPE) (val_x * val_dy);
                    *(SHIFT_POINTER(dx, c, h, w, nn, cc, x_x, x_y)) += (TYPE) (val_k * val_dy);
                }
            }
        }
    }
}


////////

__global__ void cuda_bias_sum_fwd_depthwise_conv(TYPE* x, TYPE* bias,
                                                 int co, int N,
                                                 int num_workers)
{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        *(x + idx) += *(bias + ( idx / (N/co) ) );
    }
}