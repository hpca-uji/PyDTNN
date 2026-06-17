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

__global__ void cuda_pointwise_conv_2d_fwd(TYPE* x, TYPE* k, TYPE* y,
                                           int n, int c, int h, int w,
                                           int yc, int num_workers)
{
    int idx, ni, ci, hi, wi, yci;
    TYPE val_k, val_x;

    int N = n*c*h*w;

    // k.shape = (yc, x's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_x = *(SHIFT_POINTER(x, c, h, w, ni, ci, hi, wi));
        for(yci = 0; yci < yc; yci++)
        {
            //y = x * k
            //val_k = k[yci][ci]; ==> val_k = k + (yci * c + ci);
            //val_k = k[ci][yci]; ==> val_k = k + (ci * kc + yci);
            val_k = *(SHIFT_POINTER_K(k, c, yc, ci, yci));
            // TODO: Check if this is correct.
            *(SHIFT_POINTER(y, yc, h, w, ni, yci, hi, wi)) += (TYPE) (val_x * val_k);
        }
    }
}

__global__ void cuda_pointwise_conv_2d_bwd(TYPE* dy, TYPE* x, TYPE* k,
                                           TYPE* dx, TYPE* dw,
                                           int n, int c, int h, int w,
                                           int xc, int num_workers)
{
    int idx, ni, ci, hi, wi, xci;
    TYPE val_dy, val_k, val_x;

    int N = n*c*h*w;

    // NCHW: k.shape = dw.shape = (dy's c , x's c)
    // NHWC: k.shape = dw.shape = (x's c, dy's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_dy = *SHIFT_POINTER(dy, c, h, w, ni, ci, hi, wi);
        for(xci = 0; xci < kc; xci++)
        {   
            // TODO: Check if this is correct,
            //dw = x * dy
            val_x = *(SHIFT_POINTER(x, xc, h, w, ni, xci, hi, wi));
            *(SHIFT_POINTER_K(dw, c, xc, ci, xci)) += (TYPE) (val_x * val_dy);

            //dx = w * dy
            val_k = *(SHIFT_POINTER_K(k, c, xc, ci, xci));
            // TODO: Check if this is correct,
            *(SHIFT_POINTER(dx, kc, h, w, nn, xci, hi, wi)) += (TYPE) (val_k * val_dy);
        }
    }
}


__global__ void cuda_bias_sum_fwd_pointwise_conv(TYPE* y, TYPE* b,
                                                 int n, int c, int h, int w,
                                                 int N,
                                                 int num_workers)
{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        
        ci = INDEX_C(idx, c, h, w);
        (*(y + idx)) += (*(b + ci));
    }
}