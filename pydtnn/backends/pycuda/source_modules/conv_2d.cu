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


__global__ void cuda_sum_bias_axis_023(TYPE* dy, TYPE* db
                                       int c, int h, int w,
                                       int N, int num_workers)
{
    int idx, index_c;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        index_c = (idx / (h*w)) % c;
        *(db + index_c) += *(dy + idx);
    }
}

////

__global__ void cuda_sum_bias_axis_012(TYPE* dy, TYPE* db,
                                       int c, int N,
                                       int num_workers)
{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {
        *(db + (idx % c)) += *(dy + idx);
    }
}
