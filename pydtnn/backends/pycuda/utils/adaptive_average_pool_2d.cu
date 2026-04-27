#define TYPE "TYPE"

#define TENSOR_FORMAT "Replace this define with the actual tensor format (nchw or nhwc)"

#define MACRO_INDEX_N(idx, N, n) idx * n / N
#define INDEX_C_NCHW(idx, c, h, w) (idx / (h * w)) % c
#define INDEX_H_NCHW(idx, c, h, w) (idx / w) % h
#define INDEX_W_NCHW(idx, c, h, w) idx % w
#define SHIFT_POINTER_NCHW(p, c, h, w, ni, ci, hi, wi) p + ((ni * c + ci) * h + hi) * w + wi

#define INDEX_H_NHWC(idx, c, h, w) (idx / (w * c)) % h
#define INDEX_W_NHWC(idx, c, h, w) (idx / c) % w
#define INDEX_C_NHWC(idx, c, h, w) idx % c
#define SHIFT_POINTER_NHWC(p, c, h, w, ni, ci, hi, wi) p + ((ni * h + hi) * w + wi) * c + ci

#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) (int) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) (int) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)

#define TRUE  1
#define FALSE 0

#ifdef nhwc
    #define INDEX_H INDEX_H_NHWC
    #define INDEX_W INDEX_W_NHWC
    #define INDEX_C INDEX_C_NHWC
    #define SHIFT_POINTER SHIFT_POINTER_NHWC
#else
    #define INDEX_C INDEX_C_NCHW
    #define INDEX_H INDEX_H_NCHW
    #define INDEX_W INDEX_W_NCHW
    #define SHIFT_POINTER SHIFT_POINTER_NCHW
#endif


__global__ void adaptive_average_pool_2d_fwd(TYPE* x, TYPE* y,
                                             int n, int c, int h, int w,
                                             int new_h, int new_w, int N,
                                             int num_active_workers,
                                             int num_ops_per_worker,
                                             int num_ops_last_worker)
{
    int idx, ops_remaining;
    int ni, ci, wi, hi, i, j;
    int h_start, h_end, w_start, w_end, elements_h, elements;
    unsigned short first_iteration;
    TYPE add;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    ops_remaining = ((idx + 1) == num_active_workers) ? num_ops_last_worker : num_ops_per_worker;
    idx *= num_ops_per_worker;

    ni = MACRO_INDEX_N(idx, N, n);
    ci = INDEX_C_NCHW(idx, c, new_h, new_w);
    hi = INDEX_H_NCHW(idx, c, new_h, new_w);
    wi = INDEX_W_NCHW(idx, c, new_h, new_w);
    first_iteration = TRUE;

    for(ni = ni;
        (ni < n) && (ops_remaining > 0);
        ni++)
    {
        for(ci = (first_iteration ? ci : 0);
            (ci < c) && (ops_remaining > 0);
            ci++)
        {
            for(hi = (first_iteration ? hi : 0);
                (hi < new_h) && (ops_remaining > 0);
                hi++)
            {
                h_start = INDEX_FIRST_ELEMENT(hi, h, new_h);
                h_end = INDEX_LAST_ELEMENT(hi, h, new_h);
                elements_h = h_end - h_start;

                for(wi = (first_iteration ? wi : 0), first_iteration = FALSE;
                    (wi < new_w) && (ops_remaining > 0);
                    wi++, ops_remaining--)
                {
                    w_start = INDEX_FIRST_ELEMENT(wi, w, new_w);
                    w_end = INDEX_LAST_ELEMENT(wi, w, new_w);
                    elements = elements_h * (w_end - w_start);

                    for(i = h_start, add = (TYPE) 0.0; i < h_end; i++)
                        for(j = w_start; j < w_end; j++)
                            add += (TYPE) (*(SHIFT_POINTER(x, c, h, w, ni, ci, i, j)) );

                    (*(SHIFT_POINTER(y, c, new_h, new_w, ni, ci, hi, wi))) = (TYPE) (add / elements);
                }
            }
        }
    }
}

////

__global__ void adaptive_average_pool_2d_bwd(TYPE* dx, TYPE* dy,
                                             int n, int c, int h, int w,
                                             int new_h, int new_w, int N,
                                             int num_active_workers,
                                             int num_ops_per_worker,
                                             int num_ops_last_worker)
{
    int idx, ops_remaining;
    int ni, ci, wi, hi, i, j;
    int h_start, h_end, w_start, w_end, elements_h, elements;
    unsigned short first_iteration;
    TYPE delta;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    ops_remaining = ((idx + 1) == num_active_workers) ? num_ops_last_worker : num_ops_per_worker;
    idx *= num_ops_per_worker;

    ni = MACRO_INDEX_N(idx, N, n);
    ci = INDEX_C_NCHW(idx, c, new_h, new_w);
    hi = INDEX_H_NCHW(idx, c, new_h, new_w);
    wi = INDEX_W_NCHW(idx, c, new_h, new_w);
    first_iteration = TRUE;

    for(ni = ni;
        (ni < n) && (ops_remaining > 0);
        ni++)
    {
        for(ci = (first_iteration ? ci : 0);
            (ci < c) && (ops_remaining > 0);
            ci++)
        {
            for(hi = (first_iteration ? hi : 0);
                (hi < h) && (ops_remaining > 0);
                hi++)
            {
                h_start = INDEX_FIRST_ELEMENT(hi, new_h, h);
                h_end = INDEX_LAST_ELEMENT(hi, new_h, h);
                elements_h = h_end - h_start;

                for(wi = (first_iteration ? wi : 0), first_iteration = FALSE;
                    (wi < w) && (ops_remaining > 0);
                    wi++, ops_remaining--)
                {
                    w_start = INDEX_FIRST_ELEMENT(wi, new_w, w);
                    w_end = INDEX_LAST_ELEMENT(wi, new_w, w);
                    elements = elements_h * (w_end - w_start);

                    delta = (TYPE) (*(SHIFT_POINTER(dy, c, new_h, new_w, ni, ci, hi, wi)) / elements);
                    for(i = h_start; i < h_end; i++)
                        for(j = w_start; j < w_end; j++)
                            (*(SHIFT_POINTER(dx, c, h, w, ni, ci, i, j))) += delta;
                }
            }
        }
    }
}

