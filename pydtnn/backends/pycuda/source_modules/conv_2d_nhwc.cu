#define TYPE "TYPE"

#define DEFINE_BIAS "Replace this line with BIAS_DB if the bias is used."

//// IM2ROW
#define GET_NI(row, h, w) row / (w * h)

#define SHIFT_ROWS(row, col, dim_cols) row * dim_cols + col
// NOTE: This is NHWC
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

#define GET_NO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo * ho))
#define GET_HO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo)) % ho
#define GET_WO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh)) % wo
#define GET_CI(idx, ho, wo, kh, kw, ci) (idx / (kw * kh)) % ci
#define GET_KH(idx, ho, wo, kh, kw, ci) (idx / kw) % kh
#define GET_KW(idx, ho, wo, kh, kw, ci) (idx % kw)

//// ROW2IM
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_DY(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j
#define GET_I(idx, dim_j) idx / dim_j
#define GET_J(idx, dim_j) idx % dim_j

// row2im-related macros
#define GET_N(idx, n, c, h, w) idx / (c * w * h)
#define GET_H(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W(idx, n, c, h, w) (idx / c) % w
#define GET_C(idx, n, c, h, w) idx % c


__global__ void im2row(const TYPE *const x,
                       const TYPE *const weights,
                       TYPE* im2_var, TYPE* y,
                       TYPE* bias,
                       int dim_c, int dim_n,
                       int n, int c, int h, int w,
                       int co, int ho, int wo,
                       int kh, int kw,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation)
{
    int ci, khi, kwi, ni, hoi, hi, wi, woi, idx;
    int i, j, k, i_j;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo * c * kh * kw;
    const int dim_cols = n * ho * wo;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    const int N_matmul = dim_n * co;

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

    // Im2Row
    for(idx = n_offset; idx < end_offset; idx++)
    {
        ni = GET_NO(idx, ho, wo, kh, kw, c);
        hoi = GET_HO(idx, ho, wo, kh, kw, c);
        woi = GET_WO(idx, ho, wo, kh, kw, c);
        ci = GET_CI(idx, ho, wo, kh, kw, c);
        khi = GET_KH(idx, ho, wo, kh, kw, c);
        kwi = GET_KW(idx, ho, wo, kh, kw, c);

        hi = vstride * hoi + vdilation * khi - vpadding;
        wi = hstride * woi + hdilation * kwi - hpadding;

        if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            *(im2_var + idx) = *(x + SHIFT_X(ni, ci, hi, wi, c, h, w));
        else
            *(im2_var + idx) = (TYPE) 0.0;
    }


    __syncthreads();

    overworkers = N_matmul % num_workers;
    samples_worker = N_matmul / num_workers;
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

    // Matmul - im2_var X w_rows = y
    // im_var = (i, k)
    // w_rows = (k, j)
    // y = (i, j)

    // im2_var.shape = (dim_n, dim_c); weights.shape "=" (dim_c, co); y.shape "=" (dim_n * co) || "=": because it's not equal, but "equivalent" in this situation.
    for(i_j = n_offset; i_j < end_offset; i_j++)
        for(k = 0; k < dim_c; k++)
    {
        i = GET_I(i_j, co);
        j = GET_J(i_j, co);
        // y[i, j] += im2_var[i, k] * weights[k, j]
        *(y + i_j) += (*(im2_var + SHIFT(i, k, dim_c))) * (*(weights + SHIFT(k, j, co)));
    }
#ifdef BIAS_DB

    __syncthreads();
    for(i = idx; i < dim_n; i += num_workers)
        for(j = 0; j < co; j++)
    {{
        *(im2_var + SHIFT(i, j, co)) += (*(bias + j));
    }}
#endif

}

////////////////////////////
////////////// ROW2IM 
////////////////////////////


#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_DY(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j
#define GET_I(idx, dim_j) idx / dim_j
#define GET_J(idx, dim_j) idx % dim_j

// im2col-related macros
#define GET_N(idx, n, c, h, w) idx / (c * w * h)
#define GET_H(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W(idx, n, c, h, w) (idx / c) % w
#define GET_C(idx, n, c, h, w) idx % c

__global__ void row2im(const TYPE *const dy,
                       const TYPE *const im2_var,
                       const TYPE *const weights,
                       TYPE* dw, TYPE* db, TYPE* dx,
                       TYPE* row_2im_var,
                       int dim_c, int dim_n,
                       int n, int c, int h, int w,
                       int co, int ho, int wo,
                       int kh, int kw,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N_DW = co * c * kh * kw;
    const int N_ROW2IM_VAR = dim_c * dim_n;
    const int N_ROW2IM = n * c * h * w;

    int i, j, k, i_j, khi, kwi, row, col, ni, hi, wi, ci, x_o, xx, y_o, yy;
    int overworkers, samples_worker, samples_overworker;
    int n_samples, n_offset, end_offset;

    // NOTE: c, h, w are the input ones and co, ho, wo are the output ones (they may differ)
    // base_dy.shape = (n, co, ho, wo)
    // im2_var.shape = (dim_n, dim_c) || dim_n = (n * self.ho * self.wo)
    // weights.shape = (co, kh, kw, c)
    // dy.shape = (n, ho, wo, co)
    // dw.shape = (dim_c, co) || dim_c = (c, kh, kw)
    // db.shape = (co, )
    // dx.shape = (n, c, h, w)
    // row_2im_var.shape = (dim_c, dim_n) || dim_n = n * ho * wo; dim_c = c * kh * kw

    overworkers = N_DW % num_workers;
    samples_worker = N_DW / num_workers;
    samples_overworker = samples_worker + 1;

    if (idx < overworkers)
    {
        n_samples = samples_overworker;
        n_offset = idx * n_samples;
    }
    else
    {
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (idx - overworkers);
    }
    end_offset = n_offset + n_samples;

    // dw = np.matmul(im2_var.T, dy.reshape(n*ho*wo, self.co)); im2_var.T.shape = (ci*kh*kw, n*ho*wo)
    for(i_j = n_offset; i_j < end_offset; i_j ++)
        for(k = 0; k < dim_n; k++)
    {
        i = GET_I(i_j, co);
        j = GET_J(i_j, co);
        *(dw + i_j) += (*(im2_var + SHIFT(k, i, dim_c))) * (*(dy + SHIFT(k, j, co)));
    }

    // np.sum(dy, axis=(0,1,2), out=db)
#ifdef BIAS_DB
    for (ci = idx; ci < c; ci += num_workers)
    {{
        *(db + ci) = 0;
        for (ni = 0; ni < n; ni++)
            for (hi = 0; hi < h; hi++)
                for (wi = 0; wi < w; wi++)
        {{
            *(db + ci) += *(dy + SHIFT_DY(ni, ci, hi, wi, c, h, w));
        }}
    }}
#endif

    overworkers = N_DW % num_workers;
    samples_worker = N_DW / num_workers;
    samples_overworker = samples_worker + 1;

    if (idx < overworkers)
    {
        n_samples = samples_overworker;
        n_offset = idx * n_samples;
    }
    else
    {
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (idx - overworkers);
    }
    end_offset = n_offset + n_samples;


    // row_2im_var "=" (dim_c, dim_n)
    //mamtul(weights.reshape(self.ci * self.kh * self.kw, co), tranposed dy) <== mamtul(weights.reshape(co, -1).T, tranposed dy)
    // tranposed dy.shape = (co, n*ho*wo)
    for(i_j = n_offset; i_j < end_offset; i_j++)
        for(k = 0; k < co; k++)
    {
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);

        //row_2im_var[i][j] += dy[i][k] * weights[k][j]; (weights= weights.reshape((-1, co)).T)
        *(row_2im_var + i_j) += (*(dy + SHIFT(i, k, co))) * (*(weights + SHIFT(j, k, co)));
    }

    __syncthreads();

    // Row2Im

    overworkers = N_ROW2IM % num_workers;
    samples_worker = N_ROW2IM / num_workers;
    samples_overworker = samples_worker + 1;

    if (idx < overworkers)
    {
        n_samples = samples_overworker;
        n_offset = idx * n_samples;
    }
    else
    {
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (idx - overworkers);
    }
    end_offset = n_offset + n_samples;

    for(i = n_offset; i < end_offset; i++)
    {
        ni = GET_N(i, n, h, w, c);
        hi = GET_H(i, n, h, w, c);
        wi = GET_W(i, n, h, w, c);
        ci = GET_C(i, n, h, w, c);

        for(khi = 0; khi < kh; khi++)
            for(kwi = 0; kwi < kw; kwi++)
        {
            x_o = (hi + vpadding - vdilation * khi);
            xx = x_o / vstride;
            x_o = x_o % vstride;

            y_o = (wi + hpadding - hdilation * kwi);
            yy = y_o / hstride;
            y_o = y_o % hstride;

            if((x_o == 0) && (y_o == 0) && IS_BETWEEN(0, xx, ho) && IS_BETWEEN(0, yy, wo))
            {
                row = ni * ho * wo + xx * wo + yy;
                col = ci * kh * kw + khi * kw + kwi;
                *(dx + i) += *(row_2im_var + SHIFT(row, col, dim_c));
            }
        }
    }


}