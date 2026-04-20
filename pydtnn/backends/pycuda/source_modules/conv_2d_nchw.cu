#define TYPE "TYPE"

#define DEFINE_BIAS "Replace this line with BIAS_DB if the bias is used."

//// IM2COL
// im2col-related macros
#define GET_CI(row, h, w) row / (w * h)
#define GET_KI(row, h, w) (row / w) % h
#define GET_KJ(row, h, w) row % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_COLS(row, col, dim_cols) row * dim_cols + col
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * c + ci) * h + hi) * w + wi

// col2im-related macros
#define SHIFT_DY(ni, ci, hi, wi, n, c, h, w) (((((ni * c) + ci) * h) + hi) * w + wi)
#define GET_N(idx, n, c, h, w) idx / (w * h * c)
#define GET_C(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H(idx, n, c, h, w) (idx / w) % h
#define GET_W(idx, n, c, h, w) idx % w


__global__ void im2col(const TYPE *const x,
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;

    // im2col const
    const int N = c * kh * kw;
    const int dim_cols = n * ho * wo;
    // matmul const
    const int N_MATMUL = co * dim_n;

    // im2col vars
    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;
    // matmul vars
    int i, j, k;

    // Im2Col
    for(row = idx; row < N; row += num_workers)
    {
        ci = GET_CI(row, h, w);
        ki = GET_KI(row, h, w);
        kj = GET_KJ(row, h, w);
        for (ni = 0; ni < n; ni++) for (hoi = 0; hoi < ho; hoi++)
        {
            hi = vstride * hoi + vdilation * ki - vpadding;
            for (woi = 0; woi < wo; woi++)
            {
                wi = hstride * woi + hdilation * kj - hpadding;
                col = (ni * ho + hoi) * wo + woi;
                //im2_var[row, col] = ((0 <= hi) && (hi < h) && (0 <= wi) && (wi < w)) ? x[nn, cc, x_x, x_y] : (TYPE) 0.0;
                if (IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
                    *(im2_var + SHIFT_COLS(row, col, dim_cols)) = *(x + SHIFT_X(n, ci, hi, wi, c, h, w));
                else
                    *(im2_var + SHIFT_COLS(row, col, dim_cols)) = (TYPE) 0.0;
            }
        }
    }
    __syncthreads();

    // Matmul - w_rows X x_cols = y.T
    // weights.shape "=" (co, dim_c); im2_var.shape = (dim_c, dim_n); y.T "="(co, dim_n); y.shape "=" (dim_n, co) || "=": because it's not equal, but "equivalent" in this situation.
    for(i_j = idx; i_j < N_MATMUL; i_j += num_workers)
    {
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);
        for(k = 0; k < dim_c; k++)
        {
            // y[i, j] += weights[i, k] * im2_var[k, j]
            *(y + SHIFT(j, i, co)) += (*(weights + SHIFT(i, k, dim_c))) * (*(im2_var + SHIFT(k, j, dim_n)));
        }
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

///////////////////////////////
////////////// COL2IM
////////////////////////////

__global__ void col2im(const TYPE *const dy,
                       const TYPE *const im2_var,
                       const TYPE *const weights,
                       TYPE* dw, TYPE* db, TYPE* dx,
                       TYPE* col_2im_var,
                       int dim_c, int dim_n,
                       int n, int c, int h, int w,
                       int co, int ho, int wo,
                       int kh, int kw,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation)
{
    // NOTE: c, h, w are the input ones and co, ho, wo are the output ones (they may differ)
    // base_dy.shape = (n, co, ho, wo)
    // im2_var.shape = (dim_n, co) || dim_n = (n * self.ho * self.wo)
    // weights.shape = (co, c, kh, kw);
    // dy.shape = (co, n, ho, wo)
    // dw.shape = (co, dim_c); dim_c = (c, kh, kw)
    // db.shape = (co, )
    // dx.shape = (n, c, h, w)
    // col_2im_var.shape = (dim_c, dim_n); dim_n = n * ho * wo; dim_c = c * kh * kw

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N_DW = co * c * kh * kw;
    const int N_COL2IM_VAR = dim_c * dim_n;
    const int N_COL2IM = n * c * h * w;
    const int N_TRANSPOSE = n * co * ho * wo;

    int i, j, k, i_j, dim_j, khi, kwi;
    int ni, ci, hi, wi, dy_i;

    // Matmul dy transposed and im2_var.T in and save it in dw
    // NOTE: Here dy is treated as (co, n*ho*wo); im2_var.T.shape = (n*ho*wo, ci*kh*kw)
    dim_j = N_DW / co;

    // NOTE: Remember -> dy base: NCHW, the dy needed to work: CNHW
    //dw.shape - (co, c, kh, kw)

    for(i_j = idx; i_j < N_DW; i_j += workers)
        for(k = 0; k < dim_n; k++)
    {
        i = GET_I(i_j, dim_j);
        j = GET_J(i_j, dim_j);

        // Accessing dy like it was transposed from NCHW to CNHW
        //dy "=" (co, n, self.ho, self.wo)
        //i = "co" = GET_N(k, c, n, h, w).
        ni = GET_C(k, co, n, ho, wo);
        hi = GET_H(k, co, n, ho, wo);
        wi = GET_W(k, co, n, ho, wo);

        *(dw + i_j) += (*(dy + SHIFT_DY(ni, i, hi, wi, co, ho, wo))) * (*(im2_var + SHIFT(k, j, dim_j)));
    }

    // np.sum(dy, axis=(0,2,3), out=db)
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

    // col_2im_var "=" (dim_c, dim_n) = (c * kh * kw, n * ho * wo)
    // mamtul(weights.reshape(co, -1).T, tranposed dy) ==>
    // mamtul(weights.reshape(co, c * kh * kw).T, tranposed dy)
    // tranposed dy.shape = (co, n*ho*wo)
    for(i_j = idx; i_j < N_COL2IM_VAR; i_j += num_workers)
        for(k = 0; k < co; k++)
    {
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);

        // Accessing dy like it was transposed from NCHW to CNHW
        //dy "=" (co, n, self.ho, self.wo)
        dy_i = SHIFT(k, j, dim_n)
        ni = GET_C(dy_i, co, n, ho, wo);
        hi = GET_H(dy_i, co, n, ho, wo);
        wi = GET_W(dy_i, co, n, ho, wo);

        //col_2im_var[i][j] = weights[i][k] * dy[k][j]
        *(col_2im_var + i_j) =  (*(weights + SHIFT(i, k, co))) * (*(dy + SHIFT_DY(ni, k, hi, wi, co, ho, wo)));
    }

    __syncthreads();

    // Col2Im
    for (i = idx; i < N_COL2IM; i += num_workers)
    {
        ni = GET_N(i, n, c, h, w);
        ci = GET_C(i, n, c, h, w);
        hx = GET_H(i, n, c, h, w);
        wx = GET_W(i, n, c, h, w);

        for (khi = 0; khi < kh; khi++)
            for (kwi = 0; kwi < kw; kwi++)
        {
            // hx = vstride * xx + vdilation * khi - vpadding;
            xx = (hx + vpadding - vdilation * khi) / vstride;
            // wx = hstride * yy + hdilation * kwi - hpadding;
            yy = (wx + hpadding - hdilation * kwi) / hstride;

            x_o = (int) xx;
            y_o = (int) yy;

            // if (the variables have no decimals) and (are bewteen 0 and ho/wo):
            if ((x_o == xx) && (y_o == yy) && IS_BETWEEN(0, xx, ho) && IS_BETWEEN(0, yy, wo))
            {
                row = cc * kh * kw + ii * kw + jj;
                col = nn * ho * wo + x_o * wo + y_o;
                //dx[nn, cc, x_x, x_y] += cols[row, col]
                *(dx + i) += (*(cols + SHIFT(row, col, dim_n)));
            }
        }
    }

}
