#define TYPE "TYPE"
/*
#define FLOAT32_POW powf
#define FLOAT64_POW pow
#ifdef FLOAT32
    #define POW FLOAT32_POW
#else
    #define POW FLOAT64_POW
#endif
*/
#define POW powf_or_pow
// "substitude this with the right function"

__global__ void rmsprop_gpudirect(TYPE *w, TYPE *dw, TYPE *cache,
                                  float lr, float decay, float rho,
                                  float epsilon, int N)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {{
        cache[i] = rho * cache[i] + (1 - rho) * POW(dw[i], 2);
        w[i] -= lr * (decay * w[i] + (dw[i] / sqrt(cache[i] + epsilon)));
    }}
}}