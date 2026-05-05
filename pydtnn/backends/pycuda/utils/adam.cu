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
// "substitude the previous define with the right function"


__global__ void adam_gpudirect(TYPE *w, TYPE *dw, TYPE *m, TYPE *v,
                                float it, float lr, float decay,
                                float beta1, float beta2, float epsilon, int N)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {{
        m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
        v[i] = beta2 * v[i] + (1 - beta2) * POW(dw[i], 2);
        w[i] -= lr * (decay * w[i] + ((m[i] / (1 - POW(beta1, it))) / sqrt(v[i] / (1 - POW(beta2, it)) + epsilon)));
    }}
}}