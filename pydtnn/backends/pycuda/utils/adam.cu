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
#define DECOUPLED_DECAY_OPS "Remove this line to unset decoupled decay"

#define MT(m, beta1, it) (m[i] / (1 - POW(beta1, it)))
#define VT(v, beta2, it) (v[i] / (1 - POW(beta2, it)))

__global__ void adam_gpudirect(TYPE *w, TYPE *dw, TYPE *m, TYPE *v,
                                float it, float lr, float decay,
                                float beta1, float beta2, float epsilon, int N)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;
    int i; 

    for(i = base_idx; i < N; i+= workers)
    {
    #ifdef DECOUPLED_DECAY_OPS
        w[i] *= (1 - lr * decay);
    #else
        dw[i] += decay * w[i];
    #endif

        m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
        v[i] = beta2 * v[i] + (1 - beta2) * POW(dw[i], 2);
        w[i] -= lr * (MT(m, beta1, it) / sqrt(VT(v, beta2, it)) + epsilon);
    }
}