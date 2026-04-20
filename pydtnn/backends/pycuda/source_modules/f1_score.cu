#define TYPE "TYPE"

//#define TRUE_POSITIVE  {0,0}
#define TRUE_POSITIVE_0  0
#define TRUE_POSITIVE_1  0

//#define FALSE_NEGATIVE {0,1}
#define FALSE_NEGATIVE_0 0
#define FALSE_NEGATIVE_1 1

//#define FALSE_POSITIVE {1,0}
#define FALSE_POSITIVE_0 1
#define FALSE_POSITIVE_1 0

#define SHIFT_POINTER_CM(p, label, i, j, num_i, num_j) p + (label * num_i + i) * num_j + j

__global__ void f1_score(TYPE *f1, int *cm, TYPE *local_f1, const int num_classes)
{
    int label, idx, true_positive, false_negative, false_positive, div;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < num_classes; idx += workers)
    {
        *(local_f1 + idx) = 0;
        true_positive = (*(SHIFT_POINTER_CM(cm, label, TRUE_POSITIVE_0, TRUE_POSITIVE_1, 2, 2)));
        false_negative = (*(SHIFT_POINTER_CM(cm, label, FALSE_NEGATIVE_0, FALSE_NEGATIVE_1, 2, 2)));
        false_positive = (*(SHIFT_POINTER_CM(cm, label, FALSE_POSITIVE_0, FALSE_POSITIVE_1, 2, 2)));
        div = 2 * true_positive + false_positive + false_negative;

        (*(local_f1 + idx)) += (TYPE) (div == 0 ? 0 : (2 * true_positive / div));
    }

    // Accumulating the local values into the output's tensor.
    if (base_idx == 0)
    {
        for(idx = 0; label < num_classes; label++)
            (*local_f1) += *(local_f1 + idx);

        (*f1) = (TYPE) ((*local_f1) / num_classes);
    }
}