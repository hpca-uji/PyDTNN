//#define TRUE_POSITIVE  {0,0}
#define TRUE_POSITIVE_0  0
#define TRUE_POSITIVE_1  0

//#define FALSE_NEGATIVE {0,1}
#define FALSE_NEGATIVE_0 0
#define FALSE_NEGATIVE_1 1

#define SHIFT_POINTER_CM(p, label, i, j, num_i, num_j) p + ((label * num_i + i) * num_j + j)

#define TYPE "TYPE"

__global__ void recall(TYPE *recall, int *cm, TYPE *local_recall, const int num_classes)
{
    int label, idx, true_positive, false_negative, div;

    int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    for(idx = base_idx; idx < num_classes; idx += workers)
    {
        *(local_recall + idx) = 0;
        true_positive = (*(SHIFT_POINTER_CM(cm, label, (TRUE_POSITIVE_0), (TRUE_POSITIVE_1), 2, 2)));
        false_negative = (*(SHIFT_POINTER_CM(cm, label, (FALSE_NEGATIVE_0), (FALSE_NEGATIVE_1), 2, 2)));
        div = true_positive + false_negative;

        (*(local_recall + idx)) += (TYPE) (div == 0 ? 0 : (true_positive / div));
    }
    __syncthreads();

    // Accumulating the local values into the output's tensor.
    if (base_idx == 0)
    {
        for(idx = 1; idx < num_classes; idx++)
            (*local_recall) += *(local_recall + idx);

        (*recall) = (TYPE) ((*local_recall) / num_classes);
    }
}