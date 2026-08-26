#define TYPE "TYPE"

__global__ void cross_entropy(TYPE *y_targ, TYPE *y_pred, TYPE *loss,
                              TYPE *weights, TYPE *dx, int *argmax,
                              TYPE *sample_weights, int b, int n)
{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int workers = blockDim.x * gridDim.x;

    int idx, i, max_idx;

    TYPE max_value, target_logit, max_logit, sum_exp, weight, exp_value;

    for (idx = base_idx; idx < b; idx += workers)
    {
        // Target class
        max_idx = 0;
        max_value = y_targ[idx * n];

        for (i = 1; i < n; i++)
        {
            if (y_targ[idx * n + i] > max_value)
            {
                max_idx = i;
                max_value = y_targ[idx * n + i];
            }
        }

        argmax[idx] = max_idx;

        // Weight associated with this sample.
        weight = weights[max_idx];
        sample_weights[idx] = weight;

        // Stable Softmax: max(logits)
        max_logit = y_pred[idx * n];

        for (i = 1; i < n; i++)
        {
            TYPE value = y_pred[idx * n + i];
            if (value > max_logit)
            {
                max_logit = value;
            }
        }

        // Stable Softmax denominator
        // sum(exp(logits - max_logit))
        sum_exp = (TYPE) 0;

        for (i = 0; i < n; i++)
        {
            sum_exp += exp(y_pred[idx * n + i] - max_logit);
        }

        // Loss
        //
        // weighted log_softmax[target]
        // log_softmax[target] = target_logit - max_logit - log(sum_exp)
        target_logit = y_pred[idx * n + max_idx];
        loss[idx] = weight * (target_logit - max_logit - log(sum_exp));

        // DX
        // weight * (softmax(logits) - target)
        // The final division by sum(sample_weights) is performed outside the kernel.
        for (i = 0; i < n; i++)
        {
            exp_value = exp(y_pred[idx * n + i] - max_logit);
            dx[idx * n + i] = weight * (exp_value / sum_exp - y_targ[idx * n + i]);
        }
    }
    return;
}