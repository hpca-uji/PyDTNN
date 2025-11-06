# Tests

## Main
```sh
python -m unittest -v pydtnn.tests
```
_Note: exhaustive tests are skipped_

## Exhaustive
```sh
python -m unittest -v pydtnn.tests.conv2d_conv_gemm_slow.Conv2DConvGemmSlowTestCase.test_forward_backward_multiple_params
```

## CUDA
```sh
python -m unittest -v pydtnn.tests.model_gpu
```

## Specific
```sh
python -m unittest -v pydtnn.tests.${TEST_NAME}
```
