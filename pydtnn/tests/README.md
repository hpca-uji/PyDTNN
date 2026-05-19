# Tests

## Main
```sh
python -m unittest pydtnn.tests.groups.all
```

_Note: include `-v` for verbose mode_

_Note: exhaustive tests are skipped_

## Exhaustive
```sh
python -m unittest pydtnn.tests.conv2d_cython.Conv2DCythonTestCase
python -m unittest pydtnn.tests.conv2d_conv_gemm_long.Conv2DConvGemmLongTestCase.test_forward_backward_multiple_params
```

## CUDA
```sh
python -m unittest pydtnn.tests.model_gpu
```

## Specific
```sh
python -m unittest pydtnn.tests.${TEST_NAME}
```
