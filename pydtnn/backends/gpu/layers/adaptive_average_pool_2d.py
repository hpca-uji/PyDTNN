from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.backends.gpu.layers.layer import LayerGPU

# Import from AbstractPool2DLayerGPU
from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.performance_models import im2col_time, col2im_time
import pycuda.gpuarray as gpuarray   # type: ignore
from pycuda.compiler import SourceModule   # type: ignore
from pycuda.driver import Function   # type: ignore

import numpy as np
from pydtnn.utils.tensor import TensorFormat

# --- CONSTANTS --- #
_MACRO_INDEX_FIRST_ELEMENT = "INDEX_FIRST_ELEMENT"
_MACRO_INDEX_LAST_ELEMENT = "INDEX_LAST_ELEMENT"
_MACRO_INDEX_N = "INDEX_N"
_MACRO_INDEX_C = "INDEX_C"
_MACRO_INDEX_H = "INDEX_H"
_MACRO_INDEX_W = "INDEX_W"
_MACRO_SHIFT_POINTER = "SHIFT_POINTER"
_FULL_MACRO_SHIFT_POINTER = f"#define {_MACRO_SHIFT_POINTER}(p, c, h, w, ni, ci, hi, wi)"
_SHIFT_POINTER_NCHW = "p + ((ni * c + ci) * h + hi) * w + wi"
_SHIFT_POINTER_NHWC = "p + ((ni * h + hi) * w + wi) * c + ci"

_FULL_MACRO_INDEX_C_NCHW = f"#define {_MACRO_INDEX_C}(idx, c, h, w) (idx / (h * w)) % c"
_FULL_MACRO_INDEX_H_NCHW = f"#define {_MACRO_INDEX_H}(idx, h, w) (idx / w) % h"
_FULL_MACRO_INDEX_W_NCHW = f"#define {_MACRO_INDEX_W}(idx, w) idx % w"
_DIMENSION_INDEX_CODE_NCHW = \
"""
ci = {macro_index_c}(idx, c, new_h, new_w);
hi = {macro_index_h}(idx, new_h, new_w);
wi = {macro_index_w}(idx, new_w);
"""
_FULL_MACRO_INDEX_H_NHWC = f"#define {_MACRO_INDEX_H}(idx, h, w, c) (idx / (w * c)) % h"
_FULL_MACRO_INDEX_W_NHWC = f"#define {_MACRO_INDEX_W}(idx, w, c) (idx / c) % w"
_FULL_MACRO_INDEX_C_NHWC = f"#define {_MACRO_INDEX_C}(idx, c) idx % c"
_DIMENSION_INDEX_CODE_NHWC = \
"""
hi = {macro_index_h}(idx, new_h, new_w, c);
wi = {macro_index_w}(idx, new_w, c);
ci = {macro_index_c}(idx, c);
"""


class AdaptiveAveragePool2DGPU(AdaptiveAveragePool2D[TensorGPU], LayerGPU):

    def initialize(self, prev_shape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.cuda_fwd_func = self.cuda_adaptive_average_pooling_fwd(dtype=self.model.dtype)
        self.cuda_bwd_func = self.cuda_adaptive_average_pooling_bwd(dtype=self.model.dtype)

        self.initialize_pool_2d_gpu(prev_shape, x)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # NOTE: Will be initalized later.
        self.y = None  # type: ignore

    def cuda_adaptive_average_pooling_fwd(self, dtype: np.dtype) -> Function:

        _func_name = ["cuda_adaptive_average_pooling_fwd"]
        _t = DTYPE2CTYPE[dtype]  # variable Type
        _full_macro_index_c = ""
        _full_macro_index_h = ""
        _full_macro_index_w = ""
        _dimension_index_code = ""
        _full_macro_shift_pointer = [_FULL_MACRO_SHIFT_POINTER]

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                _func_name.append("_nchw")
                _full_macro_shift_pointer.append(_SHIFT_POINTER_NCHW)
                _full_macro_index_c = _FULL_MACRO_INDEX_C_NCHW
                _full_macro_index_h = _FULL_MACRO_INDEX_H_NCHW
                _full_macro_index_w = _FULL_MACRO_INDEX_W_NCHW
                _dimension_index_code = _DIMENSION_INDEX_CODE_NCHW
                # -- END cuda_adaptive_average_pooling_fwd_nchw --
            case TensorFormat.NHWC:
                # NOTE: It has been tested and it return values that seems to make sense,
                #   but they hadn't been compared with other model's output due the format (Torch is NCHW).
                _func_name.append("_nhwc")
                _full_macro_shift_pointer.append(_SHIFT_POINTER_NHWC)
                _full_macro_index_h = _FULL_MACRO_INDEX_H_NHWC
                _full_macro_index_w = _FULL_MACRO_INDEX_W_NHWC
                _full_macro_index_c = _FULL_MACRO_INDEX_C_NHWC
                _dimension_index_code = _DIMENSION_INDEX_CODE_NHWC
                # -- END cuda_adaptive_average_pooling_fwd_nhwc --
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} is not an implemented format.")

        _func_name = "".join(_func_name)
        _full_macro_shift_pointer = "".join(_full_macro_shift_pointer)
        _dimension_index_code = _dimension_index_code.format(macro_index_c=_MACRO_INDEX_C,
                                                             macro_index_h=_MACRO_INDEX_H,
                                                             macro_index_w=_MACRO_INDEX_W)

        code = """
#define {macro_index_n}(idx, N, n) idx * n / N
{full_macro_index_c}
{full_macro_index_h}
{full_macro_index_w}

#define {macro_index_first_element}(index, dim_in, dim_out) (int) ((index * dim_in) / dim_out)
#define {macro_index_last_element}(index, dim_in, dim_out) (int) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)
{full_macro_shift_pointer}

#define TRUE  1
#define FALSE 0

__global__ void {func_name}({T}* x, {T}* y,
                            int n, int c, int h, int w,
                            int new_h, int new_w, int N,
                            int num_active_workers,
                            int num_ops_per_worker,
                            int num_ops_last_worker)
{{
    int idx, ops_remaining;
    int ni, ci, wi, hi, i, j;
    int h_start, h_end, w_start, w_end, elements_h, elements;
    unsigned short first_iteration;
    {T} add;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    ops_remaining = ((idx + 1) == num_active_workers) ? num_ops_last_worker : num_ops_per_worker;
    idx *= num_ops_per_worker;

    ni = {macro_index_n}(idx, N, n);
    {dimension_index_code}
    first_iteration = TRUE;

    for(ni = ni;
        (ni < n) && (ops_remaining > 0);
        ni++)
    {{
        for(ci = (first_iteration ? ci : 0);
            (ci < c) && (ops_remaining > 0);
            ci++)
        {{
            for(hi = (first_iteration ? hi : 0);
                (hi < new_h) && (ops_remaining > 0);
                hi++)
            {{
                h_start = {macro_index_first_element}(hi, h, new_h);
                h_end = {macro_index_last_element}(hi, h, new_h);
                elements_h = h_end - h_start;

                for(wi = (first_iteration ? wi : 0), first_iteration = FALSE;
                    (wi < new_w) && (ops_remaining > 0);
                    wi++, ops_remaining--)
                {{
                    w_start = {macro_index_first_element}(wi, w, new_w);
                    w_end = {macro_index_last_element}(wi, w, new_w);
                    elements = elements_h * (w_end - w_start);

                    for(i = h_start, add = ({T}) 0.0; i < h_end; i++)
                        for(j = w_start; j < w_end; j++)
                            add += ({T}) (*({macro_desp_pointer}(x, c, h, w, ni, ci, i, j)) );

                    (*({macro_desp_pointer}(y, c, new_h, new_w, ni, ci, hi, wi))) = ({T}) (add / elements);
                }}
            }}
        }}
    }}
}}
"""

        code = code.format(full_macro_index_c=_full_macro_index_c,
                           full_macro_index_h=_full_macro_index_h,
                           full_macro_index_w=_full_macro_index_w,
                           full_macro_shift_pointer=_full_macro_shift_pointer,
                           macro_index_n=_MACRO_INDEX_N,
                           macro_index_c=_MACRO_INDEX_C,
                           macro_index_h=_MACRO_INDEX_H,
                           macro_index_w=_MACRO_INDEX_W,
                           macro_index_first_element=_MACRO_INDEX_FIRST_ELEMENT,
                           macro_index_last_element=_MACRO_INDEX_LAST_ELEMENT,
                           macro_desp_pointer=_MACRO_SHIFT_POINTER,
                           dimension_index_code=_dimension_index_code,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module

    def cuda_adaptive_average_pooling_bwd(self, dtype: np.dtype) -> Function:

        _func_name = ["cuda_adaptive_average_pooling_bwd"]
        _t = DTYPE2CTYPE[dtype]  # variable Type
        _full_macro_index_c = ""
        _full_macro_index_h = ""
        _full_macro_index_w = ""
        _dimension_index_code = ""
        _full_macro_shift_pointer = [_FULL_MACRO_SHIFT_POINTER]

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                _func_name.append("_nchw")
                _full_macro_shift_pointer.append(_SHIFT_POINTER_NCHW)
                _full_macro_index_c = _FULL_MACRO_INDEX_C_NCHW
                _full_macro_index_h = _FULL_MACRO_INDEX_H_NCHW
                _full_macro_index_w = _FULL_MACRO_INDEX_W_NCHW
                _dimension_index_code = _DIMENSION_INDEX_CODE_NCHW
            case TensorFormat.NHWC:
                _func_name.append("_nhwc")
                _full_macro_shift_pointer.append(_SHIFT_POINTER_NHWC)
                _full_macro_index_h = _FULL_MACRO_INDEX_H_NHWC
                _full_macro_index_w = _FULL_MACRO_INDEX_W_NHWC
                _full_macro_index_c = _FULL_MACRO_INDEX_C_NHWC
                _dimension_index_code = _DIMENSION_INDEX_CODE_NHWC
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} is not an implemented format.")

        _func_name = "".join(_func_name)
        _full_macro_shift_pointer = "".join(_full_macro_shift_pointer)
        _dimension_index_code = _dimension_index_code.format(macro_index_c=_MACRO_INDEX_C,
                                                             macro_index_h=_MACRO_INDEX_H,
                                                             macro_index_w=_MACRO_INDEX_W)

        code = """
#define {macro_index_n}(idx, N, n) idx * n / N
{full_macro_index_c}
{full_macro_index_h}
{full_macro_index_w}

#define {macro_index_first_element}(index, dim_in, dim_out) (int) ((index * dim_in) / dim_out)
#define {macro_index_last_element}(index, dim_in, dim_out) (int) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)
{full_macro_shift_pointer}

#define TRUE  1
#define FALSE 0

__global__ void {func_name}({T}* dx, {T}* dy,
                            int n, int c, int h, int w,
                            int new_h, int new_w, int N,
                            int num_active_workers,
                            int num_ops_per_worker,
                            int num_ops_last_worker)
{{
    int idx, ops_remaining;
    int ni, ci, wi, hi, i, j;
    int h_start, h_end, w_start, w_end, elements_h, elements;
    unsigned short first_iteration;
    {T} delta;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    ops_remaining = ((idx + 1) == num_active_workers) ? num_ops_last_worker : num_ops_per_worker;
    idx *= num_ops_per_worker;

    ni = {macro_index_n}(idx, N, n);
    {dimension_index_code}
    first_iteration = TRUE;

    for(ni = ni;
        (ni < n) && (ops_remaining > 0);
        ni++)
    {{
        for(ci = (first_iteration ? ci : 0);
            (ci < c) && (ops_remaining > 0);
            ci++)
        {{
            for(hi = (first_iteration ? hi : 0);
                (hi < h) && (ops_remaining > 0);
                hi++)
            {{
                h_start = {macro_index_first_element}(hi, new_h, h);
                h_end = {macro_index_last_element}(hi, new_h, h);
                elements_h = h_end - h_start;

                for(wi = (first_iteration ? wi : 0), first_iteration = FALSE;
                    (wi < w) && (ops_remaining > 0);
                    wi++, ops_remaining--)
                {{
                    w_start = {macro_index_first_element}(wi, new_w, w);
                    w_end = {macro_index_last_element}(wi, new_w, w);
                    elements = elements_h * (w_end - w_start);

                    delta = ({T}) (*({macro_desp_pointer}(dy, c, new_h, new_w, ni, ci, hi, wi)) / elements);
                    for(i = h_start; i < h_end; i++)
                        for(j = w_start; j < w_end; j++)
                            (*({macro_desp_pointer}(dx, c, h, w, ni, ci, i, j))) += delta;
                }}
            }}
        }}
    }}
}}
"""

        code = code.format(full_macro_index_c=_full_macro_index_c,
                           full_macro_index_h=_full_macro_index_h,
                           full_macro_index_w=_full_macro_index_w,
                           full_macro_shift_pointer=_full_macro_shift_pointer,
                           macro_index_n=_MACRO_INDEX_N,
                           macro_index_c=_MACRO_INDEX_C,
                           macro_index_h=_MACRO_INDEX_H,
                           macro_index_w=_MACRO_INDEX_W,
                           macro_index_first_element=_MACRO_INDEX_FIRST_ELEMENT,
                           macro_index_last_element=_MACRO_INDEX_LAST_ELEMENT,
                           macro_desp_pointer=_MACRO_SHIFT_POINTER,
                           dimension_index_code=_dimension_index_code,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module

    def initialize_pool_2d_gpu(self, prev_shape, x):
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        pooling_shape = self.model.encode_shape((self.co, self.ho, self.wo))
        y = gpuarray.empty((self.model.batch_size, *pooling_shape), self.model.dtype)
        self.y = TensorGPU(y, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)

        if self.pooling_not_needed:
            self.y = x
        else:
            n, c, h, w = self.model.decode_shape(x.shape)

            # NOTE: "num_elements" (or simply "N") is the number of elements to process. Usually it would be np.prod(x.shape),
            #   but in this case we are putting elements in the output instead of processing the input's elements.
            num_elements = np.prod((n, c, self.ho, self.wo), dtype=np.int32)

            total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

            # If num_elements < total_num_threads, only will work "num_elements" threads. In the other cases will work "total_num_threads" threads.
            num_active_workers = np.int32(min(total_num_threads, num_elements))
            num_ops_per_worker = np.int32((num_elements + num_active_workers - 1) / num_active_workers)
            num_ops_last_worker = np.int32(num_elements - (num_active_workers - 1) * num_ops_per_worker)

            # NOTE: Instead of a number, PyCuda's driver expects "numpy.number"
            self.cuda_fwd_func(x.ary, self.y.ary,
                               np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                               np.int32(self.ho), np.int32(self.wo), num_elements,
                               num_active_workers, num_ops_per_worker, num_ops_last_worker,
                               grid=self.grid, block=self.block,
                               stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        n, c, h, w = self.model.decode_shape(dy.shape)

        num_elements = np.prod((n, c, self.ho, self.wo), dtype=np.int32)

        total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        num_active_workers = np.int32(min(total_num_threads, num_elements))
        num_ops_per_worker = np.int32((num_elements + num_active_workers - 1) / num_active_workers)
        num_ops_last_worker = np.int32(num_elements - (num_active_workers - 1) * num_ops_per_worker)

        self.cuda_bwd_func(self.dx.ary, self.y.ary,
                           np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                           np.int32(self.ho), np.int32(self.wo), num_elements,
                           num_active_workers, num_ops_per_worker, num_ops_last_worker,
                           grid=self.grid, block=self.block,
                           stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
