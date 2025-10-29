import ctypes

import pycuda.driver as drv  # type: ignore


# The below code will allocate the maximum used memory, which will be shared
# among all layers. This code saves having a memory allocation per layer.
ws_size = 1
ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
ws_ptr = ctypes.c_void_p(int(ws))


def checkConvolutionMemory(size):
    global ws_size
    global ws
    global ws_ptr
    # if a layer requires more memory than the allocated
    # we re-allocated that size
    if size.value > ws_size:
        ws_size = size.value
        assert not isinstance(ws, int), f"\"ws\" must not be an \"int\" here ({type(ws)=} || {ws=})."
        ws.free()
        ws = drv.mem_alloc(ws_size) if ws_size > 0 else 0
        ws_ptr = ctypes.c_void_p(int(ws))


def getConvolutionWorkspaceSize():
    return ws_size


def getConvolutionWorkspacePtr():
    return ws_ptr
