import numpy as np
type npDT = np.int8 | np.float32 | np.float64
type npDT_4Dims[T] = np.ndarray[tuple[int, int, int, int], T]
type npDT_3Dims[T] = np.ndarray[tuple[int, int, int], T]
type npDT_2Dims[T] = np.ndarray[tuple[int, int], T]
type npDT_1Dims[T] = np.ndarray[tuple[int], T]

##############################
# ADAPTIVE AVG. POOLING NCHW #
##############################
def adaptive_avg_pooling_fwd_nchw_cython[T: npDT](x: npDT_4Dims[T], pooled_x: npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): data input.
        pooled_x (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """
    ...
# ---

def adaptive_avg_pooling_bwd_nchw_cython[T: npDT](dy: npDT_4Dims[T], dx: npDT_4Dims[T]) -> None:
    """
    Args:
        dy (npDT_4Dims): data input.
        dx (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """
    ...
# ---    

##############################
# ADAPTIVE AVG. POOLING NCHW #
##############################

def adaptive_avg_pooling_fwd_nhwc_cython[T: npDT](x: npDT_4Dims[T], pooled_x: npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): data input.
        pooled_x (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """
    ...
# ---

def adaptive_avg_pooling_bwd_nhwc_cython[T: npDT](dy: npDT_4Dims[T], dx: npDT_4Dims[T]) -> None:
    """
    Args:
        dy (npDT_4Dims): data input.
        dx (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """
    ...
# ---

#######
# ADD #
#######

def add_cython[T: npDT](x: npDT_2Dims[T], b: npDT_1Dims[T]) -> None:

    """
    Args:
        x (npDT_2Dims): A contiguous memory view of the data. Since all the operations are made inplace, it's also where the output it's stored.
        b (npDT_1Dims): A contiguous memory view of the bias.

    Returns:
        Nothing. The output is stored in \"x\".
    """
    ...
# ---

##########
# ARGMAX #
##########
def argmax_cython[T: npDT](x: npDT_2Dims[T], 
                           maxv: npDT_1Dims[T], 
                           amax: np.ndarray[tuple[int], np.int32], 
                           rng: np.ndarray[tuple[int], np.int32], 
                           axis:int = 0) -> tuple[npDT_1Dims[T:np.int32], npDT_1Dims[T:np.int32]]:

    """
    Args:
        x (npDT_2Dims): A view 2 dimensional inptu's ndarray.
        maxv (npDT_1Dims): A view to a ndarray of one of the npDT's types where the max values' will be stored.
        amax (np.ndarray[tuple[int], np.int32]): view to a ndarray of type np.int32 where the arg max values' will be stored.
        rng (np.ndarray[tuple[int], np.int32]): view to a ndarray of type np.int32 where some outputs will be stored.
        axis (int): The axis where the argmax will be performed. Can be 0 or 1. Default: 0.
    
    Returns:
        Explicit: tuple[np.ndarray, np.ndarray]: a tuple formed by: [T: npDT](amax, rng) if axis is 0, or [T: npDT](rng, amax) if not.

        Implicit: maxv.
    """
    ...
# ---

##############################
# ADAPTIVE AVG. POOLING NCHW #
##############################
def average_pool_2d_fwd_nchw_cython[T: npDT](x:npDT_4Dims[T], y:npDT_4Dims[T],
                                             kh: int, kw: int, ho: int, wo: int,
                                             vpadding: int, hpadding: int,
                                             vstride: int, hstride: int,
                                             vdilation: int, hdilation: int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        y (npDT_4Dims): The 4 dimensional output's ndarray. (the output's data is stored in this parameter).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.
    
    Returns:
        Nothing. Implictily the output is stored in "y".
    """
    ...
# ---

def average_pool_2d_bwd_nchw_cython[T: npDT](dy:npDT_4Dims[T],
                                             n: int, h: int, w: int, c: int,
                                             kh: int, kw: int, ho: int, wo: int,
                                             vpadding: int, hpadding: int,
                                             vstride: int, hstride: int,
                                             vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): The 4 dimensional input's ndarray.
        dx (npDT_4Dims): The 4 dimensional output's ndarray. (the output's data will be stored in this parameter). Note: All values in this parameter should be 0.
        n (int): The number of images (usually, the batch size).
        h (int): The images' height.
        w (int): The images' width.
        c (int): The images' number of channel's(e.g.: RGB = 3 channels).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.
    
    Returns:
        Nothing. The output will be stored in "dx".
    """
    ...
# ---

##############################
# ADAPTIVE AVG. POOLING NHWC #
##############################
def average_pool_2d_fwd_nhwc_cython[T: npDT](x:npDT_4Dims[T], y:npDT_4Dims[T],
                                             kh: int, kw: int, ho: int, wo: int,
                                             vpadding: int, hpadding: int,
                                             vstride: int, hstride: int,
                                             vdilation: int, hdilation: int) -> None:

    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        y (npDT_4Dims): The 4 dimensional output's ndarray.(the output's data is stored in this parameter).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.
    
    Returns:
        Nothing. Implictily, the output is stored in "y".
    """
    ...
# ---

def average_pool_2d_bwd_nhwc_cython[T: npDT](dy:npDT_4Dims[T],
                                             dx:npDT_4Dims[T],
                                             n: int, h: int, w: int, c: int,
                                             kh: int, kw: int, ho: int, wo: int,
                                             vpadding: int, hpadding: int,
                                             vstride: int, hstride: int,
                                             vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): The 4 dimensional input's ndarray.
        dx (npDT_4Dims): The 4 dimensional output's ndarray. (the output's data will be stored in this parameter). Note: All values in this parameter should be 0.
        n (int): The number of images (usually, the batch size).
        h (int): The images' height.
        w (int): The images' width.
        c (int): The images' number of channel's (e.g.: RGB = 3 channels).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.
    
    Returns:
        Nothing. The output will be stored in "dx".
    """
    ...
# ---

###################
# B. N. INFERENCE #
###################
def bn_inference_cython[T: npDT](x: npDT_2Dims[T:npDT],
                                 y: npDT_2Dims[T:npDT],
                                 running_mean: npDT_1Dims[T:npDT], 
                                 inv_std: npDT_1Dims[T:npDT], 
                                 gamma: npDT_1Dims[T:npDT], 
                                 beta: npDT_1Dims[T:npDT]) -> None:
    """
    Args:
        x (npDT_2Dims): The 2 dimensional input's ndarray.
        y (npDT_2Dims): The 2 dimensional outputs's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        inv_std (npDT_1Dims): The input's 1 dimensions thtat stores the inverse standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        Nothing. The output is stored in "y".
    """
    ...
# ---

def bn_inference_nchw_cython[T: npDT](x: npDT_4Dims[T:npDT],
                                      y: npDT_4Dims[T:npDT],
                                      running_mean: npDT_1Dims[T:npDT], 
                                      inv_std: npDT_1Dims[T:npDT], 
                                      gamma: npDT_1Dims[T:npDT], 
                                      beta: npDT_1Dims[T:npDT]) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        y (npDT_4Dims): The 4 dimensional output's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        inv_std (npDT_1Dims): The input's 1 dimensions thtat stores the inverse standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        Nothing. The output will be stored in \"y\".
    """
    ...
# ---

def bn_relu_inference_cython[T: npDT](x: npDT_2Dims[T:npDT],
                                      y: npDT_2Dims[T:npDT],
                                      running_mean: npDT_1Dims[T:npDT], 
                                      inv_std: npDT_1Dims[T:npDT], 
                                      gamma: npDT_1Dims[T:npDT], 
                                      beta: npDT_1Dims[T:npDT]) -> None:
    """
    Args:
        x (npDT_2Dims): The 2 dimensional input's ndarray.
        y (npDT_2Dims): The 2 dimensional output's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        inv_std (npDT_1Dims): The input's 1 dimensions thtat stores the inverse standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        Nothing. The output will be stored in \"y\".
    """
    ...
# ---

##################
# B. N. TRAINING #
##################

def bn_training_fwd_cython[T: npDT](x: npDT_2Dims[T:npDT],
                                    gamma: npDT_1Dims[T:npDT],
                                    beta: npDT_1Dims[T:npDT],
                                    running_mean: npDT_1Dims[T:npDT],
                                    running_var: npDT_1Dims[T:npDT],
                                    momentum: float,
                                    eps: float) -> tuple[npDT_2Dims[T:npDT], npDT_1Dims[T:npDT], npDT_2Dims[T:npDT]]:
    """
    Args:
        x (np.ndarray[npDT, ndim=2]): The 4 dimensional input's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        inv_std (npDT_1Dims): The input's 1 dimensions thtat stores the inverse standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        out: A tuple where:
            - y (np.ndarray): A 2 dimensional ndarray that stores the output.
            - std (np.ndarray): A 1 dimensional ndarray that stores the standard deviation.
            - xn (np.ndarray): A 2 dimensional ndarray that stores the input normalized.

    Note:
        It's never used.
    """
    ...
# ---


def bn_training_bwd_cython[T: npDT](dx: npDT_2Dims[T],
                                    dy: npDT_2Dims[T],
                                    xn: npDT_2Dims[T],
                                    std: npDT_1Dims[T],
                                    gamma: npDT_1Dims[T],
                                    dgamma: npDT_1Dims[T],
                                    dbeta: npDT_1Dims[T]) -> None:
    """
    Args:
        dx (npDT_2Dims): The 2 dimensional array that contains the gradient of the input forward's (that is the output).
        dy (npDT_2Dims): The 2 dimensional array that contains the gradient of the backward's input.
        xn (npDT_2Dims): The 2 dimensional array that contains the normalized input's value.
        std (npDT_1Dims): The 1 dimensions ndarray that stores the standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions thtat stores the gamma's values
        dgamma (npDT_1Dims): The input's 1 dimensions ndarray the gradient of the gamma's values
        dbeta (npDT_1Dims): The input's 1 dimensions ndarray the  gradient of the beta's values

    Returns:
        Nothing. The output will be stored in "dx".
    """
    ...
# ---

##################
# B. N. TRAINING #
##################
def depthwise_conv_nchw_cython[T: npDT](x: npDT_4Dims[T],
                                        k: npDT_3Dims[T],
                                        res: npDT_4Dims[T] ,
                                        ho: int, wo: int,
                                        vpadding: int, hpadding: int, 
                                        vstride: int, hstride: int, 
                                        vdilation: int, hdilation: int)-> npDT_4Dims[T]:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        k (npDT_3Dims): The 3dimensions ndarray that contains the kernel.
        res (npDT_4Dims): The 4 dimensional output's ndarray.
        ho: (int): Output's height value.
        wo: (int): Output's width value.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.

    Returns:
        Nothing. The value is stores in \"res\".
    """
    ...
    ...
# ---

def depthwise_conv_backward_nchw_cython[T: npDT](dy: npDT_4Dims[T],
                                                 x: npDT_4Dims[T],
                                                 k: npDT_3Dims[T],
                                                 dx: npDT_4Dims[T],
                                                 dw: npDT_3Dims[T],
                                                 vpadding: int, hpadding:int, 
                                                 vstride: int, hstride:int, 
                                                 vdilation: int, hdilation:int)-> None:
    """
    Args:
        dy (npDT_4Dims): The 4 dimensional array that contains the gradient of the backward's input.
        x (npDT_4Dims): The 4 dimensional array that contains the input forward's.
        k (npDT_3Dims): The 3 dimensional array that contains the kernel.
        dx npDT_4Dims: The 4 dimensional array that contains the input forward's gradient.
        dw npDT_3Dims: The 3 dimensional array that contains the kernel's gradient
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The outputs are stored in \"dx\" and \"dw\".
    """
    ...
# ---

def depthwise_conv_nhwc_cython[T: npDT](x: npDT_4Dims[T],
                                        k: npDT_3Dims[T],
                                        res: npDT_4Dims[T] ,
                                        ho: int, wo: int,
                                        vpadding: int, hpadding: int, 
                                        vstride: int, hstride: int, 
                                        vdilation: int, hdilation: int)-> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        k (npDT_3Dims): The 3dimensions ndarray that contains the kernel.
        res (npDT_4Dims): The 4 dimensional output's ndarray.
        ho: (int): Output's height value.
        wo: (int): Output's width value.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.

    Returns:
        Nothing. The value is stores in \"res\".
    """
    ...
    ...
# ---

def depthwise_conv_backward_nhwc_cython[T: npDT](dy: npDT_4Dims[T],
                                                 x: npDT_4Dims[T],
                                                 k: npDT_3Dims[T],
                                                 dx: npDT_4Dims[T],
                                                 dw: npDT_3Dims[T],
                                                 vpadding: int, hpadding:int, 
                                                 vstride: int, hstride:int, 
                                                 vdilation: int, hdilation:int)-> None:
    """
    Args:
        dy (npDT_4Dims): The 4 dimensional array that contains the gradient of the backward's input.
        x (npDT_4Dims): The 4 dimensional array that contains the input forward's.
        k (npDT_3Dims): The 3 dimensional array that contains the kernel.
        dx npDT_4Dims: The 4 dimensional array that contains the input forward's gradient.
        dw npDT_3Dims: The 3 dimensional array that contains the kernel's gradient
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The outputs are stored in \"dx\" and \"dw\".
    """
    ...
# ---

############
# ELTW SUM #
############
def eltw_sum_cython[T: npDT](x_acc: npDT_1Dims[T], x: npDT_1Dims[T]) -> None:
    """
    This function adds the values of "x_acc" and "x" and accumulate them in "x_acc".
    Args:
        x_acc (npDT_1Dims): The 1 dimensional where the accumulation will be stored.
        x (npDT_1Dims): The 1 dimensional array with the data to accumulate.
    Returns:
        Nothing. The output is stored in "x_acc".
    """
    ...
# ---

#########################
# IM2COL 1 CHANNEL NCHW #
#########################
def im2col_1ch_nchw_cython[T: npDT](x:npDT_4Dims[T],
                                    kh:int, kw:int, 
                                    vpadding:int, hpadding:int,
                                    vstride:int, hstride:int, 
                                    vdilation:int, hdilation:int) -> np.ndarray:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        out (npDT_2Dims): The 2 dimensional array where the image as columns is stored.
    """
    ...
# ---

def col2im_1ch_nchw_cython[T: npDT](cols:npDT_2Dims[T],
                                    n: int, h: int, w: int, c: int,
                                    kh: int, kw: int,
                                    vpadding: int, hpadding: int,
                                    vstride: int, hstride: int,
                                    vdilation: int, hdilation: int) -> np.ndarray:
    """
    Args:
        cols (npDT_2Dims): The 2 dimensional array (the image).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        out (npDT_4Dims): The 4 dimensional array where the output image is stored.
    """
    ...
# ---

###############
# IM2COL NCHW #
###############
def im2col_nchw_cython[T: npDT](x:npDT_4Dims[T],
                                    kh:int, kw:int, 
                                    vpadding:int, hpadding:int,
                                    vstride:int, hstride:int, 
                                    vdilation:int, hdilation:int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        cols (npDT_2Dims): The 2 dimensional array where the image as columns will be stored (it should be initialized as zeros).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in \"cols\".
    """
    ...
# ---

def col2im_nchw_cython[T: npDT](cols:npDT_2Dims[T],
                                x:npDT_4Dims[T],
                                n: int, h: int, w: int, c: int,
                                kh: int, kw: int,
                                vpadding: int, hpadding: int,
                                vstride: int, hstride: int,
                                vdilation: int, hdilation: int) -> None:
    """
    Args:
        cols (npDT_2Dims): The 2 dimensional array (the image as columns).
        x (npDT_4Dims): The 4 dimensional array wher the image will be stored (it should be initialized as zeros).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in \"x\".
    """
    ...
# ---

#########################
# IM2ROW 1 CHANNEL NHWC #
#########################
def im2row_1ch_nhwc_cython[T: npDT](x:npDT_4Dims[T],
                                    rows:npDT_2Dims[T],
                                    kh:int, kw:int, 
                                    vpadding:int, hpadding:int,
                                    vstride:int, hstride:int, 
                                    vdilation:int, hdilation:int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        rows (npDT_2Dims): The 2 dimensional array where the image will be stored as rows (it should be initalized with 0s).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in \"rows\".
    """
    ...
# ---

def row2im_1ch_nhwc_cython[T: npDT](rows:npDT_2Dims[T],
                                    x: npDT_4Dims[T],
                                    n: int, h: int, w: int, c: int,
                                    kh: int, kw: int,
                                    vpadding: int, hpadding: int,
                                    vstride: int, hstride: int,
                                    vdilation: int, hdilation: int) -> None:
    """
    Args:
        rows (npDT_2Dims): The 2 dimensional array (the image as rows).
        x (npDT_4Dims): The 4 dimensional array where the image will be stored (it should be initalized with 0s).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is sotred in \"x\".
    """
    ...
# ---

###############
# IM2COL NHWC #
###############
def im2row_nhwc_cython[T: npDT](x:npDT_4Dims[T],
                                rows:npDT_2Dims[T],
                                kh:int, kw:int, 
                                vpadding:int, hpadding:int,
                                vstride:int, hstride:int, 
                                vdilation:int, hdilation:int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        rows (npDT_2Dims): The 2 dimensional array where the image as columns is stored (it should be initalized with 0s).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is sotred in \"rows\".
    """
    ...
# ---

def row2im_nhwc_cython[T: npDT](cols:npDT_2Dims[T],
                                x: npDT_4Dims[T],
                                n: int, h: int, w: int, c: int,
                                kh: int, kw: int,
                                vpadding: int, hpadding: int,
                                vstride: int, hstride: int,
                                vdilation: int, hdilation: int) -> None:
    """
    Args:
        cols (npDT_2Dims): The 2 dimensional array (the image).
        x (npDT_4Dims): The 4 dimensional array where the image will be stored (it should be initalized with 0s).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in \"x\".
    """
    ...
# ---

#################
# MAX POOL NCHW #
#################

def max_pool_2d_fwd_nchw_cython[T: npDT](x: npDT_4Dims[T],
                                         y: npDT_4Dims[T],
                                         idx_max: np.ndarray[tuple[int, int, int, int], np.int32],
                                         kh: int, kw: int, ho: int, wo: int,
                                         vpadding: int, hpadding: int,
                                         vstride: int, hstride: int, 
                                         vdilation: int, hdilation: int, 
                                         minval: npDT) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        y (npDT_4Dims): 4-dimensinal array where the output data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth 
        wo (int): Output's width 
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
        minval (npDT): minum value the selected type can have.
    Returns:
        Nothing. The output is stored in "y" and "idx_max"
    """
    ...
# ---

def max_pool_2d_bwd_nchw_cython[T: npDT](dy: npDT_4Dims[T],
                                         idx_max: np.ndarray[tuple[int, int, int, int], np.int32],
                                         dx: npDT_4Dims[T],
                                         n: int, h: int, w: int, c: int,
                                         kh: int, kw: int, ho: int, wo: int,
                                         vpadding: int, hpadding: int,
                                         vstride: int, hstride: int,
                                         vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): 4-dimensinal array where the input data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        dx (npDT_4Dims): 4 dimensional ndarray where the gradient is stored.
        n (int): Number of samples.
        h (int): Sample's heigth.
        w (int): Sample's width.
        c (int): Sample's channels.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
    Returns:
        Nothing. The output is stored in "dx".
    """
    ...
# ---

#################
# MAX POOL NHWC #
#################

def max_pool_2d_fwd_nhwc_cython[T: npDT](x: npDT_4Dims[T],
                                         y: npDT_4Dims[T],
                                         idx_max: np.ndarray[tuple[int, int, int, int], np.int32],
                                         kh: int, kw: int, ho: int, wo: int,
                                         vpadding: int, hpadding: int,
                                         vstride: int, hstride: int, 
                                         vdilation: int, hdilation: int, 
                                         minval: npDT) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        y (npDT_4Dims): 4-dimensinal array where the output data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth 
        wo (int): Output's width 
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
        minval (npDT): minum value the selected type can have.
    Returns:
        Nothing. The output is stored in "y" and "idx_max".
    """
    ...
# ---

def max_pool_2d_bwd_nhwc_cython[T: npDT](dy: npDT_4Dims[T],
                                         idx_max: np.ndarray[tuple[int, int, int, int], np.int32],
                                         dx: npDT_4Dims[T],
                                         n: int, h: int, w: int, c: int,
                                         kh: int, kw: int, ho: int, wo: int,
                                         vpadding: int, hpadding: int,
                                         vstride: int, hstride: int,
                                         vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): 4-dimensinal array where the input data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        dx (npDT_4Dims): 4 dimensional ndarray where the gradient is stored.
        n (int): Number of samples.
        h (int): Sample's heigth.
        w (int): Sample's width.
        c (int): Sample's channels.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
    Returns:
        Nothing. The output is stored in "dx".
    """
    ...
# ---

###############
# Memory View #
###############
def memoryview_index(view: memoryview, sub: bytes) -> int:
    """
    Find lowest index where substring is found.

    Args:
        view (memoryview): Memory view.
        sub (bytes): \"substring\" to find in \"view\".
    Returns:
        out (int): Index of the first apareance of \"sub\" in \"view\".
    """
    ...
# ---

##################
# POINTWISE CONV #
##################

def pointwise_conv_cython[T: npDT](x: npDT_4Dims[T],  k: npDT_2Dims[T], out: npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        k (npDT_2Dims): 2-dimensinal array where the kernel is stored.
        out (npDT_4Dims): 4-dimensinal array where the output is stored.
    Returns:
        Nothing. The output is stored in \"out\".
    """
    ...
# ---


########
# RELU #
########
def relu_cython[T:npDT](x: npDT_1Dims[T], 
                        max: npDT_1Dims[T], 
                        mask: np.ndarray[tuple[int], np.int8]) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """
    ...
# ---

# NOTE: If cap = 6, then this is a Relu6.
def capped_relu_cython[T:npDT](x: npDT_1Dims[T], 
                               max: npDT_1Dims[T], 
                               mask: np.ndarray[tuple[int], np.int8],
                               cap: float) -> None:
    """
    ReLU function where the values above "cap"'s value are set as this value.

    Note: if cap is 6, this is a Relu6

    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
        cap (float): The ReLU's superior limit. Any value in x greater that this parameter will be set to this parameter in the ouput.
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """
    ...
# ---

def leaky_relu_cython[T:npDT](x: npDT_1Dims[T], 
                               max: npDT_1Dims[T], 
                               mask: npDT_1Dims[T],
                               negative_slope: float) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
        negative_slope (float): The negative value's multiplayer (if is 0, this function acts as a normal ReLU)
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """
    ...
# ---

#############
# TRANSPOSE #
#############

def transpose_0231_ikj_cython[T:npDT](original: npDT_3Dims[T:npDT],
                                      transposed: npDT_3Dims[T:npDT]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1
    
    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
    ...
# ---

def transpose_0231_ijk_cython[T:npDT](original: npDT_3Dims[T:npDT],
                                      transposed: npDT_3Dims[T:npDT]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
    ...
    ...
# ---

def transpose_0312_ijk_cython[T:npDT](original: npDT_3Dims[T:npDT],
                                      transposed: npDT_3Dims[T:npDT]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,3,1,2).
    This is equivalent to transpose a 3D matrix 0x1·2x3 to 0x3x1·2

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
    ...
    ...
# ---

def transpose_1023_jik_cython[T:npDT](original: npDT_3Dims[T:npDT],
                                      transposed: npDT_3Dims[T:npDT]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (1,0,2,3).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
    ...
    ...
# ---

def transpose_1023_ijk_cython[T:npDT](original: npDT_3Dims[T:npDT],
                                      transposed: npDT_3Dims[T:npDT]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
    ...
    ...
# ---

