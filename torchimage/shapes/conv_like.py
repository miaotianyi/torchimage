from math import floor, ceil

from ..utils import NdSpec


def conv_1d(in_size, kernel_size, stride, dilation):
    """
    Infer the output size of convolution given input size
    and other parameters.

    This 1d function is used to process

    Parameters
    ----------
    in_size : int

    kernel_size : int

    stride : int

    dilation : int

    Returns
    -------
    out_size : int
    """
    out_size = (in_size - dilation * (kernel_size - 1) - 1) // stride + 1
    return out_size


def conv_nd(in_size, kernel_size, stride, dilation):
    out_size = NdSpec.apply(conv_1d, NdSpec(in_size), NdSpec(kernel_size), NdSpec(stride), NdSpec(dilation))
    return out_size


def n_original_elements_1d(in_size, pad_width, kernel_size, stride):
    """
    Number of original elements (rather than padded elements) used
    to obtain the corresponding output element in a convolution.

    For example, with ``in_size=3, pad_width=(2, 1), kernel_size=3, stride=1``,
    the input array ``[**|aaa|*]`` will give a result of
    [1, 2, 3, 2].

    This function is useful for average pooling, especially the
    count_include_pad keyword argument. Specifically, average filtering
    differs from other pre-defined convolution operators in its unique
    interpretation: the divisor of


    Parameters
    ----------
    in_size : int
        The original input size (doesn't include padding width)

    pad_width
    kernel_size
    stride

    Returns
    -------
    ret : tuple of int
        Has length out_size
    """
    pad_before, pad_after = pad_width
    out_size = conv_1d(in_size=in_size+pad_before+pad_after, kernel_size=kernel_size, stride=stride, dilation=1)
    ret = []

    #
    if pad_before > kernel_size:
        q, r = divmod(pad_before - kernel_size, stride)
        # zeros_before = (pad_before - kernel_size) // stride + 1
        # ret += [0] * zeros_before
        ret += [0] * (q + 1)
        # diff = zeros_before * stride + kernel_size - pad_before
        # diff = ((pad_before - kernel_size) // stride) * stride - (pad_before - kernel_size) + stride
        # diff = stride - r
        pad_before = kernel_size - (stride - r)

    print(f"{pad_before=}")
    for i in range((pad_before + in_size - kernel_size) // stride + 1):
        ret += [min(kernel_size + i * stride - pad_before, kernel_size)]
        in_size -= stride
        pad_before = 0
    # else:
    #     diff = stride - pad_before

    # ret += list(range(diff, kernel_size, stride)[:])

    # q, r = divmod(pad_before - kernel_size, stride)
    # zeros_before = max(q + 1, 0)
    # zeros_before = (pad_before - kernel_size) // stride + 1
    # ret += [0] * zeros_before
    # print(stride)
    # print(kernel_size)
    # print(zeros_before)
    # print(stride * zeros_before + kernel_size)


    # ret += list(range(r, kernel_size, stride))
    ret += [''] * (out_size - len(ret))
    return ret

