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
    interpretation. In 1 dimension, with ``[1, 2, 3, 4, 5]`` as input,
    a kernel with size 3 will result in
    ``[(1+2)/2, (1+2+3)/3, (2+3+4)/3, (3+4+5)/3, (4+5)/2]``.
    Note how the definition of arithmetic mean helps achieve
    equal output and input shapes by adjusting the divisor,
    without any padding at all.

    Parameters
    ----------
    in_size : int
        The original input size (doesn't include padding width)

    pad_width: tuple of int

    kernel_size : int

    stride : int

    Returns
    -------
    ret : tuple of int
        Has length out_size. ret[i] counts the number of original elements
        (instead of padded elements) in the receptive field.
    """
    pad_before, pad_after = pad_width
    # out_size = conv_1d(in_size=in_size+pad_before+pad_after, kernel_size=kernel_size, stride=stride, dilation=1)
    ret = []

    if kernel_size > pad_before + pad_after + in_size:
        return ret

    if pad_before > kernel_size:
        q, r = divmod(pad_before - kernel_size, stride)
        ret += [0] * (q + 1)
        pad_before = kernel_size - (stride - r)

    n_rising_monotone = (pad_before + in_size - kernel_size) // stride + 1
    n_rising = min(n_rising_monotone, pad_before // stride + 1)

    for i in range(n_rising):
        ret.append(kernel_size - pad_before)
        pad_before -= stride

    n_same = max(n_rising_monotone - n_rising, 0)
    ret += [kernel_size] * n_same
    pad_before -= stride * n_same

    while pad_before > 0:
        ret.append(in_size)
        pad_before -= stride

    in_size += pad_before

    if in_size < 0:
        n_zeros_after = (in_size + pad_after - kernel_size) // stride + 1
    else:
        for s in range(min(in_size, kernel_size), 0, -stride):
            in_size -= stride
            ret.append(s)

        pad_after += in_size
        n_zeros_after = (pad_after - kernel_size) // stride + 1
    ret += [0] * n_zeros_after
    return ret
