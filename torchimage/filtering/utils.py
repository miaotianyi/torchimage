def _same_padding_pair(kernel_size):
    """
    Shape of padding required to keep input and output shape the same, with stride=dilation=1

    The same function can also be used to almost equally partition a fixed
    padding width before and after an axis, using the same convention:
    ``_same_padding_pair(total_pad_width + 1)``. This is because for any
    kernel size ``k``, the padding width required to keep input and output
    shape the same under stride=dilation=1 is ``k-1``.

    Parameters
    ----------
    kernel_size : int
        Size of convolution or pooling kernel

    Returns
    -------
    pad_beg : int
        Number of padding at the beginning of the axis.

    pad_end : int
        Number of padding at the end of the axis
    """
    pad_beg = kernel_size // 2
    if kernel_size % 2 == 1:  # odd kernel size, most common
        pad_end = pad_beg
    else:
        # this is somewhat arbitrary: padding less at the end
        # follow the same convention as scipy.ndimage
        pad_end = pad_beg - 1
    return pad_beg, pad_end
