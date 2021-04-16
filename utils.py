"""
Miscellaneous utilities that do not yet belong to a certain package.
"""


def _repeat_tuple(param, n=2):
    """
    Repeat scalar input until it becomes an n-tuple.

    This is useful for converting parameters such as ``kernel_size`` int to tuple.

    Be aware that this function doesn't exhaustively validate input arguments.

    Parameters
    ----------
    param : scalar or tuple
        The parameter to be checked [and repeated]

    n : int
        Desired length of output tuple. ``n >= 0``

    Returns
    -------
    n-tuple
        If ``param`` is scalar, returns an n-tuple ``(param, param, ..., param)``.
        If ``param`` is already an iterable of length n, returns ``param`` itself as a tuple

    Raises
    ------
    AssertionError
        When the input is neither a scalar nor an n-tuple.
    """
    try:
        len(param)
    except TypeError:  # param doesn't have length, is scalar
        return tuple([param for _ in range(n)])
    assert len(param) == n
    return tuple(param)


def conv1d_shape_infer(l_in, kernel_size, stride, dilation):
    """
    Infer the output length of a 1d convolution module.

    Assuming the ``(n, c, l)`` format:

    - The number of samples isn't considered because it shouldn't vary at all.
    - The number of channels isn't considered because it is user-defined and easy to infer.
    - Padding isn't considered because a padding layer before convolution can achieve an equivalent effect.
      In addition, ``F.padding`` allows for more versatile padding options than convolutional layers in ``torch.nn``,
      so shape inference involving padding is best handled beforehand.

    Parameters
    ----------
    l_in : int
        Input tensor length

    kernel_size : int
        Size of the convolving kernel

    stride : int
        Stride of the convolution

    dilation : int
        Spacing between kernel elements

    Returns
    -------
    h_out : int
        Output tensor height

    w_out : int
        Output tensor width
    """
    l_out = (l_in - dilation * (kernel_size - 1) - 1) // stride + 1
    return l_out


def conv2d_shape_infer(h_in, w_in, kernel_size, stride, dilation):
    """
    Infer the output height and width of a 2d convolution module.

    Assuming the ``(n, c, h, w)`` format:

    - The number of samples isn't considered because it shouldn't vary at all.
    - The number of channels isn't considered because it is user-defined and easy to infer.
    - Padding isn't considered because a padding layer before convolution can achieve an equivalent effect.
      In addition, ``F.padding`` allows for more versatile padding options than convolutional layers in ``torch.nn``,
      so shape inference involving padding is best handled beforehand.

    Parameters
    ----------
    h_in : int
        Input tensor height

    w_in : int
        Input tensor width

    kernel_size : (int, int)
        Size of the convolving kernel

    stride : (int, int)
        Stride of the convolution

    dilation : (int, int)
        Spacing between kernel elements

    Returns
    -------
    h_out : int
        Output tensor height

    w_out : int
        Output tensor width
    """
    kernel_size = _repeat_tuple(kernel_size, n=2)
    stride = _repeat_tuple(stride, n=2)
    dilation = _repeat_tuple(dilation, n=2)
    h_out = conv1d_shape_infer(h_in, kernel_size[0], stride[0], dilation[0])
    w_out = conv1d_shape_infer(w_in, kernel_size[1], stride[1], dilation[1])
    return h_out, w_out


def _repeat_align(*args):
    """
    Given a collection of scalars and sequences, pad the scalars to match the length of longest sequences.

    If all arguments are scalars, they will be returned unchanged.

    Parameters
    ----------
    args : scalar or sequence
        Input argument to be repeated

    Returns
    -------
    ret : list of {scalar or sequence}
        The same set of arguments, all converted to sequences or all remain as scalars.

    length : int
        Length of any output argument (they're all the same). ``None`` if all input arguments are scalars.

    Raises
    ------
    ValueError
        When two non-singleton (length > 1) input sequences have different lengths.

        The padding method cannot be well-defined in this case.
    """
    if any(hasattr(a, "__len__") for a in args):
        length_set = set(len(a) for a in args if hasattr(a, "__len__"))
        if len(length_set) > 1:  #
            raise ValueError(f"Cannot align non-singleton input sequences with different lengths: {length_set}")
        length = length_set.pop()
        return [a if hasattr(a, "__len__") else [a] * length for a in args], length
    else:  # all scalars
        return args, None
