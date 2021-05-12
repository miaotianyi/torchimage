import numpy as np


def ndim(a):
    """
    Return the number of dimensions of an array.

    This method differs from ``np.ndim`` in that the array
    conversion (if invoked) uses ``dtype=object``

    Parameters
    ----------
    a : array_like
        Input array. If it is not already an ndarray, a conversion
        without copy is attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`. Scalars are zero-dimensional.
    """
    try:
        return a.ndim
    except AttributeError:
        return np.asarray(a, dtype=object).ndim




class RaggedArray:
    """
    Multi-dimensional ragged array.

    Multi-dimensional ragged arrays are a superclass to hyper-rectangular
    multi-dimensional arrays (like numpy ndarray)

    """
    def __init__(self, array, is_scalar=lambda a: np.ndim(a) == 0):
        np.asarray
        assert callable(is_scalar)
        self.is_scalar = is_scalar

    def _broadcast(self, array):
        pass

    def expand(self, shape):
        pass


def broadcast_ragged(array):
    pass
    # return array, shape


def expand_ragged(array, source_shape, target_shape):
    pass

