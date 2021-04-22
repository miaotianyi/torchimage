"""
General padding functionalities.

Padding extends the border of the original signal so the output
has a certain shape.

We design this padding package with these principles:

1. Maximal compatibility with PyTorch.
    We will implicitly use ``F.pad`` as much as we can.
    (For example, we do not implement zero and constant padding)
    We also adjust the names of the arguments to be compatible with PyTorch

2. Versatility
    We try to incorporate as many functionalities available in other packages
    as possible. Specifically, we try to reproduce the behavior of ``numpy.pad``,
    MatLab dwtmode, and PyWavelet signal extension modes.

Comparing with ``torch.nn.functional.pad``, we make the following modifcations:

1. Symmetric padding is added

2. Higher-dimension non-constant padding
    To the date of this release, PyTorch has not implemented reflect (ndim >= 3+2),
    replicate (ndim >= 4+2), and circular (ndim >= 4+2) for high-dimensional
    tensors (the +2 refers to the initial batch and channel dimensions).

    We achieve n-dimensional padding by sequentially applying 1-dimensional
    padding from the first axis (dim 0) to the last (dim n-1). In some cases
    (such as constant padding with different values before and after an axis)
    where different orders of applying 1d padding can change the final result,
    we follow numpy's convention going from the first to the last dimension.

3. Wider padding size
    Padding modes reflect and circular will cause PyTorch to fail when padding size
    is greater than the tensor's shape at a certain dimension.
    i.e. Padding value causes wrapping around more than once.

Comparing with ``numpy.pad``, we make the following modifcations:

1. Bug fixes for wider padding
    For modes such as symmetric and circular, numpy padding doesn't make
    proper repetition. For example, notice how numpy fails to repeat or
    flip the signal [0, 1] in on the right.
    >>> import numpy as np
    >>> np.pad([0, 1],(1, 10),mode="wrap")
    array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    >>> np.pad([0, 1],(1, 10),mode="symmetric")
    array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1])

    The same bug has been observed for antisymmetric as well.
    After reading the source code, I believe numpy's error comes from
    extending the padding for directions (before and after),
    which breaks the cycle of the original signal if before and after
    are relatively prime.

Padding Modes
-------------
``"empty"`` - pads with undefined values
    ``? ? ? ? | a b c d | ? ? ? ?`` where the empty values are from ``torch.empty``

``"constant"`` - pads with a constant value
    ``p p p p | a b c d | p p p p`` where ``p`` is supplied by ``constant_values`` argument.

    ``p1 p1 p1 p1 | a b c d | p2 p2 p2 p2`` where ``p1`` and ``p2`` are different constant values.

``"zeros"`` - pads with 0; a special case of constant padding
    ``0 0 0 0 | a b c d | 0 0 0 0``

``"symmetric"`` - extends signal by *mirroring* samples. Also known as half-sample symmetric
    ``d c b a | a b c d | d c b a``

``"antisymmetric"`` - extends signal by *mirroring* and *negating* samples. Also known as half-sample antisymmetric.
    ``-d -c -b -a | a b c d | -d -c -b -a``

``"reflect"`` - signal is extended by *reflecting* samples. This mode is also known as whole-sample symmetric
    ``d c b | a b c d | c b a``

``"replicate"`` - replicates the border pixels
    ``a a a a | a b c d | d d d d``

``"circular"`` - signal is treated as a periodic one
    ``a b c d | a b c d | a b c d``

``"periodize"`` - same as circular, except the last element is replicated when signal length is odd.
    ``a b c -> a b c c | a b c c | a b c c``
    Note that it first extends the signal to an even length prior to using periodic boundary conditions

``"odd_reflect"`` - extend the signal by a point-reflection across the edge element
    ``2a-d 2a-c 2a-b | a b c d | 2d-c 2d-b 2d-a | 2(2d-a)-(2d-b) 2(2d-a)-(2d-c) 2(2d-a)-d``
    Also known has whole-sample antisymmetric.

``"odd_symmetric"`` - extend the signal by a point-reflection across a hypothetical midpoint between the edge
    and the symmetrically reflected edge.
    ``2a-d 2a-c 2a-b a | a b c d | d 2d-c 2d-b 2d-a | 2d-a 2(2d-a)-(2d-b) 2(2d-a)-(2d-c) 2(2d-a)-d``

``"smooth"`` - extend the signal according to the first derivatives calculated on the edges
    (straight line extrapolation)
    ``a-4(b-a) a-3(b-a) a-2(b-a) a-(b-a) | a b c d | d+(d-c) d+2(d-c) d+3(d-c) d+4(d-c)

``"linear_ramp"`` - pads with the linear ramp between end value and the array border value.
    ``| a b c d | d+s d+2s ... d+n*s e`` where ``e`` is the end value, ``n`` is the number of
    elements between ``d`` and ``e``, and ``s = (e-d)/(n+1)``

    The end values are specified by argument ``end_values``

``"maximum"``, ``"mean"``, ``"median"``, ``"minimum"`` - pads with a statistical value
    of all or part of the vector along each axis.

    The length of the vector used for computing the statistical value is specified
    by argument ``stat_length``.

    Note that PyTorch's median behaves differently from numpy's median.
    When there is an even number of elements, ``torch.median`` returns the left element
    at the center, whereas ``numpy.median`` returns the arithmetic mean between
    the two elements at the center. We retain PyTorch's behavior here.

``<function>`` - extend the signal with a customized function
    The function should have signature ``pad_func(x, idx, dim)``, like other functions
    in :ref:`pad_1d.py`. You may find reading the source code from :ref:`pad_1d.py`
    and using the ``_modify_index`` function useful.

    To pass customized keyword arguments (like end_values, constant_values, and stat_length)
    to a self-defined padding function, especially when they depend on ``dim``,
    the user can define a separate function ``f: dim -> kwargs`` and call ``f``
    inside ``pad_func`` to get the keyword arguments from dimension.
    I use the same method for constant (when there are multiple padding values),
    linear ramp and stat padding.

    This is not the most elegant design but it works when it needs to,
    so please let me know if any improvement is urgently needed and I can fix it.


=============        =============   ===========     ==============================  =======
torchimage           PyWavelets      Matlab          numpy.pad                       Scipy
=============        =============   ===========     ==============================  =======
zeros                zero            zpd             constant, cval=0                N/A
constant             N/A             N/A             constant                        constant
replicate            constant        sp0             edge                            nearest
smooth               smooth          spd, sp1        N/A                             N/A
circular             periodic        ppd             wrap                            wrap
periodize            periodization   per             N/A                             N/A
symmetric            symmetric       sym, symh       symmetric                       reflect
reflect              reflect         symw            reflect                         mirror
antisymmetric        antisymmetric   asym, asymh     N/A                             N/A
odd_reflect          antireflect     asymw           reflect, reflect_type='odd'     N/A
odd_symmetric        N/A             N/A             symmetric, reflect_type='odd'   N/A
linear_ramp          N/A             N/A             linear_ramp                     N/A
maximum              N/A             N/A             maximum                         N/A
mean                 N/A             N/A             mean                            N/A
median               N/A             N/A             median                          N/A
minimum              N/A             N/A             minimum                         N/A
empty                N/A             N/A             empty                           N/A
<function>           N/A             N/A             <function>                      N/A
=============        =============   ===========     ==============================  =======
"""
from .tensor_pad import pad

__all__ = ["pad"]
