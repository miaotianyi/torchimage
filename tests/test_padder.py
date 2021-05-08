import unittest

from itertools import product
import torch
import numpy as np
from torchimage.padding.generic_pad import GenericPadNd, _stat_padding_set


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-8
        self.n_trials = 50
    
    def assertArrayEqual(self, a, b, tol=None, msg=None):
        if tol is None:
            tol = self.tol
        self.assertLess(((a-b)**2).sum(), tol, msg=msg)
    
    def test_const_padding(self):
        mode = "constant"
        for _ in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 7, ndim).tolist()

            # fine for pad width to exceed original length
            pad_width_scalar = np.random.randint(0, 10)
            pad_width_pair = np.random.randint(0, 10, 2).tolist()
            pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

            constant_values_scalar = np.random.rand()
            constant_values_pair = np.random.rand(2).tolist()
            constant_values_list = np.random.rand(ndim, 2).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw, cv in product([pad_width_scalar, pad_width_pair, pad_width_list],
                                  [constant_values_scalar, constant_values_pair, constant_values_list]):
                with self.subTest(shape=shape, pad_width=pw, constant_values=cv):
                    expected = np.pad(arr_1, pad_width=pw, mode=mode, constant_values=cv)
                    actual = GenericPadNd(pad_width=pw, mode=mode, constant_values=cv)(arr_2).numpy()
                    self.assertArrayEqual(expected, actual)

    def test_stat_padding(self):
        for mode in _stat_padding_set:
            if mode == "median":  # numpy is known to behave differently when it comes to medians
                continue
            for _ in range(self.n_trials):
                ndim = np.random.randint(1, 6)
                shape = np.random.randint(1, 7, ndim)

                # fine for pad width to exceed original length
                pad_width_scalar = np.random.randint(0, 10)
                pad_width_pair = np.random.randint(0, 10, 2).tolist()
                pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

                stat_length_scalar = np.random.randint(1, min(shape) + 1)
                stat_length_pair = np.random.randint(1, min(shape) + 1, 2).tolist()
                stat_length_list = np.random.randint(1, np.tile((shape + 1).reshape(-1, 1), [1, 2])).tolist()

                arr_1 = np.random.rand(*shape)
                arr_2 = torch.from_numpy(arr_1)

                for pw, sl in product([pad_width_scalar, pad_width_pair, pad_width_list],
                                      [# None,  # (None, None), [(None, None)]*ndim,
                                       stat_length_scalar, stat_length_pair, stat_length_list]):
                    with self.subTest(shape=shape, mode=mode, stat_length=sl):
                        expected = np.pad(arr_1, pad_width=pw, mode=mode, stat_length=sl)
                        actual = GenericPadNd(pad_width=pw, mode=mode, stat_length=sl)(arr_2).numpy()
                        self.assertArrayEqual(expected, actual)

    def test_replicate(self):
        pass

if __name__ == '__main__':
    unittest.main()
