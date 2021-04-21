import unittest
from padding import pad
from padding.utils import pad_width_format
import numpy as np
import torch
import pywt


def get_data():
    ndim = np.random.randint(1, 5)
    shape = np.random.randint(1, 3, size=ndim)

    ndim_padded = np.random.randint(0, ndim)
    pad_ti = np.random.randint(0, 30, size=ndim_padded * 2)
    pad_np = np.vstack([np.zeros([ndim - ndim_padded, 2], dtype=int), pad_ti.reshape(-1, 2)[::-1]]).tolist()
    pad_ti = tuple(pad_ti.tolist())
    arr_np = np.random.rand(*shape) * 100
    arr_ti = torch.from_numpy(arr_np)
    return arr_np, arr_ti, pad_np, pad_ti


ti_np_modes = [
    ["circular", "wrap"],
    ["symmetric", "symmetric"],
    ["reflect", "reflect"],
]


class PaddingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_trials = 20

    def assertArrayEqual(self, a, b, tol=1e-8, msg=None):
        self.assertLess(np.abs(a - b).sum(), tol, msg=msg)

    def test_width_conversion(self):
        result = pad_width_format(((1, 2), (3, 4)), source="numpy", target="torch")
        expected = (3, 4, 1, 2)
        self.assertEqual(result, expected)

        result = pad_width_format((3, 4, 1, 2), source="torch", target="numpy", ndim=2)
        expected = ((1, 2), (3, 4))
        self.assertEqual(result, expected)

        result = pad_width_format((3, 4, 1, 2), source="torch", target="numpy", ndim=3)
        expected = ((0, 0), (1, 2), (3, 4))
        self.assertEqual(result, expected)

    def test_const(self):
        for i in range(self.n_trials):
            arr_np, arr_ti, pad_np, pad_ti = get_data()
            res_np = np.pad(arr_np, pad_np, mode="constant", constant_values=3)
            res_ti = pad(arr_ti, pad_ti, mode="constant", value=3)
            self.assertLess(np.abs(res_ti.numpy() - res_np).sum(), 1e-8)

    def test_keyword_modes(self):
        for kw_ti, kw_np in ti_np_modes:
            for i in range(self.n_trials):
                arr_np, arr_ti, pad_np, pad_ti = get_data()
                res_np = np.pad(arr_np, pad_np, mode=kw_np)
                res_ti = pad(arr_ti, pad_ti, mode=kw_ti).numpy()
                with self.subTest(mode=kw_ti, old_shape=arr_np.shape, new_shape=res_np.shape, padding=pad_np, res_np=res_np, res_ti=res_ti):
                    self.assertLess(np.abs(res_ti - res_np).sum(), 1e-8)

    @unittest.skip
    def test_circular(self):
        a = np.array([1, 3])
        b = torch.from_numpy(a)
        padding = [13, 12]
        res_np = np.pad(a, padding, mode="wrap")
        res_ti = pad(b, padding, mode="circular").numpy()
        print(a)
        print(res_np)
        print(res_ti)

    @unittest.skip
    def test_circular_2(self):
        a = np.array([[[1, 2], [3, 4]]])
        np_padding = [[0, 0], [4, 7], [1, 9]]
        ti_padding = [1, 9, 4, 7]
        res_np = np.pad(a, pad_width=np_padding, mode="wrap")
        res_ti = pad(torch.from_numpy(a), pad=tuple(ti_padding), mode="circular").numpy()
        res_wt = pywt.pad(a, pad_widths=np_padding, mode="periodic")
        print(res_np.shape)
        print(res_np)
        print(res_ti)
        print(res_wt)
        self.assertLess(np.abs(res_ti - res_np).sum(), 1e-8)

    @unittest.skip
    def test_circular_3(self):
        # Warning: this is a known bug in np.pad
        # notice how there's a repeated 1 2 1 2 in the middle of numpy's result
        a = np.array([0, 1, 2])
        padding = (2, 9)
        res_np = np.pad(a, padding, mode="wrap")
        res_ti = pad(torch.from_numpy(a), padding, mode="circular").numpy()
        print(f"numpy result: {res_np}")
        print(f"torchimage result: {res_ti}")
        # numpy result: [1 2 0 1 2 0 1 2 1 2 0 1 2 0]
        # torchimage result: [1 2 0 1 2 0 1 2 0 1 2 0 1 2]

    @unittest.expectedFailure
    def test_symmetric(self):
        data_np = np.arange(4).reshape([2, 2])
        data_torch = torch.from_numpy(data_np)

        # this is a known bug for numpy pad symmetric
        pad_torch = (1, 8)
        pad_np = pad_width_format(pad_torch, source="torch", target="numpy", ndim=data_torch.ndim)
        res_np = np.pad(data_np, pad_np, mode="symmetric")
        res_torch = pad(data_torch, pad_torch, mode="symmetric").numpy()
        print(res_np)
        print(res_torch)
        self.assertArrayEqual(res_np, res_torch)


if __name__ == '__main__':
    unittest.main()
