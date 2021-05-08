import unittest
from torchimage.padding import pad
from torchimage.padding.utils import pad_width_format
import numpy as np
import torch
import pywt


def get_random_data():
    ndim = np.random.randint(1, 5)
    shape = np.random.randint(1, 7, size=ndim)

    # ndim_padded = np.random.randint(0, ndim)
    ndim_padded = ndim

    # use smaller padding size to avoid cycle bug in numpy
    padding = np.random.randint(0, 8, size=ndim_padded * 2)
    if False:  # small padding
        padding = np.clip(padding, 0, np.repeat(shape[ndim - ndim_padded:][::-1], 2))
    pad_ti = tuple(padding.tolist())
    pad_np = pad_width_format(pad_ti, source="torch", target="numpy", ndim=ndim)

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
        self.n_trials = 50

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
            arr_np, arr_ti, pad_np, pad_ti = get_random_data()
            res_np = np.pad(arr_np, pad_np, mode="constant", constant_values=3)
            res_ti = pad(arr_ti, pad_ti, mode="constant", constant_values=3)
            self.assertLess(np.abs(res_ti.numpy() - res_np).sum(), 1e-8)

    @unittest.skip
    def test_keyword_modes(self):
        for kw_ti, kw_np in ti_np_modes:
            for i in range(self.n_trials):
                arr_np, arr_ti, pad_np, pad_ti = get_random_data()
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
        res_ti = pad(torch.from_numpy(a), padding=tuple(ti_padding), mode="circular").numpy()
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

    @unittest.skip
    def test_symmetric(self):
        data_np = np.arange(2)
        data_torch = torch.from_numpy(data_np)

        # this is a known bug for numpy pad symmetric
        pad_torch = (1, 8)
        pad_np = pad_width_format(pad_torch, source="torch", target="numpy", ndim=data_torch.ndim)
        res_np = np.pad(data_np, pad_np, mode="symmetric")
        res_torch = pad(data_torch, pad_torch, mode="symmetric").numpy()
        # print(res_np)
        # print(res_torch)
        self.assertArrayEqual(res_np, res_torch)

    @unittest.skip
    def test_antisymmetric(self):
        for i in range(self.n_trials):
            print(i)
            arr_np, arr_ti, pad_np, pad_ti = get_random_data()
            print(f"{arr_np.shape=}, {pad_np=}")
            print("getting pywt result")
            try:
                res_pywt = pywt.pad(arr_np, pad_np, mode="antisymmetric")
            except ValueError:
                continue
            print("getting torch image result")
            res_ti = pad(arr_ti, pad_ti, mode="antisymmetric").numpy()
            with self.subTest(old_shape=arr_np.shape, new_shape=res_ti.shape, pad=pad_np):
                self.assertArrayEqual(res_ti, res_pywt)

    @unittest.skip
    def test_antisymmetric_2(self):
        # this is also doomed to fail
        padding = (1, 8)
        a = [1, 2]
        res_pywt = pywt.pad(a, padding, mode="antisymmetric")
        res_ti = pad(torch.tensor(a), padding, mode="antisymmetric").numpy()
        print(res_pywt)
        print(res_ti)
        self.assertArrayEqual(res_pywt, res_ti)
        # f = lambda **kwargs: kwargs
        # f(old_shape=(6, 1, 6), new_shape = (56, 14, 31), pad = ((24, 26), (11, 2), (24, 1))),
        # f(old_shape=(4, 4, 4), new_shape = (24, 39, 24), pad = ((3, 17), (28, 7), (6, 14))),
        # f(old_shape=(6,), new_shape = (34,), pad = ((26, 2),)),
        # f(old_shape=(5, 2, 5), new_shape = (36, 50, 56), pad = ((27, 4), (22, 26), (24, 27)))

    @unittest.skip
    def test_periodize(self):
        for i in range(self.n_trials):
            print(i)
            arr_np, arr_ti, pad_np, pad_ti = get_random_data()
            print(f"{arr_np.shape=}, {pad_np=}")
            print("getting pywt result")
            try:
                res_pywt = pywt.pad(arr_np, pad_np, mode="periodization")
            except ValueError:
                continue
            print("getting torch image result")
            res_ti = pad(arr_ti, pad_ti, mode="periodize").numpy()
            with self.subTest(old_shape=arr_np.shape, new_shape=res_ti.shape, pad=pad_np):
                self.assertArrayEqual(res_ti, res_pywt)

    @unittest.skip
    def test_periodize_2(self):
        # it's hard to test periodize now that circular is problematic
        padding = (0, 8)
        a = [1, 2, 3]
        res_pywt = pywt.pad(a, padding, mode="periodization")
        res_ti = pad(torch.tensor(a), padding, mode="periodize").numpy()
        print(res_pywt)
        print(res_ti)
        self.assertArrayEqual(res_pywt, res_ti)

    def test_constant(self):
        for i in range(self.n_trials):
            arr_np, arr_ti, pad_np, pad_ti = get_random_data()
            vals_ti = np.random.rand(arr_np.ndim * 2)
            vals_np = pad_width_format(vals_ti, source="torch", target="numpy", ndim=arr_np.ndim)
            res_np = np.pad(arr_np, pad_np, mode="constant", constant_values=vals_np)
            res_ti = pad(arr_ti, pad_ti, mode="constant", constant_values=vals_ti).numpy()
            with self.subTest(pad_ti=pad_ti):
                self.assertArrayEqual(res_np, res_ti)

    def test_linear_ramp(self):
        for i in range(self.n_trials):
            arr_np, arr_ti, pad_np, pad_ti = get_random_data()
            # end values
            vals_ti = np.random.rand(arr_np.ndim * 2)
            vals_np = pad_width_format(vals_ti, source="torch", target="numpy", ndim=arr_np.ndim)
            res_np = np.pad(arr_np, pad_np, mode="linear_ramp", end_values=vals_np)
            res_ti = pad(arr_ti, pad_ti, mode="linear_ramp", end_values=vals_ti).numpy()
            with self.subTest(pad_ti=pad_ti):
                self.assertArrayEqual(res_np, res_ti)

    @unittest.skip
    def test_stat(self):
        for mode in ("maximum", "mean", "median", "minimum"):
            for i in range(self.n_trials):
                arr_np, arr_ti, pad_np, pad_ti = get_random_data()
                # stat length
                vals_ti = np.random.randint(0, 3, arr_np.ndim * 2)
                vals_ti = np.clip(vals_ti, 1, None)
                # vals_ti = np.clip(vals_ti, 1, np.repeat(arr_np.shape[::-1], 2))
                vals_ti = tuple(vals_ti.tolist())
                vals_np = pad_width_format(vals_ti, source="torch", target="numpy", ndim=arr_np.ndim)
                res_np = np.pad(arr_np, pad_np, mode=mode, stat_length=vals_np)
                res_ti = pad(arr_ti, pad_ti, mode=mode, stat_length=vals_ti).numpy()
                with self.subTest(mode=mode, pad_ti=pad_ti, stat_length=vals_ti):
                    self.assertArrayEqual(res_np, res_ti)


if __name__ == '__main__':
    unittest.main()
