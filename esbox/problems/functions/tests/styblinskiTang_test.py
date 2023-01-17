import unittest
import numpy as np
from esbox.problems.functions import StyblinskiTang


class StyblinskiTangFuncTest(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.problem = StyblinskiTang(dim=self.dim, scale=False)

        # test x_s and corresponding y_s
        self.test_dims = [1, 2, 10, 100]
        self.test_x = [-0.68]
        self.test_y = {1: -5.292293120000001, 2: -10.584586240000002, 10: -52.922931200000015, 100: -529.229312}
        self.low_y = {1: 100.0, 2: 200.0, 10: 1000.0, 100: 10000.0}

    def test_evaluate(self):
        x = self.test_x * self.dim
        out = self.problem(x)[0]
        self.assertAlmostEqual(out, self.test_y[self.dim], 8, "Griewank-dim1 function.evaluate() test fail!")

    def test_pf(self):
        pf_x = self.problem.pf_x
        pf = self.problem.pf

        self.assertListEqual([-2.903534] * self.dim, pf_x)
        self.assertAlmostEqual(pf, self.problem(self.problem.pf_x)[0], 3)

    def test_bounds(self):
        x_low = self.problem.x_low
        x_up = self.problem.x_up

        out = self.problem(x_low)[0]
        self.assertAlmostEqual(out, self.low_y[self.dim], 8)

        with self.assertRaises(ValueError):
            out = self.problem(x_low - 1)
            out = self.problem(x_up + 1)

    def test_evaluate_batch(self):
        batch_size = 6
        batch_x = [self.test_x * self.dim] * batch_size

        outs = self.problem(batch_x)
        np.testing.assert_almost_equal(list(outs), [self.test_y[self.dim]] * batch_size, 8)

    def test_diff_dims(self):
        for dim in self.test_dims[1:]:
            self.dim = dim
            self.problem = StyblinskiTang(dim=self.dim)

            self.test_evaluate()
            self.test_pf()
            self.test_bounds()
            self.test_evaluate_batch()


if __name__ == "__main__":
    unittest.main()
