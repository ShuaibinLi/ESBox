import unittest
import numpy as np
from esbox.problems.functions import Ackley


class AckleyFuncTest(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.problem = Ackley(dim=self.dim)

    def test_evaluate(self):
        x = [-0.68] * self.dim
        out = self.problem(x)[0]
        self.assertAlmostEqual(out, 4.60816867, 8, "Ackley-dim1 function.evaluate() test fail!")

    def test_pf(self):
        pf_x = self.problem.pf_x
        pf = self.problem.pf

        self.assertListEqual([0] * self.dim, pf_x)
        self.assertAlmostEqual(pf, self.problem(self.problem.pf_x)[0], 8)

    def test_bounds(self):
        x_low = self.problem.x_low
        x_up = self.problem.x_up

        out = self.problem(x_low)[0]
        self.assertAlmostEqual(out, 21.57031115, 8)

        with self.assertRaises(ValueError):
            out = self.problem(x_low - 1)
            out = self.problem(x_up + 1)

    def test_evaluate_batch(self):
        batch_size = 6
        batch_x = [[-0.68] * self.dim] * batch_size

        outs = self.problem(batch_x)
        np.testing.assert_almost_equal(list(outs), [4.60816867] * batch_size, 8)

    def test_diff_dims(self):
        for dim in [2, 10, 100]:
            self.dim = dim
            self.problem = Ackley(dim=self.dim)

            self.test_evaluate()
            self.test_pf()
            self.test_bounds()
            self.test_evaluate_batch()


if __name__ == "__main__":
    unittest.main()
