import unittest
from esbox.problems.problem_base import FunctionBase


class TestBaseFunction(FunctionBase):

    def __init__(self, dim=1, x_low=-5, x_up=5, scale=True, scaled_x_low=-1, scaled_x_up=1, a=5):
        FunctionBase.__init__(self,
                              dim=dim,
                              x_low=x_low,
                              x_up=x_up,
                              scale=scale,
                              scaled_x_low=scaled_x_low,
                              scaled_x_up=scaled_x_up)

        self.a = a

    def evaluate(self, x):
        return self.a + x


class FunctionBaseTest(unittest.TestCase):

    def setUp(self):
        self.function_scaled = TestBaseFunction()
        self.function_no_scaled = TestBaseFunction(scale=False)

        self.function_dim2_scaled = TestBaseFunction(dim=2)
        self.function_dim2_no_scaled = TestBaseFunction(dim=2, scale=False)

    def test_rescale(self):
        x = 1
        scaled_x = self.function_scaled.rescale(x)
        no_scaled_x = self.function_no_scaled.rescale(x)
        self.assertEqual(x, no_scaled_x)
        self.assertEqual(5, scaled_x)

    def test_evaluate(self):
        x = 1
        x_scaled = 5
        expect_out = self.function_scaled.a + x_scaled
        out1 = self.function_scaled(x)
        out2 = self.function_no_scaled(x_scaled)
        self.assertTrue((out1 == expect_out).all())
        self.assertTrue((out2 == expect_out).all())

    def test_dim2_evaluate1(self):
        x = [1] * 2
        x_scaled = [5] * 2
        out1 = self.function_dim2_scaled(x)
        out2 = self.function_dim2_no_scaled(x_scaled)
        self.assertTrue((out1 == out2).all())

    def test_dim2_evaluate2(self):
        x = [0.1, 0.5]
        x_scaled = [0.5, 2.5]
        out3 = self.function_dim2_scaled(x)
        out4 = self.function_dim2_no_scaled(x_scaled)
        self.assertTrue((out3 == out4).all())

    def test_dim2_evaluate2(self):
        x = [[0.1, 0.5], [0.2, 0.3]]
        x_scaled = [[0.5, 2.5], [1.0, 1.5]]
        out3 = self.function_dim2_scaled(x)
        out4 = self.function_dim2_no_scaled(x_scaled)
        self.assertTrue((out3 == out4).all())


if __name__ == "__main__":
    unittest.main()
