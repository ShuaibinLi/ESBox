import unittest
from esbox.problems.problem_base import ProblemBase


class TestBaseProblem(ProblemBase):
    def evaluate(self, x):
        return x


class ProblemBaseTest(unittest.TestCase):
    def setUp(self):
        self.problem = TestBaseProblem()

    def test_evaluate(self):
        x = 10
        call_ = self.problem(x)
        eval_ = self.problem.evaluate(x)
        self.assertEqual(x, call_, "ProblemBase.__call__() test fail!")
        self.assertEqual(x, eval_, "ProblemBase.evaluate() test fail!")


if __name__ == "__main__":
    unittest.main()
