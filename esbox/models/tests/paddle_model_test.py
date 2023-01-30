import unittest

import paddle
import paddle.nn as nn
import numpy as np
from esbox.models import PaddleModel


class TestModel(PaddleModel):
    def __init__(self, input_dim=17, output_dim=1):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

        self.initialization()

    def forward(self, input):
        out = self.fc1(input)
        out = self.fc2(out)
        return out


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.target_model = TestModel()

    def test_get_weights(self):
        params = self.model.get_weights()
        expected_params = list(self.model.parameters())
        self.assertEqual(len(params), len(expected_params))
        for i, key in enumerate(params):
            self.assertLess((params[key].sum().item() - expected_params[i].sum().item()), 1e-5)

    def test_set_weights(self):
        params = self.model.get_weights()
        self.target_model.set_weights(params)

        for i, j in zip(params.values(), self.target_model.get_weights().values()):
            self.assertLessEqual(abs(i.sum().item() - j.sum().item()), 1e-3)

    def test_get_flat_weights(self):
        flat_weights = self.model.get_flat_weights()
        expected_flat_weights = list(self.model.get_weights().values())
        expected_flat_weights = [element for lis in expected_flat_weights for element in lis.flatten()]
        self.assertTrue((flat_weights == expected_flat_weights).all())

    def test_set_flat_weights(self):
        flat_weights = self.model.get_flat_weights()
        self.target_model.set_flat_weights(flat_weights)

        for i, j in zip(flat_weights, self.target_model.get_flat_weights()):
            self.assertLessEqual(abs(i.sum().item() - j.sum().item()), 1e-3)

    def test_set_weights_between_different_models(self):
        model1 = TestModel()
        model2 = TestModel()

        N = 10
        random_obs = np.random.randn(N, 17).astype(np.float32)
        for i in range(N):
            x = paddle.to_tensor(random_obs[i])
            model1_output = model1(x).numpy().sum()
            model2_output = model2(x).numpy().sum()
            self.assertNotEqual(model1_output, model2_output)

        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = np.random.randn(N, 17).astype(np.float32)
        for i in range(N):
            x = paddle.to_tensor(random_obs[i])
            model1_output = model1(x).numpy().sum()
            model2_output = model2(x).numpy().sum()
            self.assertEqual(model1_output, model2_output)

    def test_set_weights_wrong_params_num(self):
        params = self.model.get_weights()
        with self.assertRaises(TypeError):
            self.model.set_weights(params[1:])

    def test_set_weights_wrong_params_shape(self):
        params = self.model.get_weights()
        params['fc1.weight'] = params['fc2.bias']
        with self.assertRaises(AssertionError):
            self.model.set_weights(params)

    def test_set_weights_with_modified_params(self):
        params = self.model.get_weights()
        params['fc1.weight'][0][0] = 100
        params['fc1.bias'][0] = 100
        params['fc2.weight'][0][0] = 100
        params['fc2.bias'][0] = 100
        self.model.set_weights(params)
        new_params = self.model.get_weights()
        for i, j in zip(params.values(), new_params.values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)


if __name__ == '__main__':
    unittest.main()
