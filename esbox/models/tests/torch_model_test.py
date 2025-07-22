import unittest

import torch
import torch.nn as nn

from esbox.models import TorchModel


class TestModel(TorchModel):

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
        random_obs = torch.randn(N, 17)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model1_output = model1(x).item()
            model2_output = model2(x).item()
            self.assertNotEqual(model1_output, model2_output)

        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = torch.randn(N, 17)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model1_output = model1(x).item()
            model2_output = model2(x).item()
            self.assertEqual(model1_output, model2_output)

    def test_set_weights_wrong_params_num(self):
        params = self.model.get_weights()
        with self.assertRaises(TypeError):
            self.model.set_weights(params[1:])

    def test_set_weights_wrong_params_shape(self):
        params = self.model.get_weights()
        params['fc1.weight'] = params['fc2.bias']
        with self.assertRaises(RuntimeError):
            self.model.set_weights(params)


if __name__ == '__main__':
    unittest.main()
