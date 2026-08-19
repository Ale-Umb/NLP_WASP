from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed in this environment")
class BilinearModelTest(unittest.TestCase):
    def test_forward_and_gradient_shapes(self) -> None:
        import torch

        from nhanes_semantic.model import LowRankBilinearRegressor

        bank = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)
        model = LowRankBilinearRegressor(bank, rank=4)
        z = torch.randn(11, 16)
        task_index = torch.randint(0, 5, (11,))
        prediction = model(z, task_index)
        self.assertEqual(tuple(prediction.shape), (11,))
        prediction.square().mean().backward()
        self.assertIsNotNone(model.left.grad)
        self.assertEqual(tuple(model.dense_operator().shape), (16, 16))


if __name__ == "__main__":
    unittest.main()

