"""İndirme veya eğitim gerektirmeyen temel kontroller: python -m unittest -v."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

from predict import preprocess
from train import make_model


class ProjectTests(unittest.TestCase):
    def test_mlp_shapes(self):
        for hidden in ([256, 128], [128, 64]):
            model = make_model(hidden)
            self.assertEqual(model(torch.zeros(2, 1, 28, 28)).shape, (2, 10))
            self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model), 3)
            self.assertEqual(sum(isinstance(layer, nn.ReLU) for layer in model), 2)

    def test_preprocess(self):
        # Sentetik çizgi yalnızca birim testi içindir; ödev verisi değildir.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "line.png"
            image = Image.new("L", (80, 100), 230)
            ImageDraw.Draw(image).line((40, 20, 40, 80), fill=20, width=5)
            image.save(path)
            result, tensor = preprocess(path)
            self.assertEqual(result.size, (28, 28))
            self.assertEqual(tensor.shape, (1, 28, 28))
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertGreater(tensor.max().item(), 0)

    def test_blank_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            Image.new("L", (50, 50), 255).save(path)
            with self.assertRaises(ValueError):
                preprocess(path)


if __name__ == "__main__":
    unittest.main()
