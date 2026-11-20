"""
SeisNet-1D: Automated Unit & Pipeline Verification Tests
Tests preprocessing, model forward passes, loss computation, and peak detection.
"""

import unittest
import numpy as np
import torch

from src.preprocessing import preprocess_stream, butter_bandpass_filter, normalize_waveform
from src.dataset import make_gaussian_curve
from src.model import SeisNet1D, get_model_summary
from src.losses import MultiTaskLoss
from src.evaluate import detect_peaks_1d


class TestSeisNet1DPipeline(unittest.TestCase):

    def test_preprocessing_shape_and_dtype(self):
        # Test shape (6000, 3)
        raw = np.random.randn(6000, 3).astype(np.float32)
        proc = preprocess_stream(raw, fs=100.0, f_min=1.0, f_max=45.0)
        self.assertEqual(proc.shape, (3, 6000))
        self.assertEqual(proc.dtype, np.float32)

        # Test normalization: mean ~ 0, std ~ 1
        for ch in range(3):
            self.assertAlmostEqual(float(np.mean(proc[ch])), 0.0, places=2)
            self.assertAlmostEqual(float(np.std(proc[ch])), 1.0, places=2)

    def test_dead_channel_safeguard(self):
        # Dead channel (all zeros)
        raw = np.zeros((3, 6000), dtype=np.float32)
        raw[0] = np.random.randn(6000)  # Only channel 0 active
        proc = preprocess_stream(raw, fs=100.0)
        self.assertEqual(proc.shape, (3, 6000))
        self.assertTrue(np.all(proc[1] == 0.0))
        self.assertTrue(np.all(proc[2] == 0.0))
        self.assertFalse(np.isnan(proc).any())

    def test_gaussian_target_curve(self):
        arr_sample = 1250
        curve, mask = make_gaussian_curve(arr_sample, length=6000, sigma=20.0)
        self.assertEqual(mask, 1.0)
        self.assertEqual(len(curve), 6000)
        self.assertAlmostEqual(float(curve[arr_sample]), 1.0, places=4)
        # Check tails decay to near zero
        self.assertLess(float(curve[arr_sample - 100]), 1e-4)
        self.assertLess(float(curve[arr_sample + 100]), 1e-4)

        # Missing arrival
        curve_nan, mask_nan = make_gaussian_curve(np.nan, length=6000)
        self.assertEqual(mask_nan, 0.0)
        self.assertTrue(np.all(curve_nan == 0.0))

    def test_model_forward_pass(self):
        model = SeisNet1D(
            in_channels=3,
            encoder_channels=[16, 32, 64, 128],
            bottleneck_channels=128,
            bottleneck_dilations=[2, 4],
            phase_classes=2
        )
        x = torch.randn(2, 3, 6000)
        det_logits, phase_logits = model(x)

        self.assertEqual(det_logits.shape, (2,))
        self.assertEqual(phase_logits.shape, (2, 2, 6000))

        summary = get_model_summary(model)
        self.assertGreater(summary["trainable_parameters"], 0)

    def test_multitask_loss_computation(self):
        criterion = MultiTaskLoss()
        det_logits = torch.randn(4)
        phase_logits = torch.randn(4, 2, 6000)
        is_eq = torch.tensor([1.0, 1.0, 0.0, 0.0])
        p_target = torch.zeros(4, 6000)
        s_target = torch.zeros(4, 6000)
        p_mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
        s_mask = torch.tensor([1.0, 0.0, 1.0, 1.0]) # Second sample has missing S-pick

        loss, breakdown = criterion(
            detection_logits=det_logits,
            phase_logits=phase_logits,
            is_earthquake=is_eq,
            p_target=p_target,
            s_target=s_target,
            p_mask=p_mask,
            s_mask=s_mask
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIn("loss_total", breakdown)
        self.assertIn("loss_det", breakdown)
        self.assertIn("loss_p", breakdown)
        self.assertIn("loss_s", breakdown)

    def test_peak_detection(self):
        curve = np.zeros(6000, dtype=np.float32)
        true_peak = 1845
        t = np.arange(6000)
        curve = np.exp(-0.5 * ((t - true_peak) / 20.0) ** 2)

        peak_idx, conf = detect_peaks_1d(curve, threshold=0.30, distance=50)
        self.assertEqual(peak_idx, true_peak)
        self.assertAlmostEqual(conf, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
