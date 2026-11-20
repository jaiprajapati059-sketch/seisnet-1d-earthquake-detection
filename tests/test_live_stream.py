"""
SeisNet-1D: Live Streaming & Ring Buffer Unit Tests
Tests SeismicRingBuffer wrap-around, ObsPy packet dispatching, and real-time inference
without requiring live network connectivity.
"""

import unittest
import time
import numpy as np
import torch
import obspy

from src.live_stream import SeismicRingBuffer, ObsPySeedLinkStreamer, run_live_inference
from src.model import SeisNet1D


class TestLiveStream(unittest.TestCase):

    def setUp(self):
        self.config = {
            "waveform": {"sampling_rate": 100},
            "preprocessing": {"f_min": 1.0, "f_max": 45.0, "filter_order": 4},
            "evaluation": {
                "detection_threshold": 0.50,
                "pick_threshold": 0.30,
                "peak_distance": 50,
                "velocity_estimate_km_s": 8.0,
            },
            "seedlink": {
                "alert_cooldown_sec": 15.0,
                "target_sampling_rate": 100.0
            }
        }

    def test_ring_buffer_init(self):
        rb = SeismicRingBuffer(max_samples=6000, sampling_rate=100.0)
        snapshot, fill_sec, last_ts = rb.get_snapshot()

        self.assertEqual(snapshot.shape, (3, 6000))
        self.assertEqual(snapshot.dtype, np.float32)
        self.assertEqual(fill_sec, 0.0)
        self.assertIsNone(last_ts)
        self.assertFalse(rb.is_ready(min_fill_sec=5.0))

    def test_ring_buffer_append_channel(self):
        rb = SeismicRingBuffer(max_samples=6000, sampling_rate=100.0)
        data_e = np.ones(250, dtype=np.float32) * 1.5
        data_n = np.ones(250, dtype=np.float32) * 2.5
        data_z = np.ones(250, dtype=np.float32) * 3.5

        rb.append_channel(0, data_e, timestamp=1600000000.0)
        rb.append_channel(1, data_n, timestamp=1600000000.1)
        rb.append_channel(2, data_z, timestamp=1600000000.2)

        snapshot, fill_sec, last_ts = rb.get_snapshot()
        self.assertEqual(fill_sec, 2.5)  # 250 / 100.0
        self.assertAlmostEqual(last_ts, 1600000000.2, places=3)

        # Check values placed at the tail of the buffer
        np.testing.assert_array_equal(snapshot[0, -250:], data_e)
        np.testing.assert_array_equal(snapshot[1, -250:], data_n)
        np.testing.assert_array_equal(snapshot[2, -250:], data_z)
        # Check older prefix is zero
        self.assertTrue(np.all(snapshot[:, :-250] == 0.0))

    def test_ring_buffer_wrap_around(self):
        rb = SeismicRingBuffer(max_samples=6000, sampling_rate=100.0)

        # 1. Append 4,000 samples of 10.0
        first_chunk = np.full(4000, 10.0, dtype=np.float32)
        rb.append_channel(0, first_chunk)

        # 2. Append 4,000 samples of 20.0 (total 8,000 samples)
        second_chunk = np.full(4000, 20.0, dtype=np.float32)
        rb.append_channel(0, second_chunk)

        # For channels 1 and 2, fill 6,000 samples
        rb.append_channel(1, np.ones(6000, dtype=np.float32))
        rb.append_channel(2, np.ones(6000, dtype=np.float32))

        snapshot, fill_sec, _ = rb.get_snapshot()
        self.assertEqual(snapshot.shape, (3, 6000))
        self.assertEqual(fill_sec, 60.0)
        self.assertTrue(rb.is_ready(60.0))

        # Channel 0 must retain exactly the last 6,000 samples:
        # First 2,000 should be 10.0 and subsequent 4,000 should be 20.0
        np.testing.assert_array_equal(snapshot[0, :2000], np.full(2000, 10.0, dtype=np.float32))
        np.testing.assert_array_equal(snapshot[0, 2000:], np.full(4000, 20.0, dtype=np.float32))

        # 3. Oversized chunk: append 7,000 samples
        large_chunk = np.arange(7000, dtype=np.float32)
        rb.append_channel(0, large_chunk)
        snapshot, _, _ = rb.get_snapshot()
        # Should contain the last 6,000 of the 7,000 samples (from 1000 to 6999)
        np.testing.assert_array_equal(snapshot[0], np.arange(1000, 7000, dtype=np.float32))

    def test_ring_buffer_invalid_channel(self):
        rb = SeismicRingBuffer(max_samples=6000)
        with self.assertRaises(ValueError):
            rb.append_channel(3, np.ones(10))

    def test_ring_buffer_clear(self):
        rb = SeismicRingBuffer(max_samples=6000)
        rb.append_channel(0, np.ones(100))
        rb.clear()
        snapshot, fill_sec, last_ts = rb.get_snapshot()
        self.assertEqual(fill_sec, 0.0)
        self.assertIsNone(last_ts)
        self.assertTrue(np.all(snapshot == 0.0))

    def test_run_live_inference_shapes_and_keys(self):
        model = SeisNet1D(
            in_channels=3,
            encoder_channels=[16, 32, 64, 128],
            bottleneck_channels=128,
            bottleneck_dilations=[2, 4],
            phase_classes=2
        )
        device = torch.device("cpu")
        snapshot = np.random.randn(3, 6000).astype(np.float32)

        results = run_live_inference(
            snapshot_3x6000=snapshot,
            model=model,
            device=device,
            config=self.config,
            last_alert_time=0.0,
            alert_cooldown_sec=15.0
        )

        expected_keys = [
            "raw_waveform", "preprocessed_waveform", "det_prob", "is_earthquake",
            "p_prob_curve", "s_prob_curve", "p_arrival_sec", "s_arrival_sec",
            "p_conf", "s_conf", "s_p_lag_sec", "est_distance_km",
            "eew_lead_sec", "alert_triggered", "last_alert_time"
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"Missing key '{key}' in inference output")

        self.assertEqual(results["preprocessed_waveform"].shape, (3, 6000))
        self.assertEqual(results["p_prob_curve"].shape, (6000,))
        self.assertEqual(results["s_prob_curve"].shape, (6000,))
        self.assertIsInstance(results["det_prob"], float)
        self.assertGreaterEqual(results["det_prob"], 0.0)
        self.assertLessEqual(results["det_prob"], 1.0)
        self.assertIsInstance(results["is_earthquake"], bool)
        self.assertIsInstance(results["alert_triggered"], bool)

    def test_alert_cooldown_behavior(self):
        model = SeisNet1D(
            in_channels=3,
            encoder_channels=[16, 32, 64, 128],
            bottleneck_channels=128,
            bottleneck_dilations=[2, 4],
            phase_classes=2
        )
        device = torch.device("cpu")
        snapshot = np.random.randn(3, 6000).astype(np.float32)

        # Force detection threshold to 0.0 so earthquake is always detected
        cfg = dict(self.config)
        cfg["evaluation"]["detection_threshold"] = 0.0

        # Call 1: initial alert should trigger
        res1 = run_live_inference(snapshot, model, device, cfg, last_alert_time=0.0, alert_cooldown_sec=30.0)
        self.assertTrue(res1["alert_triggered"])
        alert_ts = res1["last_alert_time"]
        self.assertGreater(alert_ts, 0.0)

        # Call 2 immediately within cooldown: should NOT trigger
        res2 = run_live_inference(snapshot, model, device, cfg, last_alert_time=alert_ts, alert_cooldown_sec=30.0)
        self.assertFalse(res2["alert_triggered"])
        self.assertEqual(res2["last_alert_time"], alert_ts)

    def test_obspy_streamer_offline_packet_dispatch(self):
        rb = SeismicRingBuffer(max_samples=6000, sampling_rate=100.0)
        streamer = ObsPySeedLinkStreamer(
            server_url="geofon.gfz.de:18000",
            network="IU",
            station="GRFO",
            channel_selector="BH?",
            ring_buffer=rb,
            target_fs=100.0,
            autoconnect=False
        )

        self.assertFalse(streamer.is_running)
        self.assertEqual(streamer.packets_received, 0)
        self.assertIsNone(streamer.last_error)

        # Synthesize an ObsPy Trace with 50 Hz sampling rate for BHZ
        raw_50hz = np.ones(100, dtype=np.float32) * 5.0
        trace_z = obspy.Trace(
            data=raw_50hz,
            header={
                "network": "IU",
                "station": "GRFO",
                "channel": "BHZ",
                "sampling_rate": 50.0,
                "starttime": obspy.UTCDateTime(1600000000)
            }
        )

        # Dispatch via on_data callback
        streamer.on_data(trace_z)
        self.assertEqual(streamer.packets_received, 1)

        snapshot, fill_sec, last_ts = rb.get_snapshot()
        # 100 samples @ 50 Hz resampled to 100 Hz = 200 samples
        self.assertEqual(rb.fill_samples[2], 200)

        # Test BHE and BHN channels
        trace_e = obspy.Trace(
            data=np.ones(100, dtype=np.float32),
            header={"channel": "BHE", "sampling_rate": 100.0}
        )
        trace_n = obspy.Trace(
            data=np.ones(100, dtype=np.float32),
            header={"channel": "BHN", "sampling_rate": 100.0}
        )
        streamer.on_data(trace_e)
        streamer.on_data(trace_n)
        self.assertEqual(streamer.packets_received, 3)
        self.assertEqual(rb.fill_samples[0], 100)
        self.assertEqual(rb.fill_samples[1], 100)


if __name__ == "__main__":
    unittest.main()
