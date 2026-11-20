"""
SeisNet-1D: Live ObsPy SeedLink Streaming & Real-Time Inference Module
Implements:
1. SeismicRingBuffer: Thread-safe 3-channel (E, N, Z) 6,000-sample (60s @ 100 Hz) rolling buffer
2. ObsPySeedLinkStreamer: Subclass of EasySeedLinkClient running in background daemon thread
3. run_live_inference: Real-time inference with P/S phase picking, kinematics, and alert cooldown
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.preprocessing import preprocess_stream
    from src.model import SeisNet1D
    from src.evaluate import detect_peaks_1d
except ImportError:
    from preprocessing import preprocess_stream
    from model import SeisNet1D
    from evaluate import detect_peaks_1d


class SeismicRingBuffer:
    """
    Thread-safe 3-channel rolling circular buffer for seismic waveforms.
    Channels:
      0: East-West (E or 1)
      1: North-South (N or 2)
      2: Vertical (Z)
    Capacity: 6,000 samples (60 seconds at 100 Hz).
    """
    def __init__(self, max_samples: int = 6000, sampling_rate: float = 100.0):
        self.max_samples = int(max_samples)
        self.sampling_rate = float(sampling_rate)
        self.buffer = np.zeros((3, self.max_samples), dtype=np.float32)
        self.fill_samples = [0, 0, 0]
        self.last_timestamp: Optional[float] = None
        self._lock = threading.Lock()

    def append_channel(
        self,
        channel_idx: int,
        data: np.ndarray,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Appends a 1D chunk of samples to the specified channel (0: E, 1: N, 2: Z).
        Shifts older samples left as new samples arrive once capacity is reached.
        """
        if channel_idx not in (0, 1, 2):
            raise ValueError(f"Invalid channel_idx {channel_idx}. Must be 0 (E), 1 (N), or 2 (Z).")

        data_arr = np.asarray(data, dtype=np.float32).ravel()
        n = len(data_arr)
        if n == 0:
            return

        with self._lock:
            if n >= self.max_samples:
                # New chunk completely overwrites or exceeds buffer length
                self.buffer[channel_idx, :] = data_arr[-self.max_samples:]
                self.fill_samples[channel_idx] = self.max_samples
            else:
                # Roll existing data left and insert new data at the end
                self.buffer[channel_idx, :-n] = self.buffer[channel_idx, n:]
                self.buffer[channel_idx, -n:] = data_arr
                self.fill_samples[channel_idx] = min(self.max_samples, self.fill_samples[channel_idx] + n)

            if timestamp is not None:
                self.last_timestamp = float(timestamp)

    def get_snapshot(self) -> Tuple[np.ndarray, float, Optional[float]]:
        """
        Retrieves a thread-safe snapshot of the current 3-component buffer.

        Returns:
            Tuple of:
            - buffer_3x6000: np.ndarray of shape (3, 6000) in float32
            - fill_seconds: float representing shortest filled duration across channels (0.0 - 60.0s)
            - last_timestamp: float epoch seconds of last received packet or None
        """
        with self._lock:
            snapshot = self.buffer.copy()
            min_samples = min(self.fill_samples)
            fill_seconds = float(min_samples / self.sampling_rate)
            last_ts = self.last_timestamp
            return snapshot, fill_seconds, last_ts

    def clear(self) -> None:
        """Resets the ring buffer and fill counters to zero."""
        with self._lock:
            self.buffer.fill(0.0)
            self.fill_samples = [0, 0, 0]
            self.last_timestamp = None

    def is_ready(self, min_fill_sec: float = 10.0) -> bool:
        """Checks if all channels have accumulated at least min_fill_sec of data."""
        with self._lock:
            return (min(self.fill_samples) / self.sampling_rate) >= min_fill_sec


class ObsPySeedLinkStreamer(EasySeedLinkClient):
    """
    Background SeedLink waveform streaming client using ObsPy.
    Connects to any public SeedLink server, selects desired network/station/channel,
    resamples incoming data to 100 Hz if necessary, and feeds a SeismicRingBuffer.
    """
    def __init__(
        self,
        server_url: str = "geofon.gfz.de:18000",
        network: str = "IU",
        station: str = "GRFO",
        channel_selector: str = "BH?",
        ring_buffer: Optional[SeismicRingBuffer] = None,
        target_fs: float = 100.0,
        autoconnect: bool = False
    ):
        # Format server URL: add default port 18000 if omitted
        if ":" not in server_url:
            server_url = f"{server_url}:18000"

        super().__init__(server_url, autoconnect=autoconnect)
        self.network = str(network).strip().upper()
        self.station = str(station).strip().upper()
        self.channel_selector = str(channel_selector).strip().upper()
        self.ring_buffer = ring_buffer or SeismicRingBuffer(max_samples=6000, sampling_rate=target_fs)
        self.target_fs = float(target_fs)

        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._packets_received = 0
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def packets_received(self) -> int:
        with self._lock:
            return self._packets_received

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def on_data(self, trace) -> None:
        """
        ObsPy callback triggered when a MiniSEED data packet arrives.
        Maps channel to [E, N, Z] (indices 0, 1, 2) and feeds the ring buffer.
        """
        try:
            channel_name = trace.stats.channel.upper()
            last_char = channel_name[-1] if len(channel_name) > 0 else ""

            # Channel component mapping
            comp_map = {"E": 0, "1": 0, "N": 1, "2": 1, "Z": 2}
            if last_char not in comp_map:
                return
            ch_idx = comp_map[last_char]

            # Resample trace to 100 Hz if needed
            if abs(trace.stats.sampling_rate - self.target_fs) > 0.05:
                trace = trace.copy()
                trace.resample(self.target_fs)

            data = np.asarray(trace.data, dtype=np.float32)
            timestamp = trace.stats.endtime.timestamp

            self.ring_buffer.append_channel(ch_idx, data, timestamp=timestamp)

            with self._lock:
                self._packets_received += 1

        except Exception as e:
            with self._lock:
                self._last_error = f"Packet error: {e}"

    def on_seedlink_error(self) -> None:
        with self._lock:
            self._last_error = "SeedLink protocol error encountered."

    def on_terminate(self) -> None:
        self._is_running = False

    def _worker(self) -> None:
        """Internal daemon worker loop."""
        try:
            self.connect()
            self.select_stream(self.network, self.station, self.channel_selector)
            self.run()
        except Exception as e:
            if self._is_running:
                with self._lock:
                    self._last_error = f"Connection failed: {e}"
        finally:
            self._is_running = False

    def start_background(self) -> None:
        """Launches SeedLink streaming in a background daemon thread."""
        if self._is_running:
            return
        self._is_running = True
        with self._lock:
            self._last_error = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop_background(self, timeout: float = 3.0) -> None:
        """Terminates SeedLink connection and stops background thread cleanly."""
        self._is_running = False
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.terminate()
            self.close()
        except Exception:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)


def run_live_inference(
    snapshot_3x6000: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    config: Dict[str, Any],
    last_alert_time: float = 0.0,
    alert_cooldown_sec: float = 15.0
) -> Dict[str, Any]:
    """
    Performs real-time seismological inference on a 3-channel 60-second waveform:
    1. Preprocesses stream (detrending, 1-45 Hz bandpass filter, std normalization)
    2. Executes SeisNet1D forward pass
    3. Detects P and S wave arrival peaks
    4. Computes S - P lag and epicentral distance estimation
    5. Manages automated alert triggering with cooldown

    Returns:
        Dictionary containing preprocessed waveform, detection metrics, phase curves,
        phase pick arrival times, kinematic distance, and alert state.
    """
    fs = float(config.get("waveform", {}).get("sampling_rate", 100.0))
    f_min = float(config.get("preprocessing", {}).get("f_min", 1.0))
    f_max = float(config.get("preprocessing", {}).get("f_max", 45.0))
    order = int(config.get("preprocessing", {}).get("filter_order", 4))
    det_thresh = float(config.get("evaluation", {}).get("detection_threshold", 0.50))
    pick_thresh = float(config.get("evaluation", {}).get("pick_threshold", 0.30))
    peak_dist = int(config.get("evaluation", {}).get("peak_distance", 50))
    vel_factor = float(config.get("evaluation", {}).get("velocity_estimate_km_s", 8.0))

    # Preprocessing
    preprocessed = preprocess_stream(
        snapshot_3x6000,
        fs=fs,
        f_min=f_min,
        f_max=f_max,
        order=order,
        detrend=True,
        normalize=True
    )

    # PyTorch Forward Pass
    tensor_in = torch.from_numpy(preprocessed).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        det_logit, phase_logits = model(tensor_in)
        det_prob = float(torch.sigmoid(det_logit)[0].cpu().numpy())
        p_prob_curve = torch.sigmoid(phase_logits[0, 0, :]).cpu().numpy()
        s_prob_curve = torch.sigmoid(phase_logits[0, 1, :]).cpu().numpy()

    is_earthquake = bool(det_prob >= det_thresh)

    # Peak detection for P & S arrivals
    p_peak_samp, p_conf = detect_peaks_1d(p_prob_curve, threshold=pick_thresh, distance=peak_dist)
    s_peak_samp, s_conf = detect_peaks_1d(s_prob_curve, threshold=pick_thresh, distance=peak_dist)

    p_arrival_sec = (p_peak_samp / fs) if p_peak_samp is not None else None
    s_arrival_sec = (s_peak_samp / fs) if s_peak_samp is not None else None

    # Kinematics: S - P lag and distance
    s_p_lag_sec: Optional[float] = None
    est_distance_km: Optional[float] = None
    eew_lead_sec: Optional[float] = None

    if p_arrival_sec is not None and s_arrival_sec is not None and s_arrival_sec > p_arrival_sec:
        s_p_lag_sec = float(s_arrival_sec - p_arrival_sec)
        est_distance_km = float(s_p_lag_sec * vel_factor)
        eew_lead_sec = s_p_lag_sec

    # Alert cooldown logic
    now = time.time()
    alert_triggered = False
    new_alert_time = last_alert_time

    if is_earthquake and (now - last_alert_time >= alert_cooldown_sec):
        alert_triggered = True
        new_alert_time = now

    return {
        "raw_waveform": snapshot_3x6000,
        "preprocessed_waveform": preprocessed,
        "det_prob": det_prob,
        "is_earthquake": is_earthquake,
        "p_prob_curve": p_prob_curve,
        "s_prob_curve": s_prob_curve,
        "p_arrival_sec": p_arrival_sec,
        "s_arrival_sec": s_arrival_sec,
        "p_conf": p_conf,
        "s_conf": s_conf,
        "s_p_lag_sec": s_p_lag_sec,
        "est_distance_km": est_distance_km,
        "eew_lead_sec": eew_lead_sec,
        "alert_triggered": alert_triggered,
        "last_alert_time": new_alert_time
    }
