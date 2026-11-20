"""
SeisNet-1D: Synthetic STEAD Demo Chunk Generator
Generates a deterministic, physically realistic 1,200-trace sample chunk:
- 800 Local Earthquakes:
    * 3-component physics: P-wave onset prominent on vertical (Z), S-wave shear packet
      prominent on horizontal (E, N).
    * Realistic frequency dispersion: P (6-14 Hz), S (2-6 Hz), coda scattering decay.
    * Epicentral distance lag (S - P time: 2-15s), variable SNR (5-35 dB), earthquake magnitudes.
- 400 Colored Ambient/Cultural Noise Traces:
    * 1/f pink/brown noise, microseismic hums, cultural tremor without coherent phase onsets.
- Output:
    * data/sample/chunk_demo.hdf5 (HDF5 datasets under /data/<trace_name>, shape 6000x3)
    * data/sample/chunk_demo.csv (Full STEAD-compliant metadata catalog)
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import h5py
import scipy.signal as signal


def generate_colored_noise(n_samples: int, exponent: float = 1.0, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generates 1/f^exponent colored noise using Fourier domain shaping.
    exponent = 0.0 -> White noise
    exponent = 1.0 -> Pink noise (ambient seismic background)
    exponent = 2.0 -> Brown/Red noise (low-frequency drift)
    """
    if rng is None:
        rng = np.random.default_rng()
    white = rng.normal(0.0, 1.0, size=n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    scale = 1.0 / (freqs ** (exponent / 2.0))
    scale[0] = 0.0  # Zero DC
    colored = np.fft.irfft(fft_vals * scale, n=n_samples)
    std = np.std(colored)
    if std > 1e-8:
        colored = colored / std
    return colored


def synthesize_earthquake(
    rng: np.random.Generator,
    fs: float = 100.0,
    n_samples: int = 6000
) -> Tuple[np.ndarray, int, int, float, float, float]:
    """
    Synthesizes a realistic 3-component local earthquake seismogram.
    
    Returns:
        waveform: Shape (6000, 3) [E, N, Z]
        p_arrival: integer sample index
        s_arrival: integer sample index
        snr_db: signal-to-noise ratio in dB
        magnitude: local magnitude Ml
        distance_km: epicentral distance in km
    """
    # 1. Kinematic timing
    # P arrival between 5.0s and 18.0s (samples 500 to 1800)
    p_arrival = int(rng.integers(600, 1800))
    # Distance between 12 km and 110 km
    distance_km = float(rng.uniform(12.0, 110.0))
    # S - P lag: delta_t ≈ distance / 8.0 (seconds) -> in samples @ 100Hz
    sp_lag_sec = distance_km / 8.0
    sp_lag_samples = int(np.clip(sp_lag_sec * fs, 200, 1500))
    s_arrival = p_arrival + sp_lag_samples

    magnitude = float(np.round(rng.uniform(1.2, 4.8), 2))
    snr_db = float(np.round(rng.uniform(8.0, 35.0), 1))
    signal_amp = 10.0 ** (snr_db / 20.0)

    time = np.arange(n_samples) / fs

    # 2. P-wave synthesis (Compressional, high-frequency, mostly Vertical Z)
    p_t = time - (p_arrival / fs)
    p_mask = p_t >= 0.0
    p_t_pos = np.maximum(0.0, p_t)
    f_p = rng.uniform(7.0, 14.0)  # P-wave dominant frequency 7-14 Hz
    decay_p = rng.uniform(1.5, 3.5)
    # Fast rise, exponential coda envelope
    p_envelope = np.where(p_mask, (p_t_pos ** 1.2) * np.exp(-decay_p * p_t_pos), 0.0)
    p_phase = 2.0 * np.pi * f_p * p_t + rng.uniform(0, 2 * np.pi)
    p_signal = p_envelope * np.sin(p_phase)
    # Normalize P amplitude
    if np.max(np.abs(p_signal)) > 0:
        p_signal = p_signal / np.max(np.abs(p_signal))

    # P-wave radiation projection: High Z (0.7-0.95), lower E & N (0.1-0.4)
    p_z_scale = rng.uniform(0.75, 0.95)
    p_e_scale = rng.uniform(0.15, 0.35) * rng.choice([-1, 1])
    p_n_scale = rng.uniform(0.15, 0.35) * rng.choice([-1, 1])

    # 3. S-wave synthesis (Shear, lower frequency, mostly Horizontal E & N)
    s_t = time - (s_arrival / fs)
    s_mask = s_t >= 0.0
    s_t_pos = np.maximum(0.0, s_t)
    f_s = rng.uniform(2.5, 5.5)   # S-wave dominant frequency 2.5-5.5 Hz
    decay_s = rng.uniform(0.4, 1.2)
    s_envelope = np.where(s_mask, (s_t_pos ** 1.5) * np.exp(-decay_s * s_t_pos), 0.0)
    s_phase = 2.0 * np.pi * f_s * s_t + rng.uniform(0, 2 * np.pi)
    s_signal = s_envelope * np.sin(s_phase)
    if np.max(np.abs(s_signal)) > 0:
        s_signal = s_signal / np.max(np.abs(s_signal))

    # S-wave amplitude is typically 2.0 - 4.5x P-wave
    s_p_ratio = rng.uniform(2.2, 4.5)
    # Polarization angle in horizontal plane
    theta = rng.uniform(0, 2 * np.pi)
    s_e_scale = s_p_ratio * np.cos(theta)
    s_n_scale = s_p_ratio * np.sin(theta)
    s_z_scale = rng.uniform(0.1, 0.3) * rng.choice([-1, 1])

    # 4. Multi-path scattering / Coda envelope
    coda_end_sample = min(n_samples - 1, s_arrival + int(rng.uniform(1500, 3200)))

    # Composite clean seismic wavefield
    e_wave = (p_signal * p_e_scale + s_signal * s_e_scale) * signal_amp
    n_wave = (p_signal * p_n_scale + s_signal * s_n_scale) * signal_amp
    z_wave = (p_signal * p_z_scale + s_signal * s_z_scale) * signal_amp

    # 5. Background ambient seismic noise (pink + microseism 0.2 Hz)
    noise_e = generate_colored_noise(n_samples, exponent=1.1, rng=rng)
    noise_n = generate_colored_noise(n_samples, exponent=1.1, rng=rng)
    noise_z = generate_colored_noise(n_samples, exponent=1.0, rng=rng)

    # Microseismic swell (~0.15 - 0.3 Hz)
    micro_t = 2.0 * np.pi * rng.uniform(0.15, 0.3) * time
    noise_e += 0.4 * np.sin(micro_t + rng.uniform(0, 6.28))
    noise_n += 0.4 * np.sin(micro_t + rng.uniform(0, 6.28))
    noise_z += 0.3 * np.sin(micro_t + rng.uniform(0, 6.28))

    raw_e = e_wave + noise_e
    raw_n = n_wave + noise_n
    raw_z = z_wave + noise_z

    waveform = np.stack([raw_e, raw_n, raw_z], axis=-1).astype(np.float32) # (6000, 3)
    return waveform, p_arrival, s_arrival, snr_db, magnitude, distance_km


def synthesize_noise(
    rng: np.random.Generator,
    fs: float = 100.0,
    n_samples: int = 6000
) -> np.ndarray:
    """
    Synthesizes a realistic 3-component ambient/cultural seismic noise trace.
    Includes pink background noise, industrial cultural harmonics (e.g. 12 Hz / 24 Hz),
    and occasional random non-earthquake transient spikes.
    """
    time = np.arange(n_samples) / fs

    noise_e = generate_colored_noise(n_samples, exponent=rng.uniform(0.8, 1.4), rng=rng)
    noise_n = generate_colored_noise(n_samples, exponent=rng.uniform(0.8, 1.4), rng=rng)
    noise_z = generate_colored_noise(n_samples, exponent=rng.uniform(0.8, 1.4), rng=rng)

    # Cultural hum / machinery resonance (e.g. pumps, turbines, traffic)
    hum_freq = rng.uniform(8.0, 22.0)
    hum_amp = rng.uniform(0.1, 0.6)
    noise_e += hum_amp * np.sin(2.0 * np.pi * hum_freq * time + rng.uniform(0, 6.28))
    noise_n += hum_amp * np.sin(2.0 * np.pi * hum_freq * time + rng.uniform(0, 6.28))
    noise_z += hum_amp * np.sin(2.0 * np.pi * hum_freq * time + rng.uniform(0, 6.28))

    # Occasional cultural glitch/spike (footsteps, vehicle bump)
    if rng.random() > 0.5:
        spike_loc = rng.integers(500, n_samples - 500)
        spike_width = rng.integers(10, 40)
        spike_env = np.exp(-0.5 * ((np.arange(n_samples) - spike_loc) / spike_width) ** 2)
        noise_e += rng.uniform(-2.0, 2.0) * spike_env
        noise_n += rng.uniform(-2.0, 2.0) * spike_env
        noise_z += rng.uniform(-2.0, 2.0) * spike_env

    waveform = np.stack([noise_e, noise_n, noise_z], axis=-1).astype(np.float32) # (6000, 3)
    return waveform


def generate_demo_dataset(
    output_hdf5_path: str = "data/sample/chunk_demo.hdf5",
    output_csv_path: str = "data/sample/chunk_demo.csv",
    n_earthquakes: int = 800,
    n_noise: int = 400,
    seed: int = 42
) -> None:
    """
    Generates a deterministic STEAD sample chunk HDF5 + CSV dataset.
    """
    rng = np.random.default_rng(seed)
    total_traces = n_earthquakes + n_noise
    
    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"[Demo Generator] Synthesizing {total_traces} seismic traces ({n_earthquakes} EQs + {n_noise} Noise)...")
    print(f"[Demo Generator] HDF5 destination: {output_hdf5_path}")
    print(f"[Demo Generator] CSV destination:  {output_csv_path}")

    # Generate source IDs for grouping (simulating multiple stations recording same events)
    n_unique_sources = max(10, n_earthquakes // 5)
    source_ids = [f"EV_{i:04d}" for i in range(1, n_unique_sources + 1)]
    station_codes = [f"STA{i:02d}" for i in range(1, 51)]

    records = []

    with h5py.File(output_hdf5_path, "w") as h5_file:
        # Create STEAD /data group
        data_group = h5_file.create_group("data")

        # 1. Generate Earthquakes
        for i in range(n_earthquakes):
            trace_id = f"EQ_{i+1:06d}"
            assigned_source = source_ids[i % n_unique_sources]
            assigned_station = station_codes[rng.integers(0, len(station_codes))]

            waveform, p_arr, s_arr, snr, mag, dist_km = synthesize_earthquake(rng)

            # Store in HDF5: shape (6000, 3) under /data/<trace_name>
            dset = data_group.create_dataset(trace_id, data=waveform, dtype="float32", compression="gzip")
            dset.attrs["p_arrival_sample"] = p_arr
            dset.attrs["s_arrival_sample"] = s_arr
            dset.attrs["trace_category"] = "earthquake_local"

            records.append({
                "trace_name": trace_id,
                "trace_category": "earthquake_local",
                "source_id": assigned_source,
                "receiver_code": assigned_station,
                "network_code": "SN",
                "receiver_type": "EH",
                "p_arrival_sample": p_arr,
                "s_arrival_sample": s_arr,
                "p_status": "manual",
                "s_status": "manual",
                "p_weight": 1.0,
                "s_weight": 1.0,
                "snr_db": snr,
                "source_magnitude": mag,
                "source_distance_km": np.round(dist_km, 2),
                "coda_end_sample": min(5999, s_arr + 1800),
            })

        # 2. Generate Noise traces
        for j in range(n_noise):
            trace_id = f"NO_{j+1:06d}"
            assigned_station = station_codes[rng.integers(0, len(station_codes))]
            waveform = synthesize_noise(rng)

            dset = data_group.create_dataset(trace_id, data=waveform, dtype="float32", compression="gzip")
            dset.attrs["p_arrival_sample"] = np.nan
            dset.attrs["s_arrival_sample"] = np.nan
            dset.attrs["trace_category"] = "noise"

            records.append({
                "trace_name": trace_id,
                "trace_category": "noise",
                "source_id": f"NOISE_GRP_{j % 20:03d}",
                "receiver_code": assigned_station,
                "network_code": "SN",
                "receiver_type": "EH",
                "p_arrival_sample": np.nan,
                "s_arrival_sample": np.nan,
                "p_status": "",
                "s_status": "",
                "p_weight": np.nan,
                "s_weight": np.nan,
                "snr_db": float(np.round(rng.uniform(-5.0, 5.0), 1)),
                "source_magnitude": np.nan,
                "source_distance_km": np.nan,
                "coda_end_sample": np.nan,
            })

    # Save CSV catalog
    df = pd.DataFrame(records)
    # Shuffle records deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)

    print(f"[Demo Generator] Successfully generated {len(df)} traces in {output_hdf5_path} and {output_csv_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic STEAD sample chunk")
    parser.add_argument("--output-hdf5", type=str, default="data/sample/chunk_demo.hdf5", help="Path to output HDF5")
    parser.add_argument("--output-csv", type=str, default="data/sample/chunk_demo.csv", help="Path to output CSV")
    parser.add_argument("--earthquakes", type=int, default=800, help="Number of earthquake traces")
    parser.add_argument("--noise", type=int, default=400, help="Number of noise traces")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_demo_dataset(
        output_hdf5_path=args.output_hdf5,
        output_csv_path=args.output_csv,
        n_earthquakes=args.earthquakes,
        n_noise=args.noise,
        seed=args.seed
    )
