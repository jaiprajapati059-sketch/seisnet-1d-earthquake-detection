"""
SeisNet-1D: PyTorch Dataset & DataLoader
Handles:
1. Dual Data Mode:
   - "demo": Loads auto-generated synthetic STEAD sample chunk (or generates it on the fly)
   - "stead": Loads authentic STEAD chunk (e.g. data/raw/chunk1.hdf5 & chunk1.csv)
2. Group-based train/val/test splitting by source_id to prevent event/source data leakage
3. Gaussian target probability curve generation (sigma = 20 samples / 0.2s)
4. Missing S-wave pick handling via loss masks
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.preprocessing import preprocess_stream
    from src.generate_demo_chunk import generate_demo_dataset
except ImportError:
    from preprocessing import preprocess_stream
    from generate_demo_chunk import generate_demo_dataset


def make_gaussian_curve(
    arrival_sample: Optional[float],
    length: int = 6000,
    sigma: float = 20.0
) -> Tuple[np.ndarray, float]:
    """
    Constructs a 1D continuous Gaussian probability target curve:
      y(t) = exp(-(t - t_arrival)^2 / (2 * sigma^2))
      
    Returns:
      curve: np.ndarray of shape (length,) with peak value 1.0
      mask: 1.0 if arrival is valid, 0.0 if NaN or missing
    """
    curve = np.zeros(length, dtype=np.float32)
    if arrival_sample is None or np.isnan(arrival_sample) or arrival_sample < 0:
        return curve, 0.0

    center = float(arrival_sample)
    t = np.arange(length, dtype=np.float32)
    # Truncate computation to +/- 4*sigma for efficiency and exact zeros outside
    window = 4.0 * sigma
    idx_min = max(0, int(center - window))
    idx_max = min(length, int(center + window + 1))

    if idx_min < idx_max:
        sub_t = t[idx_min:idx_max]
        curve[idx_min:idx_max] = np.exp(-0.5 * ((sub_t - center) / sigma) ** 2)

    return curve, 1.0


class SteadDataset(Dataset):
    """
    PyTorch Dataset for 3-component seismic waveforms from STEAD HDF5 and CSV catalogs.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        hdf5_path: str,
        sigma: float = 20.0,
        fs: float = 100.0,
        f_min: float = 1.0,
        f_max: float = 45.0,
        order: int = 4,
        detrend: bool = True,
        normalize: bool = True,
        cache_waveforms: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.sigma = sigma
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.order = order
        self.detrend = detrend
        self.normalize = normalize
        self.cache_waveforms = cache_waveforms

        self._h5_file = None
        self._cache = {} if cache_waveforms else None

    def _open_hdf5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        trace_name = str(row["trace_name"])
        category = str(row.get("trace_category", "noise"))
        is_earthquake = 1.0 if category == "earthquake_local" else 0.0

        # Retrieve raw waveform
        if self._cache is not None and trace_name in self._cache:
            waveform = self._cache[trace_name]
        else:
            self._open_hdf5()
            # Handle STEAD path conventions: /data/<trace_name> or /<trace_name>
            if "data" in self._h5_file and trace_name in self._h5_file["data"]:
                raw_data = self._h5_file["data"][trace_name][()]
            elif trace_name in self._h5_file:
                raw_data = self._h5_file[trace_name][()]
            else:
                raise KeyError(f"Trace '{trace_name}' not found in HDF5 file {self.hdf5_path}")

            # STEAD stores waveforms as (6000, 3) with [E, N, Z]
            raw_data = np.asarray(raw_data, dtype=np.float32)
            if raw_data.ndim == 2 and raw_data.shape[0] != 3 and raw_data.shape[1] == 3:
                raw_data = raw_data.T  # Transpose to (3, 6000)

            # Apply seismological signal conditioning
            waveform = preprocess_stream(
                raw_data,
                fs=self.fs,
                f_min=self.f_min,
                f_max=self.f_max,
                order=self.order,
                detrend=self.detrend,
                normalize=self.normalize
            )

            if self._cache is not None:
                self._cache[trace_name] = waveform

        # Build Gaussian target curves
        p_arr = row.get("p_arrival_sample", np.nan)
        s_arr = row.get("s_arrival_sample", np.nan)

        if is_earthquake == 1.0:
            p_curve, p_mask = make_gaussian_curve(p_arr, length=6000, sigma=self.sigma)
            s_curve, s_mask = make_gaussian_curve(s_arr, length=6000, sigma=self.sigma)
        else:
            # Noise traces: curves are all zeros; supervision drives false positive suppression
            p_curve = np.zeros(6000, dtype=np.float32)
            s_curve = np.zeros(6000, dtype=np.float32)
            p_mask = 1.0
            s_mask = 1.0

        p_arrival_int = int(p_arr) if not pd.isna(p_arr) else -1
        s_arrival_int = int(s_arr) if not pd.isna(s_arr) else -1
        snr_db = float(row.get("snr_db", 0.0)) if not pd.isna(row.get("snr_db", 0.0)) else 0.0
        distance_km = float(row.get("source_distance_km", 0.0)) if not pd.isna(row.get("source_distance_km", 0.0)) else 0.0

        sample = {
            "waveform": torch.from_numpy(waveform).float(),     # (3, 6000)
            "p_target": torch.from_numpy(p_curve).float(),       # (6000,)
            "s_target": torch.from_numpy(s_curve).float(),       # (6000,)
            "is_earthquake": torch.tensor(is_earthquake, dtype=torch.float32),
            "p_mask": torch.tensor(p_mask, dtype=torch.float32),
            "s_mask": torch.tensor(s_mask, dtype=torch.float32),
            "p_arrival_sample": p_arrival_int,
            "s_arrival_sample": s_arrival_int,
            "trace_name": trace_name,
            "snr_db": snr_db,
            "distance_km": distance_km
        }
        return sample

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


def split_catalog_by_group(
    df: pd.DataFrame,
    group_col: str = "source_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs pure-Python group-based splitting so that no seismic source/event appears
    in both training and evaluation partitions (ensuring zero data leakage).
    """
    rng = np.random.default_rng(seed)

    if group_col not in df.columns or df[group_col].isna().all():
        group_col = "receiver_code" if "receiver_code" in df.columns else None

    if group_col is None:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = int(len(df) * train_size)
        n_val = int(len(df) * val_size)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True)
        )

    # Extract unique group labels
    unique_groups = list(df[group_col].dropna().unique())
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train_g = max(1, int(n_groups * train_size))
    n_val_g = max(1, int(n_groups * val_size))

    train_groups = set(unique_groups[:n_train_g])
    val_groups = set(unique_groups[n_train_g:n_train_g + n_val_g])
    test_groups = set(unique_groups[n_train_g + n_val_g:])

    train_mask = df[group_col].isin(train_groups)
    val_mask = df[group_col].isin(val_groups)
    test_mask = df[group_col].isin(test_groups)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # Any remaining rows without assigned group (e.g. NaN) are partitioned into train
    unassigned_mask = ~(train_mask | val_mask | test_mask)
    if unassigned_mask.any():
        unassigned_df = df[unassigned_mask].reset_index(drop=True)
        train_df = pd.concat([train_df, unassigned_df], ignore_index=True)

    return train_df, val_df, test_df


def get_dataloaders(
    config: Dict[str, Any],
    mode: str = "demo",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Prepares train, validation, and test PyTorch DataLoaders.
    
    Args:
        config: Configuration dictionary loaded from config.yaml
        mode: "demo" or "stead"
        batch_size: Batch size override
        num_workers: DataLoader num_workers override (default 0 on Windows)
    """
    if batch_size is None:
        batch_size = config["training"]["batch_size"]
    if num_workers is None:
        num_workers = config["system"]["num_workers"]

    sigma = config["dataset"]["sigma"]
    fs = config["waveform"]["sampling_rate"]
    f_min = config["preprocessing"]["f_min"]
    f_max = config["preprocessing"]["f_max"]
    order = config["preprocessing"]["filter_order"]
    detrend = config["preprocessing"]["detrend"]
    group_col = config["dataset"].get("split_group_key", "source_id")
    seed = config["system"]["seed"]

    if mode == "demo":
        hdf5_path = config["paths"]["demo_hdf5"]
        csv_path = config["paths"]["demo_csv"]
        # If demo files do not exist, generate them automatically
        if not os.path.exists(hdf5_path) or not os.path.exists(csv_path):
            print(f"[Dataset] Demo files not found. Auto-generating sample chunk...")
            generate_demo_dataset(
                output_hdf5_path=hdf5_path,
                output_csv_path=csv_path,
                n_earthquakes=800,
                n_noise=400,
                seed=seed
            )
    elif mode == "stead":
        hdf5_path = config["paths"]["raw_hdf5"]
        csv_path = config["paths"]["raw_csv"]
        if not os.path.exists(hdf5_path) or not os.path.exists(csv_path):
            # Check if any chunk*.hdf5 exists in data/raw/
            raw_dir = config["paths"]["raw_dir"]
            candidate_h5 = list(Path(raw_dir).glob("chunk*.hdf5"))
            candidate_csv = list(Path(raw_dir).glob("chunk*.csv"))
            if candidate_h5 and candidate_csv:
                hdf5_path = str(candidate_h5[0])
                csv_path = str(candidate_csv[0])
                print(f"[Dataset] Found STEAD files in raw dir: {hdf5_path}, {csv_path}")
            else:
                raise FileNotFoundError(
                    f"STEAD chunk files not found at {hdf5_path} or in {raw_dir}.\n"
                    "Please download a STEAD chunk (e.g. chunk1.hdf5 & chunk1.csv) into data/raw/ "
                    "or run with '--mode demo' to use the realistic auto-generated sample chunk."
                )
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'demo' or 'stead'.")

    # Load CSV catalog
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[Dataset] Loaded catalog: {len(df)} traces from {csv_path}")

    # Split into train, val, test using group isolation
    train_df, val_df, test_df = split_catalog_by_group(
        df,
        group_col=group_col,
        train_size=config["dataset"]["train_split"],
        val_size=config["dataset"]["val_split"],
        test_size=config["dataset"]["test_split"],
        seed=seed
    )

    print(f"[Dataset] Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Instantiate datasets
    train_ds = SteadDataset(
        train_df, hdf5_path, sigma=sigma, fs=fs, f_min=f_min, f_max=f_max,
        order=order, detrend=detrend, normalize=True, cache_waveforms=True
    )
    val_ds = SteadDataset(
        val_df, hdf5_path, sigma=sigma, fs=fs, f_min=f_min, f_max=f_max,
        order=order, detrend=detrend, normalize=True, cache_waveforms=True
    )
    test_ds = SteadDataset(
        test_df, hdf5_path, sigma=sigma, fs=fs, f_min=f_min, f_max=f_max,
        order=order, detrend=detrend, normalize=True, cache_waveforms=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    meta = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "hdf5_path": hdf5_path,
        "csv_path": csv_path,
        "test_df": test_df
    }

    return train_loader, val_loader, test_loader, meta
