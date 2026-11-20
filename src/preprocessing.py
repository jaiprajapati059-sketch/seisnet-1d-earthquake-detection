"""
SeisNet-1D: Preprocessing Module
Implements seismological signal conditioning:
1. Detrending and demeaning (scipy.signal.detrend)
2. Zero-phase Butterworth bandpass filtering (1-45 Hz, order 4) using Second-Order Sections (SOS)
3. Per-channel standard deviation normalization with epsilon safeguard for dead traces
"""

from typing import Union, Tuple, Optional
import numpy as np
import scipy.signal as signal
import torch


def butter_bandpass_sos(
    lowcut: float = 1.0,
    highcut: float = 45.0,
    fs: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Constructs Second-Order Sections (SOS) for a Butterworth bandpass filter.
    SOS representation avoids numerical instability and coefficient quantization
    errors present in high-order transfer function polynomials.
    """
    nyquist = 0.5 * fs
    low = max(0.01, lowcut / nyquist)
    high = min(0.99, highcut / nyquist)
    if low >= high:
        raise ValueError(f"Invalid bandpass cutoff: low ({lowcut} Hz) must be < high ({highcut} Hz)")
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 1.0,
    highcut: float = 45.0,
    fs: float = 100.0,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Applies zero-phase forward-backward Butterworth bandpass filter using SOS.
    Zero-phase filtering preserves exact phase arrival onset times without group delay.
    """
    sos = butter_bandpass_sos(lowcut, highcut, fs, order)
    # sosfiltfilt applies forward-backward zero phase shift along specified axis
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    return filtered


def detrend_and_demean(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Removes linear trend and DC bias from seismic waveform channels.
    """
    # detrend with type='linear' automatically removes both mean (DC) and best linear fit
    detrended = signal.detrend(data, axis=axis, type="linear")
    return detrended


def normalize_waveform(
    waveform: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Performs per-channel standardization (zero mean, unit standard deviation).
    Includes epsilon safeguard to prevent divide-by-zero on dead/clipped channels.
    If std is below epsilon, sets channel to zero to prevent noise amplification.
    """
    mean = np.mean(waveform, axis=axis, keepdims=True)
    std = np.std(waveform, axis=axis, keepdims=True)
    
    # Identify active channels vs dead/clipped channels
    active_mask = (std > eps)
    normalized = np.zeros_like(waveform)
    normalized = np.where(active_mask, (waveform - mean) / (std + eps), 0.0)
    return normalized.astype(np.float32)


def preprocess_stream(
    waveform: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    f_min: float = 1.0,
    f_max: float = 45.0,
    order: int = 4,
    detrend: bool = True,
    normalize: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Full standard seismological preprocessing pipeline for 3-component waveforms.
    
    Args:
        waveform: Array of shape (3, N) or (N, 3), where N is number of samples.
        fs: Sampling frequency in Hz (STEAD default = 100.0).
        f_min: Lower cutoff frequency (default = 1.0 Hz).
        f_max: Upper cutoff frequency (default = 45.0 Hz).
        order: Butterworth filter order.
        detrend: Whether to remove linear trend and DC component.
        normalize: Whether to normalize each channel to unit standard deviation.
        eps: Epsilon safeguard for zero-division.
        
    Returns:
        Preprocessed waveform as np.ndarray with shape (3, N) in float32.
    """
    is_torch = isinstance(waveform, torch.Tensor)
    if is_torch:
        data = waveform.detach().cpu().numpy()
    else:
        data = np.asarray(waveform, dtype=np.float32)

    # Ensure shape is (3, N)
    if data.ndim == 2:
        if data.shape[0] != 3 and data.shape[1] == 3:
            data = data.T  # Transpose from (N, 3) to (3, N)
    elif data.ndim == 1:
        data = data.reshape(1, -1)

    # Step 1: Detrend and demean
    if detrend:
        data = detrend_and_demean(data, axis=-1)

    # Step 2: Zero-phase bandpass filtering
    if f_min is not None and f_max is not None:
        data = butter_bandpass_filter(data, lowcut=f_min, highcut=f_max, fs=fs, order=order, axis=-1)

    # Step 3: Per-channel standard deviation normalization
    if normalize:
        data = normalize_waveform(data, axis=-1, eps=eps)

    return data.astype(np.float32)
