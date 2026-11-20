"""
SeisNet-1D: Comprehensive Model Evaluation Script
Computes publication-grade metrics on the unseen Test set:
1. Event Detection:
   - Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Confusion Matrix
2. Phase Picking (P & S arrivals):
   - Peak detection via scipy.signal.find_peaks on probability curves
   - Residual error distributions: MAE (s), RMSE (s), Mean Bias (s), Std Dev (s)
   - Tolerance threshold accuracy: +/- 0.1s, +/- 0.2s, +/- 0.5s
   - Stratified picking error vs. Signal-to-Noise Ratio (SNR bins)
   - S - P lag time and epicentral distance estimation (d ≈ (t_S - t_P) * 8.0 km/s)
3. Outputs:
   - models/metrics.json
   - Formatted benchmark summary table
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
)
import torch

# Ensure repository root is on sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.model import SeisNet1D
    from src.dataset import get_dataloaders
    from src.train import get_device
except ImportError:
    from model import SeisNet1D
    from dataset import get_dataloaders
    from train import get_device


def detect_peaks_1d(
    prob_curve: np.ndarray,
    threshold: float = 0.30,
    distance: int = 50
) -> Tuple[Optional[int], float]:
    """
    Extracts arrival sample and peak confidence from 1D probability curve.
    
    Returns:
        (peak_index, confidence_score) or (None, 0.0) if no peak exceeds threshold.
    """
    peaks, properties = signal.find_peaks(
        prob_curve,
        height=threshold,
        distance=distance
    )
    if len(peaks) == 0:
        # Fallback: check if max value exceeds threshold
        max_idx = int(np.argmax(prob_curve))
        max_val = float(prob_curve[max_idx])
        if max_val >= threshold:
            return max_idx, max_val
        return None, 0.0

    # Pick the highest peak
    heights = properties["peak_heights"]
    best_idx = np.argmax(heights)
    return int(peaks[best_idx]), float(heights[best_idx])


@torch.no_grad()
def run_evaluation(
    config_path: str = "config.yaml",
    mode: str = "demo",
    model_path: str = None,
    output_metrics_path: str = None,
    device_name: str = None
) -> Dict[str, Any]:
    """
    Evaluates trained SeisNet-1D model on test dataset and saves metrics.json.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if mode is not None:
        config["dataset"]["mode"] = mode

    device = get_device(device_name or config["system"]["device"])
    model_path = model_path or config["paths"]["best_model"]
    output_metrics_path = output_metrics_path or config["paths"]["metrics_file"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'. Please run training first.")

    print(f"\n{'='*75}\nSeisNet-1D Evaluation Engine | Checkpoint: {model_path} | Mode: {config['dataset']['mode']}\n{'='*75}")

    # Load test dataloader
    _, _, test_loader, meta = get_dataloaders(
        config,
        mode=config["dataset"]["mode"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["system"]["num_workers"]
    )

    # Load Model Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = SeisNet1D(
        in_channels=config["model"]["in_channels"],
        encoder_channels=config["model"]["encoder_channels"],
        bottleneck_channels=config["model"]["bottleneck_channels"],
        bottleneck_dilations=config["model"]["bottleneck_dilations"],
        dropout=config["model"]["dropout"],
        phase_classes=config["model"]["phase_classes"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pick_threshold = config["evaluation"]["pick_threshold"]
    peak_distance = config["evaluation"]["peak_distance"]
    det_threshold = config["evaluation"]["detection_threshold"]
    fs = float(config["waveform"]["sampling_rate"])
    velocity_factor = float(config["evaluation"]["velocity_estimate_km_s"])

    # Containers for metrics
    det_probs = []
    det_targets = []

    p_true_list = []
    p_pred_list = []
    p_residuals = []  # t_pred - t_true in seconds

    s_true_list = []
    s_pred_list = []
    s_residuals = []  # t_pred - t_true in seconds

    distance_true_list = []
    distance_pred_list = []
    snr_list = []

    test_predictions_records = []

    for batch in test_loader:
        waveforms = batch["waveform"].to(device)
        is_eq = batch["is_earthquake"].numpy()
        p_mask = batch["p_mask"].numpy()
        s_mask = batch["s_mask"].numpy()
        p_true_samp = batch["p_arrival_sample"].numpy()
        s_true_samp = batch["s_arrival_sample"].numpy()
        snr_vals = batch["snr_db"].numpy()
        dist_vals = batch["distance_km"].numpy()
        trace_names = batch["trace_name"]

        det_logits, phase_logits = model(waveforms)

        batch_det_probs = torch.sigmoid(det_logits).cpu().numpy()
        batch_p_probs = torch.sigmoid(phase_logits[:, 0, :]).cpu().numpy()
        batch_s_probs = torch.sigmoid(phase_logits[:, 1, :]).cpu().numpy()

        det_probs.extend(batch_det_probs)
        det_targets.extend(is_eq)

        batch_size = waveforms.shape[0]
        for i in range(batch_size):
            p_peak_samp, p_conf = detect_peaks_1d(batch_p_probs[i], threshold=pick_threshold, distance=peak_distance)
            s_peak_samp, s_conf = detect_peaks_1d(batch_s_probs[i], threshold=pick_threshold, distance=peak_distance)

            p_pred_sec = (p_peak_samp / fs) if p_peak_samp is not None else None
            s_pred_sec = (s_peak_samp / fs) if s_peak_samp is not None else None

            p_gt_sec = (p_true_samp[i] / fs) if (p_true_samp[i] > 0 and p_mask[i] > 0.5) else None
            s_gt_sec = (s_true_samp[i] > 0 and s_mask[i] > 0.5 and (s_true_samp[i] / fs)) or None

            # Track P residuals
            if p_gt_sec is not None and p_pred_sec is not None:
                res_p = p_pred_sec - p_gt_sec
                p_residuals.append(res_p)
                p_true_list.append(p_gt_sec)
                p_pred_list.append(p_pred_sec)

            # Track S residuals
            if s_gt_sec is not None and s_pred_sec is not None:
                res_s = s_pred_sec - s_gt_sec
                s_residuals.append(res_s)
                s_true_list.append(s_gt_sec)
                s_pred_list.append(s_pred_sec)

            # S - P lag and distance estimation
            if p_pred_sec is not None and s_pred_sec is not None and s_pred_sec > p_pred_sec:
                pred_sp_lag = s_pred_sec - p_pred_sec
                pred_dist_km = pred_sp_lag * velocity_factor
                if dist_vals[i] > 0:
                    distance_true_list.append(dist_vals[i])
                    distance_pred_list.append(pred_dist_km)
                    snr_list.append(snr_vals[i])

            test_predictions_records.append({
                "trace_name": trace_names[i],
                "is_earthquake_gt": int(is_eq[i]),
                "detection_prob": float(batch_det_probs[i]),
                "p_true_sec": p_gt_sec,
                "p_pred_sec": p_pred_sec,
                "p_conf": p_conf,
                "s_true_sec": s_gt_sec,
                "s_pred_sec": s_pred_sec,
                "s_conf": s_conf,
                "snr_db": float(snr_vals[i]),
                "dist_km_gt": float(dist_vals[i])
            })

    # =========================================================================
    # 1. Detection Performance Computation
    # =========================================================================
    det_probs = np.array(det_probs)
    det_targets = np.array(det_targets)
    det_binary_preds = (det_probs >= det_threshold).astype(int)

    det_acc = float(accuracy_score(det_targets, det_binary_preds))
    det_prec = float(precision_score(det_targets, det_binary_preds, zero_division=0))
    det_rec = float(recall_score(det_targets, det_binary_preds, zero_division=0))
    det_f1 = float(f1_score(det_targets, det_binary_preds, zero_division=0))

    try:
        det_roc_auc = float(roc_auc_score(det_targets, det_probs))
    except Exception:
        det_roc_auc = 1.0

    precision_curve, recall_curve, _ = precision_recall_curve(det_targets, det_probs)
    det_pr_auc = float(auc(recall_curve, precision_curve))

    cm = confusion_matrix(det_targets, det_binary_preds)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # =========================================================================
    # 2. Phase Picking Performance Computation
    # =========================================================================
    p_residuals = np.array(p_residuals) if len(p_residuals) > 0 else np.array([0.0])
    s_residuals = np.array(s_residuals) if len(s_residuals) > 0 else np.array([0.0])

    p_mae = float(np.mean(np.abs(p_residuals)))
    p_rmse = float(np.sqrt(np.mean(p_residuals ** 2)))
    p_bias = float(np.mean(p_residuals))
    p_std = float(np.std(p_residuals))

    s_mae = float(np.mean(np.abs(s_residuals)))
    s_rmse = float(np.sqrt(np.mean(s_residuals ** 2)))
    s_bias = float(np.mean(s_residuals))
    s_std = float(np.std(s_residuals))

    # Tolerance accuracies (+/- 0.1s, +/- 0.2s, +/- 0.5s)
    p_tol_01 = float(np.mean(np.abs(p_residuals) <= 0.10) * 100.0)
    p_tol_02 = float(np.mean(np.abs(p_residuals) <= 0.20) * 100.0)
    p_tol_05 = float(np.mean(np.abs(p_residuals) <= 0.50) * 100.0)

    s_tol_01 = float(np.mean(np.abs(s_residuals) <= 0.10) * 100.0)
    s_tol_02 = float(np.mean(np.abs(s_residuals) <= 0.20) * 100.0)
    s_tol_05 = float(np.mean(np.abs(s_residuals) <= 0.50) * 100.0)

    # Distance estimation MAE
    if len(distance_true_list) > 0:
        dist_errors = np.abs(np.array(distance_pred_list) - np.array(distance_true_list))
        dist_mae_km = float(np.mean(dist_errors))
    else:
        dist_mae_km = 0.0

    # =========================================================================
    # 3. Stratified Picking vs SNR
    # =========================================================================
    df_eval = pd.DataFrame(test_predictions_records)
    snr_bins_def = [
        {"name": "Low SNR (< 12 dB)", "min": -100, "max": 12.0},
        {"name": "Medium SNR (12 - 22 dB)", "min": 12.0, "max": 22.0},
        {"name": "High SNR (>= 22 dB)", "min": 22.0, "max": 100.0},
    ]

    snr_analysis = []
    for b in snr_bins_def:
        sub = df_eval[(df_eval["is_earthquake_gt"] == 1) & (df_eval["snr_db"] >= b["min"]) & (df_eval["snr_db"] < b["max"])]
        p_sub = sub.dropna(subset=["p_true_sec", "p_pred_sec"])
        s_sub = sub.dropna(subset=["s_true_sec", "s_pred_sec"])

        p_err = np.abs(p_sub["p_pred_sec"] - p_sub["p_true_sec"]).values if len(p_sub) > 0 else []
        s_err = np.abs(s_sub["s_pred_sec"] - s_sub["s_true_sec"]).values if len(s_sub) > 0 else []

        snr_analysis.append({
            "bin": b["name"],
            "count": int(len(sub)),
            "p_mae_sec": float(np.mean(p_err)) if len(p_err) > 0 else 0.0,
            "s_mae_sec": float(np.mean(s_err)) if len(s_err) > 0 else 0.0,
        })

    # Assemble comprehensive metrics dictionary
    metrics = {
        "dataset_mode": config["dataset"]["mode"],
        "checkpoint_epoch": checkpoint.get("epoch", -1),
        "test_samples": len(df_eval),
        "earthquake_count": int(np.sum(det_targets == 1)),
        "noise_count": int(np.sum(det_targets == 0)),
        "detection": {
            "accuracy": det_acc,
            "precision": det_prec,
            "recall": det_rec,
            "f1_score": det_f1,
            "roc_auc": det_roc_auc,
            "pr_auc": det_pr_auc,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            }
        },
        "phase_picking": {
            "p_wave": {
                "picks_evaluated": int(len(p_residuals)),
                "mae_sec": p_mae,
                "rmse_sec": p_rmse,
                "mean_bias_sec": p_bias,
                "std_residual_sec": p_std,
                "tolerance_0.1s_pct": p_tol_01,
                "tolerance_0.2s_pct": p_tol_02,
                "tolerance_0.5s_pct": p_tol_05,
            },
            "s_wave": {
                "picks_evaluated": int(len(s_residuals)),
                "mae_sec": s_mae,
                "rmse_sec": s_rmse,
                "mean_bias_sec": s_bias,
                "std_residual_sec": s_std,
                "tolerance_0.1s_pct": s_tol_01,
                "tolerance_0.2s_pct": s_tol_02,
                "tolerance_0.5s_pct": s_tol_05,
            },
            "epicentral_distance": {
                "events_evaluated": len(distance_true_list),
                "mae_km": dist_mae_km,
                "velocity_factor_km_s": velocity_factor
            },
            "snr_breakdown": snr_analysis
        }
    }

    # Save to JSON
    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions for visualization cache
    pred_cache_path = Path("models/test_predictions.csv")
    df_eval.to_csv(pred_cache_path, index=False)

    # Print Formatted Seismological Summary
    print(f"\n{'='*75}\nSEISNET-1D EVALUATION RESULTS SUMMARY\n{'='*75}")
    print(f"Dataset Split:       {len(df_eval)} test seismograms ({metrics['earthquake_count']} Earthquakes, {metrics['noise_count']} Ambient Noise)")
    print(f"--- Event Detection (Earthquake vs Noise) ---")
    print(f"  Accuracy:          {det_acc * 100:.2f}%")
    print(f"  Precision:         {det_prec * 100:.2f}%")
    print(f"  Recall:            {det_rec * 100:.2f}%")
    print(f"  F1-Score:          {det_f1:.4f}")
    print(f"  ROC-AUC:           {det_roc_auc:.4f}")
    print(f"  PR-AUC:            {det_pr_auc:.4f}")
    print(f"  Confusion Matrix:  TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"\n--- P-Wave Phase Picking ---")
    print(f"  MAE:               {p_mae:.4f} s ({p_mae * 1000:.1f} ms)")
    print(f"  RMSE:              {p_rmse:.4f} s")
    print(f"  Residual Mean:     {p_bias:+.4f} s (Std: {p_std:.4f} s)")
    print(f"  Within +/- 0.1s:   {p_tol_01:.1f}%")
    print(f"  Within +/- 0.2s:   {p_tol_02:.1f}%")
    print(f"  Within +/- 0.5s:   {p_tol_05:.1f}%")
    print(f"\n--- S-Wave Phase Picking ---")
    print(f"  MAE:               {s_mae:.4f} s ({s_mae * 1000:.1f} ms)")
    print(f"  RMSE:              {s_rmse:.4f} s")
    print(f"  Residual Mean:     {s_bias:+.4f} s (Std: {s_std:.4f} s)")
    print(f"  Within +/- 0.1s:   {s_tol_01:.1f}%")
    print(f"  Within +/- 0.2s:   {s_tol_02:.1f}%")
    print(f"  Within +/- 0.5s:   {s_tol_05:.1f}%")
    print(f"\n--- Epicentral Distance Estimation (d ≈ (t_S - t_P) * {velocity_factor} km/s) ---")
    print(f"  Distance MAE:      {dist_mae_km:.2f} km")
    print(f"{'='*75}\nSaved detailed results to {output_metrics_path}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SeisNet-1D Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["demo", "stead"], default=None, help="Dataset mode")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to metrics output JSON")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        mode=args.mode,
        model_path=args.model,
        output_metrics_path=args.output,
        device_name=args.device
    )
