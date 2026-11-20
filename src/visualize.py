"""
SeisNet-1D: Publication-Grade Visualizations Generator
Produces 4 high-DPI (300 DPI) publication figures in assets/:
1. training_curves.png: Multi-task loss trajectories, detection F1, and phase picking MAE over epochs
2. detection_performance.png: Confusion Matrix, ROC Curve, and Precision-Recall Curve
3. waveform_picks_panel.png: 4-panel showcase of 3-channel seismograms (E, N, Z) with ground-truth vs.
   predicted P & S arrivals, continuous probability curves, S-P lag, and epicentral distance estimation
4. pick_residuals_snr.png: P/S residual error distributions (histograms + KDE) and stratified MAE vs. SNR
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import torch

# Ensure repository root is on sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.model import SeisNet1D
    from src.dataset import SteadDataset, split_catalog_by_group, get_dataloaders
    from src.train import get_device
    from src.evaluate import detect_peaks_1d
except ImportError:
    from model import SeisNet1D
    from dataset import SteadDataset, split_catalog_by_group, get_dataloaders
    from train import get_device
    from evaluate import detect_peaks_1d


def set_plot_style():
    """Applies clean, high-contrast publication aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 1.6,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })


def plot_training_curves(history: Dict[str, Any], output_path: str = "assets/training_curves.png"):
    """
    Figure 1: Training & Validation Loss, Task Breakdown, Detection F1, and Phase MAE.
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train Total Loss", color="#1f77b4", marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"], label="Val Total Loss", color="#d62728", marker="s", markersize=4)
    ax.set_title("(a) Multi-Task Loss Convergence", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted Loss")
    ax.legend(frameon=True, loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 2. Task Loss Breakdown
    ax = axes[0, 1]
    ax.plot(epochs, history["val_loss_det"], label="Val Detection Loss (BCE)", color="#2ca02c", linestyle="-")
    ax.plot(epochs, history["val_loss_p"], label="Val P-Wave Loss (Focal)", color="#9467bd", linestyle="--")
    ax.plot(epochs, history["val_loss_s"], label="Val S-Wave Loss (Focal)", color="#ff7f0e", linestyle="-.")
    ax.set_title("(b) Validation Loss by Objective Task", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task Loss")
    ax.legend(frameon=True, loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 3. Detection Metrics
    ax = axes[1, 0]
    ax.plot(epochs, history["val_det_f1"], label="Val Detection F1", color="#17becf", marker="^", markersize=4)
    ax.plot(epochs, history["val_det_acc"], label="Val Detection Accuracy", color="#bcbd22", linestyle=":")
    ax.set_title("(c) Event Detection Classification Metric", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score [0.0 - 1.0]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="lower right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 4. Phase Picking MAE
    ax = axes[1, 1]
    ax.plot(epochs, history["val_p_mae_sec"], label="P-Wave Arrival MAE (s)", color="#e377c2", marker="o", markersize=4)
    ax.plot(epochs, history["val_s_mae_sec"], label="S-Wave Arrival MAE (s)", color="#8c564b", marker="s", markersize=4)
    ax.set_title("(d) Phase Picking Mean Absolute Error", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (seconds)")
    ax.legend(frameon=True, loc="upper right")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle("SeisNet-1D: Deep Learning Multi-Task Training Trajectory", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved: {output_path}")


def plot_detection_performance(metrics: Dict[str, Any], predictions_df: pd.DataFrame, output_path: str = "assets/detection_performance.png"):
    """
    Figure 2: Confusion Matrix, ROC Curve, and PR Curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    y_true = predictions_df["is_earthquake_gt"].values
    y_scores = predictions_df["detection_prob"].values
    y_pred = (y_scores >= 0.5).astype(int)

    # 1. Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = np.array([[f"{count}\n({pct:.1%})" for count, pct in zip(row_c, row_p)] for row_c, row_p in zip(cm, cm_norm)])

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Ambient Noise", "Earthquake"],
        yticklabels=["Ambient Noise", "Earthquake"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_title("(a) Event Detection Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")

    # 2. ROC Curve
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#d62728", lw=2.2, label=f"SeisNet-1D (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("(b) Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    # 3. Precision-Recall Curve
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    baseline = np.mean(y_true)
    ax.plot(rec, prec, color="#2ca02c", lw=2.2, label=f"SeisNet-1D (PR-AUC = {pr_auc:.4f})")
    ax.plot([0, 1], [baseline, baseline], color="gray", linestyle="--", label=f"Prevalence Baseline ({baseline:.2f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(c) Precision-Recall (PR) Curve", fontweight="bold")
    ax.legend(loc="lower left", frameon=True)

    plt.suptitle("SeisNet-1D: Binary Seismic Event Classification Benchmark", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved: {output_path}")


def plot_waveform_showcase(
    model: torch.nn.Module,
    test_ds: SteadDataset,
    output_path: str = "assets/waveform_picks_panel.png",
    device: torch.device = torch.device("cpu")
):
    """
    Figure 3: 4-Panel Seismological Waveform & Phase Picking Showcase:
    - 3 earthquake events (high SNR, mid SNR, low SNR) + 1 ambient noise trace.
    - 3 components (E, N, Z) with predicted vs GT P and S arrivals.
    - Probability curves below each seismogram.
    - Calculated S-P lag time and estimated epicentral distance.
    """
    model.eval()
    fs = float(test_ds.fs)
    velocity_factor = 8.0  # km/s

    # Find sample indices representing diverse cases
    eq_indices = []
    noise_indices = []
    for idx in range(len(test_ds)):
        row = test_ds.df.iloc[idx]
        if row.get("trace_category") == "earthquake_local":
            eq_indices.append(idx)
        else:
            noise_indices.append(idx)

    # Sort earthquakes by SNR
    eq_sub = test_ds.df.iloc[eq_indices].copy()
    eq_sub["orig_idx"] = eq_indices
    eq_sub_sorted = eq_sub.sort_values(by="snr_db", ascending=False)

    selected_indices = []
    if len(eq_sub_sorted) >= 3:
        # High SNR
        selected_indices.append(int(eq_sub_sorted.iloc[0]["orig_idx"]))
        # Mid SNR
        selected_indices.append(int(eq_sub_sorted.iloc[len(eq_sub_sorted) // 2]["orig_idx"]))
        # Low SNR
        selected_indices.append(int(eq_sub_sorted.iloc[-1]["orig_idx"]))
    else:
        selected_indices.extend(eq_indices[:3])

    if len(noise_indices) > 0:
        selected_indices.append(noise_indices[0])
    elif len(test_ds) > 3:
        selected_indices.append(3)

    fig = plt.figure(figsize=(16, 14), dpi=300)
    gs = fig.add_gridspec(4, 2, width_ratios=[3.2, 1.0], hspace=0.35, wspace=0.18)

    time_sec = np.arange(6000) / fs

    for row_idx, sample_idx in enumerate(selected_indices[:4]):
        sample = test_ds[sample_idx]
        waveform = sample["waveform"].unsqueeze(0).to(device) # (1, 3, 6000)

        with torch.no_grad():
            det_logit, phase_logits = model(waveform)
            det_prob = float(torch.sigmoid(det_logit)[0].cpu().numpy())
            p_curve = torch.sigmoid(phase_logits[0, 0, :]).cpu().numpy()
            s_curve = torch.sigmoid(phase_logits[0, 1, :]).cpu().numpy()

        p_gt_samp = sample["p_arrival_sample"]
        s_gt_samp = sample["s_arrival_sample"]
        is_eq = (sample["is_earthquake"].item() > 0.5)

        # Detect peaks
        p_pred_samp, p_conf = detect_peaks_1d(p_curve, threshold=0.30, distance=50)
        s_pred_samp, s_conf = detect_peaks_1d(s_curve, threshold=0.30, distance=50)

        p_pred_sec = (p_pred_samp / fs) if p_pred_samp is not None else None
        s_pred_sec = (s_pred_samp / fs) if s_pred_samp is not None else None
        p_gt_sec = (p_gt_samp / fs) if p_gt_samp > 0 else None
        s_gt_sec = (s_gt_samp / fs) if s_gt_samp > 0 else None

        # Distance estimate
        if p_pred_sec is not None and s_pred_sec is not None and s_pred_sec > p_pred_sec:
            sp_lag = s_pred_sec - p_pred_sec
            est_dist_km = sp_lag * velocity_factor
        else:
            sp_lag = None
            est_dist_km = None

        # Left: Waveform Subplot
        ax_wave = fig.add_subplot(gs[row_idx, 0])
        wf = sample["waveform"].numpy() # (3, 6000) [E, N, Z]

        # Plot 3 components stacked or offset
        offset = 3.5
        ax_wave.plot(time_sec, wf[0] + offset, label="E (East-West)", color="#1f77b4", lw=0.9, alpha=0.9)
        ax_wave.plot(time_sec, wf[1] + 0.0,    label="N (North-South)", color="#2ca02c", lw=0.9, alpha=0.9)
        ax_wave.plot(time_sec, wf[2] - offset, label="Z (Vertical)", color="#6a3d9a", lw=0.9, alpha=0.9)

        # Plot ground truth vertical lines
        if p_gt_sec is not None:
            ax_wave.axvline(p_gt_sec, color="#d62728", linestyle="--", lw=1.6, label=f"GT P ({p_gt_sec:.2f}s)")
        if s_gt_sec is not None:
            ax_wave.axvline(s_gt_sec, color="#ff7f0e", linestyle="--", lw=1.6, label=f"GT S ({s_gt_sec:.2f}s)")

        # Plot predicted vertical lines
        if p_pred_sec is not None:
            ax_wave.axvline(p_pred_sec, color="#e41a1c", linestyle="-", lw=2.0, label=f"Pred P ({p_pred_sec:.2f}s)")
        if s_pred_sec is not None:
            ax_wave.axvline(s_pred_sec, color="#377eb8", linestyle="-", lw=2.0, label=f"Pred S ({s_pred_sec:.2f}s)")

        category_label = "Earthquake" if is_eq else "Ambient Noise"
        snr_label = f"SNR: {sample['snr_db']:.1f} dB" if is_eq else "Noise Field"
        title_text = f"Trace: {sample['trace_name']} | GT: {category_label} | {snr_label} | P(EQ)={det_prob:.3f}"
        if est_dist_km is not None:
            title_text += f" | Δt(S-P)={sp_lag:.2f}s | Est. Dist: {est_dist_km:.1f} km"

        ax_wave.set_title(title_text, fontsize=11, fontweight="bold", loc="left")
        ax_wave.set_xlim(0, 60)
        ax_wave.set_ylabel("Standardized Amp", fontsize=9)
        if row_idx == 3:
            ax_wave.set_xlabel("Time (seconds)", fontsize=11)
        else:
            ax_wave.set_xticklabels([])
        ax_wave.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)

        # Right: Probability Curves Subplot
        ax_prob = fig.add_subplot(gs[row_idx, 1])
        ax_prob.plot(time_sec, p_curve, color="#e41a1c", lw=1.8, label="P-Phase Prob")
        ax_prob.plot(time_sec, s_curve, color="#377eb8", lw=1.8, label="S-Phase Prob")
        ax_prob.axhline(0.30, color="gray", linestyle=":", lw=1.0, label="Pick Threshold (0.30)")
        ax_prob.set_xlim(0, 60)
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.set_ylabel("Probability", fontsize=9)
        ax_prob.set_title(f"Predicted Phase Curves (P_conf={p_conf:.2f}, S_conf={s_conf:.2f})", fontsize=10)
        if row_idx == 3:
            ax_prob.set_xlabel("Time (seconds)", fontsize=11)
        else:
            ax_prob.set_xticklabels([])
        ax_prob.legend(loc="upper right", fontsize=8, frameon=True)

    plt.suptitle("SeisNet-1D: 3-Component Seismic Waveforms, Pick Predictions & Phase Probabilities", fontsize=15, fontweight="bold", y=0.995)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved: {output_path}")


def plot_residuals_and_snr(metrics: Dict[str, Any], predictions_df: pd.DataFrame, output_path: str = "assets/pick_residuals_snr.png"):
    """
    Figure 4: P/S Arrival Residual Histograms and Stratified Picking MAE vs. SNR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    eq_df = predictions_df[predictions_df["is_earthquake_gt"] == 1].copy()

    # Calculate residuals
    p_valid = eq_df.dropna(subset=["p_true_sec", "p_pred_sec"])
    s_valid = eq_df.dropna(subset=["s_true_sec", "s_pred_sec"])

    p_res = (p_valid["p_pred_sec"] - p_valid["p_true_sec"]).values
    s_res = (s_valid["s_pred_sec"] - s_valid["s_true_sec"]).values

    # 1. P Residuals Histogram
    ax = axes[0]
    if len(p_res) > 0:
        bins = np.linspace(-0.8, 0.8, 33)
        sns.histplot(p_res, bins=bins, kde=True, color="#d62728", ax=ax, stat="density", alpha=0.6)
        p_mean = np.mean(p_res)
        p_std = np.std(p_res)
        p_mae = np.mean(np.abs(p_res))
        ax.axvline(0, color="black", linestyle="--", lw=1.2)
        ax.set_title(f"(a) P-Wave Arrival Residuals (Δt)\nMAE={p_mae:.3f}s | Mean={p_mean:+.3f}s | σ={p_std:.3f}s", fontweight="bold", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No P Picks Available", ha="center", va="center")
    ax.set_xlabel("Time Residual [t_pred - t_true] (seconds)")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.8, 0.8)

    # 2. S Residuals Histogram
    ax = axes[1]
    if len(s_res) > 0:
        bins = np.linspace(-1.0, 1.0, 33)
        sns.histplot(s_res, bins=bins, kde=True, color="#1f77b4", ax=ax, stat="density", alpha=0.6)
        s_mean = np.mean(s_res)
        s_std = np.std(s_res)
        s_mae = np.mean(np.abs(s_res))
        ax.axvline(0, color="black", linestyle="--", lw=1.2)
        ax.set_title(f"(b) S-Wave Arrival Residuals (Δt)\nMAE={s_mae:.3f}s | Mean={s_mean:+.3f}s | σ={s_std:.3f}s", fontweight="bold", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No S Picks Available", ha="center", va="center")
    ax.set_xlabel("Time Residual [t_pred - t_true] (seconds)")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.0, 1.0)

    # 3. Stratified MAE vs SNR Bins
    ax = axes[2]
    snr_data = metrics.get("phase_picking", {}).get("snr_breakdown", [])
    if len(snr_data) > 0:
        bin_names = [d["bin"].replace(" (< 12 dB)", "\n(<12dB)").replace(" (12 - 22 dB)", "\n(12-22dB)").replace(" (>= 22 dB)", "\n(>=22dB)") for d in snr_data]
        p_maes = [d["p_mae_sec"] for d in snr_data]
        s_maes = [d["s_mae_sec"] for d in snr_data]

        x = np.arange(len(bin_names))
        width = 0.35

        rects1 = ax.bar(x - width/2, p_maes, width, label="P-Wave MAE (s)", color="#e41a1c", alpha=0.85)
        rects2 = ax.bar(x + width/2, s_maes, width, label="S-Wave MAE (s)", color="#377eb8", alpha=0.85)

        ax.set_title("(c) Picking Accuracy vs. Signal-to-Noise Ratio", fontweight="bold", fontsize=11)
        ax.set_xlabel("SNR Bracket")
        ax.set_ylabel("Mean Absolute Error (seconds)")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names, fontsize=9)
        ax.legend(frameon=True, loc="upper right")

        # Add bar labels
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}s", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}s", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, "No SNR Breakdown Available", ha="center", va="center")

    plt.suptitle("SeisNet-1D: Arrival Time Residuals & Robustness Across Noise Regimes", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved: {output_path}")


def generate_all_plots(
    config_path: str = "config.yaml",
    mode: str = "demo",
    device_name: str = None
):
    """Generates all 4 publication charts."""
    set_plot_style()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if mode is not None:
        config["dataset"]["mode"] = mode

    device = get_device(device_name or config["system"]["device"])
    assets_dir = Path(config["paths"]["assets_dir"])
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}\nGenerating Publication Figures for SeisNet-1D\n{'='*75}")

    # 1. Training curves
    history_file = Path(config["paths"]["history_file"])
    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)
        plot_training_curves(history, str(assets_dir / "training_curves.png"))
    else:
        print(f"[Warning] {history_file} not found; skipping training curves plot.")

    # 2. Metrics & Test Predictions
    metrics_file = Path(config["paths"]["metrics_file"])
    pred_cache_file = Path("models/test_predictions.csv")

    if metrics_file.exists() and pred_cache_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        pred_df = pd.read_csv(pred_cache_file)
        plot_detection_performance(metrics, pred_df, str(assets_dir / "detection_performance.png"))
        plot_residuals_and_snr(metrics, pred_df, str(assets_dir / "pick_residuals_snr.png"))
    else:
        print(f"[Warning] Metrics or predictions cache not found. Run evaluate.py first.")

    # 3. Waveform Showcase
    model_path = Path(config["paths"]["best_model"])
    if model_path.exists():
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

        # Load test dataset
        _, _, test_loader, meta = get_dataloaders(
            config,
            mode=config["dataset"]["mode"],
            batch_size=32,
            num_workers=0
        )
        test_ds = test_loader.dataset
        plot_waveform_showcase(model, test_ds, str(assets_dir / "waveform_picks_panel.png"), device=device)
    else:
        print(f"[Warning] Model checkpoint {model_path} not found; skipping waveform showcase.")

    print(f"\n[Visualizer] All figures successfully written to {assets_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SeisNet-1D Publication Visualizations")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["demo", "stead"], default=None, help="Dataset mode")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    generate_all_plots(
        config_path=args.config,
        mode=args.mode,
        device_name=args.device
    )
