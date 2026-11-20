"""
SeisNet-1D: Model Training Script
Implements full training and validation lifecycle:
- Multi-task loss backpropagation (detection + P & S Gaussian curves)
- Cosine Annealing LR scheduler
- Early stopping based on validation loss
- Model checkpointing (best_model.pt and latest_model.pt)
- Complete training history logging (models/history.json)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure repository root is on sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.model import SeisNet1D, get_model_summary
    from src.dataset import get_dataloaders
    from src.losses import MultiTaskLoss
except ImportError:
    from model import SeisNet1D, get_model_summary
    from dataset import get_dataloaders
    from losses import MultiTaskLoss


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolves computational device with CPU/GPU auto-detection."""
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    return device


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_det_loss = 0.0
    total_p_loss = 0.0
    total_s_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for batch in pbar:
        waveforms = batch["waveform"].to(device)
        is_eq = batch["is_earthquake"].to(device)
        p_target = batch["p_target"].to(device)
        s_target = batch["s_target"].to(device)
        p_mask = batch["p_mask"].to(device)
        s_mask = batch["s_mask"].to(device)

        optimizer.zero_grad()
        det_logits, phase_logits = model(waveforms)

        loss, breakdown = criterion(
            detection_logits=det_logits,
            phase_logits=phase_logits,
            is_earthquake=is_eq,
            p_target=p_target,
            s_target=s_target,
            p_mask=p_mask,
            s_mask=s_mask
        )

        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += breakdown["loss_total"]
        total_det_loss += breakdown["loss_det"]
        total_p_loss += breakdown["loss_p"]
        total_s_loss += breakdown["loss_s"]
        n_batches += 1

        pbar.set_postfix({
            "Loss": f"{breakdown['loss_total']:.4f}",
            "Det": f"{breakdown['loss_det']:.3f}",
            "P": f"{breakdown['loss_p']:.3f}",
            "S": f"{breakdown['loss_s']:.3f}"
        })

    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_loss_det": total_det_loss / max(1, n_batches),
        "train_loss_p": total_p_loss / max(1, n_batches),
        "train_loss_s": total_s_loss / max(1, n_batches),
    }


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates validation loss and basic performance metrics."""
    model.eval()
    total_loss = 0.0
    total_det_loss = 0.0
    total_p_loss = 0.0
    total_s_loss = 0.0
    n_batches = 0

    all_det_preds = []
    all_det_targets = []
    p_errors = []
    s_errors = []

    for batch in dataloader:
        waveforms = batch["waveform"].to(device)
        is_eq = batch["is_earthquake"].to(device)
        p_target = batch["p_target"].to(device)
        s_target = batch["s_target"].to(device)
        p_mask = batch["p_mask"].to(device)
        s_mask = batch["s_mask"].to(device)
        p_arr_gt = batch["p_arrival_sample"].numpy()
        s_arr_gt = batch["s_arrival_sample"].numpy()

        det_logits, phase_logits = model(waveforms)

        loss, breakdown = criterion(
            detection_logits=det_logits,
            phase_logits=phase_logits,
            is_earthquake=is_eq,
            p_target=p_target,
            s_target=s_target,
            p_mask=p_mask,
            s_mask=s_mask
        )

        total_loss += breakdown["loss_total"]
        total_det_loss += breakdown["loss_det"]
        total_p_loss += breakdown["loss_p"]
        total_s_loss += breakdown["loss_s"]
        n_batches += 1

        # Detection predictions
        det_probs = torch.sigmoid(det_logits).cpu().numpy()
        all_det_preds.extend(det_probs)
        all_det_targets.extend(is_eq.cpu().numpy())

        # Phase picking predictions (argmax approximation for rapid validation)
        p_probs = torch.sigmoid(phase_logits[:, 0, :]).cpu().numpy()
        s_probs = torch.sigmoid(phase_logits[:, 1, :]).cpu().numpy()

        p_peaks = np.argmax(p_probs, axis=-1)
        s_peaks = np.argmax(s_probs, axis=-1)

        for b in range(len(p_arr_gt)):
            if p_arr_gt[b] > 0 and p_mask[b].item() > 0.5:
                p_errors.append(abs(p_peaks[b] - p_arr_gt[b]) / 100.0) # seconds
            if s_arr_gt[b] > 0 and s_mask[b].item() > 0.5:
                s_errors.append(abs(s_peaks[b] - s_arr_gt[b]) / 100.0) # seconds

    # Compute validation detection metrics
    all_det_preds = np.array(all_det_preds)
    all_det_targets = np.array(all_det_targets)
    binary_preds = (all_det_preds >= 0.5).astype(float)
    acc = float(np.mean(binary_preds == all_det_targets))

    tp = np.sum((binary_preds == 1.0) & (all_det_targets == 1.0))
    fp = np.sum((binary_preds == 1.0) & (all_det_targets == 0.0))
    fn = np.sum((binary_preds == 0.0) & (all_det_targets == 1.0))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    val_p_mae = float(np.mean(p_errors)) if len(p_errors) > 0 else 0.0
    val_s_mae = float(np.mean(s_errors)) if len(s_errors) > 0 else 0.0

    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_loss_det": total_det_loss / max(1, n_batches),
        "val_loss_p": total_p_loss / max(1, n_batches),
        "val_loss_s": total_s_loss / max(1, n_batches),
        "val_det_acc": acc,
        "val_det_precision": precision,
        "val_det_recall": recall,
        "val_det_f1": f1,
        "val_p_mae_sec": val_p_mae,
        "val_s_mae_sec": val_s_mae
    }


def train_pipeline(
    config_path: str = "config.yaml",
    mode: str = "demo",
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    device_name: str = None
) -> Dict[str, Any]:
    """Executes the full training and checkpointing pipeline."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Overrides
    if mode is not None:
        config["dataset"]["mode"] = mode
    if epochs is not None:
        config["training"]["epochs"] = epochs
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if lr is not None:
        config["training"]["learning_rate"] = lr

    seed = config["system"]["seed"]
    set_seed(seed)

    device = get_device(device_name or config["system"]["device"])
    print(f"\n{'='*70}\nSeisNet-1D Training Pipeline | Mode: {config['dataset']['mode'].upper()} | Device: {device}\n{'='*70}")

    # Load Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        config,
        mode=config["dataset"]["mode"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["system"]["num_workers"]
    )

    # Initialize Architecture
    model = SeisNet1D(
        in_channels=config["model"]["in_channels"],
        encoder_channels=config["model"]["encoder_channels"],
        bottleneck_channels=config["model"]["bottleneck_channels"],
        bottleneck_dilations=config["model"]["bottleneck_dilations"],
        dropout=config["model"]["dropout"],
        phase_classes=config["model"]["phase_classes"]
    ).to(device)

    summary = get_model_summary(model)
    print(f"[Model Architecture] {summary['model_name']} | Trainable Parameters: {summary['trainable_parameters']:,} (~{summary['model_size_mb']} MB)")

    # Loss & Optimizer
    criterion = MultiTaskLoss(
        detection_weight=config["loss"]["detection_weight"],
        p_phase_weight=config["loss"]["p_phase_weight"],
        s_phase_weight=config["loss"]["s_phase_weight"],
        focal_alpha=config["loss"]["focal_alpha"],
        focal_gamma=config["loss"]["focal_gamma"]
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    total_epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config["training"]["min_lr"]
    )

    # Output directory
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(config["paths"]["best_model"])
    latest_model_path = Path(config["paths"]["latest_model"])
    history_path = Path(config["paths"]["history_file"])

    best_val_loss = float("inf")
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0

    history = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "train_loss_det": [],
        "train_loss_p": [],
        "train_loss_s": [],
        "val_loss": [],
        "val_loss_det": [],
        "val_loss_p": [],
        "val_loss_s": [],
        "val_det_acc": [],
        "val_det_f1": [],
        "val_p_mae_sec": [],
        "val_s_mae_sec": []
    }

    start_time = time.time()
    print(f"\nStarting training for {total_epochs} epochs with early stopping patience={patience}...\n")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_res = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=config["training"]["gradient_clip"]
        )

        val_res = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # Record history
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        for k, v in train_res.items():
            if k not in history:
                history[k] = []
            history[k].append(v)
        for k, v in val_res.items():
            if k not in history:
                history[k] = []
            history[k].append(v)

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} [{epoch_duration:.1f}s] "
            f"| Train Loss: {train_res['train_loss']:.4f} "
            f"| Val Loss: {val_res['val_loss']:.4f} "
            f"| Det F1: {val_res['val_det_f1']:.4f} "
            f"| P-MAE: {val_res['val_p_mae_sec']:.3f}s "
            f"| S-MAE: {val_res['val_s_mae_sec']:.3f}s"
        )

        # Checkpoint: Save Latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_res
        }, latest_model_path)

        # Checkpoint: Save Best
        if val_res["val_loss"] < best_val_loss:
            best_val_loss = val_res["val_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_res
            }, best_model_path)
            print(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f}) to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} consecutive epochs.")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s. Best Val Loss: {best_val_loss:.4f}")

    # Save training history JSON
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Logging] Saved full training trajectory to {history_path}")

    return {"best_val_loss": best_val_loss, "epochs_trained": epoch, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeisNet-1D on STEAD or Demo chunk")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["demo", "stead"], default=None, help="Dataset mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    train_pipeline(
        config_path=args.config,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device
    )
