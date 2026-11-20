"""
SeisNet-1D: Multi-Task Loss Functions
Combines:
1. Binary Cross-Entropy with Logits for Event Detection (Earthquake vs Noise)
2. Masked Focal / Weighted BCE Loss for Continuous Gaussian Phase Arrival Curves (P & S)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedFocalPhaseLoss(nn.Module):
    """
    Focal Loss for continuous 1D probability targets (Gaussian curves) with sample masking.
    
    Given logits z and target probability y in [0, 1]:
      p = sigmoid(z)
      FL(p, y) = - alpha * y * (1 - p)^gamma * log(p + eps)
                 - (1 - alpha) * (1 - y) * p^gamma * log(1 - p + eps)
                 
    Handles the extreme sparsity of seismic phase arrivals (~80 positive samples out of 6000)
    and allows selective masking for missing phase picks (e.g. unpicked S-phases in real STEAD).
    """
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, N)
            targets: Target Gaussian curve of shape (B, N) with values in [0, 1]
            mask: Optional boolean or float mask of shape (B,) or (B, N)
        """
        probs = torch.sigmoid(logits)
        
        # Focal weights
        pt_pos = probs
        pt_neg = 1.0 - probs

        loss_pos = - self.alpha * targets * ((1.0 - pt_pos) ** self.gamma) * torch.log(pt_pos + self.eps)
        loss_neg = - (1.0 - self.alpha) * (1.0 - targets) * (pt_neg ** self.gamma) * torch.log(pt_neg + self.eps)
        loss = loss_pos + loss_neg  # (B, N)

        # Time-mean per trace
        trace_loss = torch.mean(loss, dim=-1)  # (B,)

        if mask is not None:
            # Flatten mask to (B,) if necessary
            if mask.ndim > 1:
                mask = mask.mean(dim=-1)
            mask = mask.float()
            valid_count = torch.sum(mask) + self.eps
            return torch.sum(trace_loss * mask) / valid_count
        else:
            return torch.mean(trace_loss)


class MultiTaskLoss(nn.Module):
    """
    Total multi-task objective for SeisNet-1D:
      L_total = w_det * L_det + w_p * L_p + w_s * L_s
    """
    def __init__(
        self,
        detection_weight: float = 1.0,
        p_phase_weight: float = 10.0,
        s_phase_weight: float = 10.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.w_det = detection_weight
        self.w_p = p_phase_weight
        self.w_s = s_phase_weight

        self.det_loss_fn = nn.BCEWithLogitsLoss()
        self.phase_loss_fn = MaskedFocalPhaseLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        detection_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        is_earthquake: torch.Tensor,
        p_target: torch.Tensor,
        s_target: torch.Tensor,
        p_mask: Optional[torch.Tensor] = None,
        s_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            detection_logits: (B,)
            phase_logits: (B, 2, 6000)
            is_earthquake: (B,) float32 (1.0 for EQ, 0.0 for Noise)
            p_target: (B, 6000)
            s_target: (B, 6000)
            p_mask: (B,)
            s_mask: (B,)
        """
        # Detection Loss
        loss_det = self.det_loss_fn(detection_logits, is_earthquake.float())

        # P-wave Phase Picking Loss
        p_logits = phase_logits[:, 0, :]  # (B, 6000)
        loss_p = self.phase_loss_fn(p_logits, p_target, mask=p_mask)

        # S-wave Phase Picking Loss
        s_logits = phase_logits[:, 1, :]  # (B, 6000)
        loss_s = self.phase_loss_fn(s_logits, s_target, mask=s_mask)

        # Combined Total Loss
        total_loss = self.w_det * loss_det + self.w_p * loss_p + self.w_s * loss_s

        loss_breakdown = {
            "loss_total": float(total_loss.item()),
            "loss_det": float(loss_det.item()),
            "loss_p": float(loss_p.item()),
            "loss_s": float(loss_s.item()),
        }

        return total_loss, loss_breakdown
