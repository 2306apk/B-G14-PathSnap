"""Probability calibration and reliability evaluation for candidate road-segment classifications.

Implements Brier score, log loss, Expected Calibration Error (ECE), out-of-fold calibration fitting
(Isotonic and Platt Sigmoid), and reliability diagram generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationReport:
    """Stores calibration evaluation metrics before and after calibration."""

    method: str
    brier_score_uncalibrated: float
    brier_score_calibrated: float
    log_loss_uncalibrated: float
    log_loss_calibrated: float
    ece_uncalibrated: float
    ece_calibrated: float


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculates the Expected Calibration Error (ECE) across n_bins equal-width probability bins."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_t)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_p >= low) & (y_p < high) if i < n_bins - 1 else (y_p >= low) & (y_p <= high)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = float(np.mean(y_t[mask]))
            bin_conf = float(np.mean(y_p[mask]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


class OutOfFoldCalibrator:
    """Fits and applies probability calibration on out-of-fold classifier scores."""

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method == "isotonic":
            self.regressor = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        elif self.method in ["sigmoid", "platt"]:
            self.regressor = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, oof_probs: np.ndarray, oof_labels: np.ndarray) -> OutOfFoldCalibrator:
        """Fits calibration curve using training out-of-fold predictions only."""
        p = np.asarray(oof_probs, dtype=np.float64).reshape(-1, 1 if self.method in ["sigmoid", "platt"] else 1)
        y = np.asarray(oof_labels, dtype=int)

        if self.method == "isotonic":
            self.regressor.fit(p.ravel(), y)
        else:
            self.regressor.fit(p, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated probabilities into calibrated probabilities."""
        p = np.asarray(probs, dtype=np.float64)
        if self.method == "isotonic":
            cal_p = self.regressor.predict(p.ravel())
        else:
            cal_p = self.regressor.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(cal_p, 1e-6, 1.0 - 1e-6)


def evaluate_calibration(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    method: str = "isotonic",
) -> CalibrationReport:
    """Calculates comparative calibration metrics (Brier, LogLoss, ECE)."""
    y_t = np.asarray(y_true, dtype=int)
    p_uncal = np.clip(np.asarray(uncalibrated_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    p_cal = np.clip(np.asarray(calibrated_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)

    brier_uncal = float(brier_score_loss(y_t, p_uncal))
    brier_cal = float(brier_score_loss(y_t, p_cal))

    ll_uncal = float(log_loss(y_t, p_uncal))
    ll_cal = float(log_loss(y_t, p_cal))

    ece_uncal = compute_expected_calibration_error(y_t, p_uncal)
    ece_cal = compute_expected_calibration_error(y_t, p_cal)

    return CalibrationReport(
        method=method,
        brier_score_uncalibrated=brier_uncal,
        brier_score_calibrated=brier_cal,
        log_loss_uncalibrated=ll_uncal,
        log_loss_calibrated=ll_cal,
        ece_uncalibrated=ece_uncal,
        ece_calibrated=ece_cal,
    )


def cross_val_evaluate_calibration(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    fold_ids: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, CalibrationReport]:
    """Evaluates probability calibration using strictly nested cross-fitting across folds.

    Guarantees that no calibrator is evaluated on predictions from folds used to fit it.
    """
    p_in = np.asarray(oof_probs, dtype=np.float64)
    y_in = np.asarray(oof_labels, dtype=int)
    f_in = np.asarray(fold_ids)

    unique_folds = np.unique(f_in)
    calibrated_oof = np.zeros_like(p_in)

    for k in unique_folds:
        val_mask = (f_in == k)
        tr_mask = ~val_mask

        calibrator = OutOfFoldCalibrator(method=method)
        calibrator.fit(p_in[tr_mask], y_in[tr_mask])
        calibrated_oof[val_mask] = calibrator.predict_proba(p_in[val_mask])

    report = evaluate_calibration(y_in, p_in, calibrated_oof, method=method)
    return calibrated_oof, report


def plot_calibration_curves(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    model_name: str,
    output_path: Optional[str | Path] = None,
    n_bins: int = 10,
) -> None:
    """Plots comparative reliability diagram (calibration curves)."""
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncalibrated_probs, n_bins=n_bins)
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=n_bins)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#d62728", label="Uncalibrated")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", color="#2ca02c", label="Calibrated (Isotonic)")

    plt.title(f"Reliability Diagram (Calibration Curve) — {model_name}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Empirical Accuracy)")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.close()

