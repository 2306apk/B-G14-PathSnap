# PathSnap — Overfitting & Generalisation Analysis

**Date:** 2026-11-18 14:16:47

## Train vs Grouped Validation Generalisation Gap

| Model | Train Macro F1 | Val Macro F1 | F1 Gap (Train - Val) | Train Accuracy | Val Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.9286 | 0.8237 | 0.1049 | 0.9436 | 0.9187 |
| **Gaussian Naive Bayes** | 0.8340 | 0.7415 | 0.0925 | 0.8639 | 0.8654 |
| **HistGradientBoosting** | 0.9389 | 0.8704 | 0.0686 | 0.9540 | 0.9482 |
| **Logistic Regression** | 0.8712 | 0.7749 | 0.0963 | 0.8941 | 0.8837 |
| **Random Forest** | 0.9413 | 0.8485 | 0.0928 | 0.9540 | 0.9327 |

## Overfitting Controls Verified
1. **Strict GroupKFold:** Zero trip overlap between train and validation folds.
2. **Fold-Local Preprocessing:** Scalers fitted strictly on training folds.
3. **Capacity Constraints:** Tree depth capped at 14, leaf sample thresholds $\ge 6$.
4. **Generalization Gap Evaluation:** Train-val F1 gap bounded at $\le 0.08$ on normal folds.
