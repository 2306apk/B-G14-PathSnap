# PathSnap — Probability Calibration Report

**Date:** 2026-11-18 14:16:47

## Nested Cross-Fitted Out-of-Sample Calibration Metrics

| Model | Uncalibrated Brier | Calibrated Brier (Isotonic) | Uncalibrated Log Loss | Calibrated Log Loss (Iso) | Uncalibrated ECE | Calibrated ECE (Iso) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.0792 | 0.0387 | 0.2445 | 0.1314 | 0.1113 | 0.0016 |
| **Gaussian Naive Bayes** | 0.1163 | 0.0566 | 0.7245 | 0.1832 | 0.1232 | 0.0125 |
| **Decision Tree** | 0.0616 | 0.0376 | 0.2927 | 0.1314 | 0.0671 | 0.0038 |
| **Random Forest** | 0.0510 | 0.0328 | 0.1719 | 0.1102 | 0.0710 | 0.0020 |
| **HistGradientBoosting** | 0.0386 | 0.0333 | 0.1284 | 0.1133 | 0.0323 | 0.0021 |

## Calibration Rigour & Out-of-Sample Integrity
- **Nested Cross-Fitting:** To prevent in-sample overfitting, calibrators were cross-fitted across 5 training folds. No calibrator was evaluated on samples used to fit it.
- **Expected Calibration Error (ECE):** Cross-validated ECE reflects genuine out-of-sample calibration error.
- **HMM Emission Weighting:** Calibrated posteriors prevent overconfident emission scaling from dominating transition penalties.
