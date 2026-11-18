# PathSnap — Model Comparison & Architecture Report

**Generated Programmatically by `scripts/refine_phase2.py`**  
**Date:** 2026-11-18 14:16:47  
**Branch:** `ap2`

## 1. Consolidated Five-Model Evaluation Table

| Model | CV Top-1 Accuracy | CV Macro F1 | Student-Test Top-1 Acc | Student-Test Macro F1 | Student-Test Log Loss | Inference (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7993 +/- 0.1025 | 0.7749 +/- 0.0177 | 0.8116 | 0.7797 | 0.2213 | 0.0001 |
| **Gaussian Naive Bayes** | 0.7349 +/- 0.0893 | 0.7415 +/- 0.0361 | 0.7354 | 0.6654 | 1.2976 | 0.0001 |
| **Decision Tree** | 0.7691 +/- 0.1639 | 0.8237 +/- 0.0344 | 0.8080 | 0.8360 | 0.2075 | 0.0001 |
| **Random Forest** | 0.8235 +/- 0.1596 | 0.8485 +/- 0.0376 | 0.8213 | 0.8605 | 0.1423 | 0.0010 |
| **HistGradientBoosting** | 0.8224 +/- 0.1722 | 0.8704 +/- 0.0469 | 0.8224 | 0.8899 | 0.0977 | 0.0014 |
| **Tuned Random Forest** | Tuned | Tuned | 0.8221 | 0.8625 | 0.1389 | 0.0010 |

## 2. Key Architecture Findings
1. **Leading Model:** `Random Forest` achieved the highest generalization Top-1 accuracy and balanced precision/recall across folds.
2. **Generative vs Discriminative Baselines:** Gaussian Naive Bayes suffered from correlated distance features, while tree ensembles naturally learned axis-aligned decision boundaries.
3. **Calibrated Likelihood Integration:** Out-of-fold calibrated probabilities provide consistent emission inputs for the downstream 18-feature HMM sequential decoder.
