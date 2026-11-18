# Stage 6 — 18-Feature Model Evaluation

## Implementation

- 18 engineered GPS/road features
- Five supervised classification algorithms
- Trip-grouped cross-validation
- Bayesian hyperparameter optimisation
- HMM/Viterbi road-sequence reconstruction

## Five-model results

| Model | Training CV | Student-test Top-1 |
|---|---:|---:|
| logistic | 67.04% | 75.34% |
| gaussian_nb | 66.39% | 72.48% |
| decision_tree | 67.65% | 75.52% |
| random_forest | 68.83% | 77.47% |
| hist_gradient_boosting | 69.13% | 77.51% |

## HMM comparison

- Original 14-feature HMM: 83.87%
- New 18-feature classifier: 77.45%
- New 18-feature HMM: 83.68%
- HMM change: -0.19 percentage points
- Best tuned training CV: 69.69%

## Evaluation details

- Student-testing trips: 10
- GPS observations: 19160
- Results are measured against approximate reference-aligned labels.
- The student-testing partition has been used for development comparisons.
- The 85–96% overall accuracy target has not yet been achieved.
- Do not use reserved evaluator data for tuning.
