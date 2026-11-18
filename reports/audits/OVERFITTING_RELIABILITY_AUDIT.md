# Overfitting and Result-Reliability Audit

Status: **executed, inference-only**. Cached features and fitted artifacts were reused; no preprocessing, feature generation, model fitting, CV fitting, or tuning was rerun.

## Verdict

**Conditionally trustworthy.** The saved student-test figures reproduce exactly and no cross-trip row leakage or student-test fitting was found. However, CV is measured on a globally balanced sample rather than the natural candidate distribution, probability outputs are prior-shifted, student-test results were used for model ranking, and approximate/missing labels materially affect some training trips.

## 1. Leakage and split evidence

- All three reconstructed `GroupKFold` splits have zero trip overlap. Fold details are in `outputs/csv/reliability_cv_fold_integrity.csv`.
- The deterministic sample contains 80 training trip groups. Student-test trip IDs are absent from it.
- Imputation and scaling are pipeline steps fitted separately inside each CV fold. No global learned scaler/imputer is present.
- Features are trip-local and causal. Reference fields are retained for evaluation but are absent from `FEATURE_COLUMNS`.
- Candidate radius/outlier detection uses training labels only. Student-test labels do not affect features, fitting, preprocessing, or HMM parameters.
- **Selection caveat:** `initial_model_results.csv` is sorted by student-test Top-1, and HistGradientBoosting was chosen for HMM evaluation from that ranking. This is permitted development-stage comparison, but the student-test set is no longer an untouched estimate after selection.
- **Sampling caveat:** class-stratified sampling is performed globally before GroupKFold. It does not mix trips across folds, but validation labels influence which validation rows enter the artificial 50/50 sample.

## 2. Train vs GroupKFold CV vs student test

| model | train_sample_macro_f1 | group_cv_macro_f1 | train_minus_cv_macro_f1 | student_candidate_macro_f1 | cv_minus_student_macro_f1 | student_top1_accuracy |
|---|---|---|---|---|---|---|
| hist_gradient_boosting | 0.962424 | 0.936107 | 0.026317 | 0.844368 | 0.091739 | 0.822338 |
| random_forest | 0.981732 | 0.938742 | 0.042990 | 0.843527 | 0.095214 | 0.822129 |
| extra_trees | 0.979791 | 0.937657 | 0.042134 | 0.831693 | 0.105965 | 0.816962 |
| mlp | 0.951426 | 0.933995 | 0.017431 | 0.820820 | 0.113176 | 0.809969 |
| logistic_regression | 0.926382 | 0.924291 | 0.002091 | 0.751118 | 0.173173 | 0.796921 |

- Train-CV gaps above show resubstitution overfitting on the exact balanced sample used to fit saved artifacts. Tree ensembles show the largest gaps; logistic regression is the least overfit by this measure.
- CV-to-student gaps are not directly apples-to-apples because CV uses 50% positives while student candidate prevalence is 5.901798%.

Worst student-test trip per model:

| model | trip_id | top1_accuracy | candidate_macro_f1 |
|---|---|---|---|
| extra_trees | 00000077 | 0.631041 | 0.774600 |
| hist_gradient_boosting | 00000077 | 0.642547 | 0.778377 |
| logistic_regression | 00000077 | 0.590897 | 0.700819 |
| mlp | 00000077 | 0.641780 | 0.765508 |
| random_forest | 00000077 | 0.637689 | 0.777743 |

Full per-trip results are in `outputs/csv/reliability_student_per_trip.csv`; aggregate scores conceal substantial trip variation.

## 3. Sampling, calibration, and label reliability

- Natural training positive rate: 6.626632%; sampled rate: 50.000000%; student-test rate: 5.901798%.
- Global balancing therefore changes the class prior by 7.55x. Candidate accuracy, log loss, and 0.5-threshold metrics from balanced CV do not represent deployment prevalence.
- Sample trip shares range from 0.156667% to 6.378000%; sampling is row-weighted, not trip-balanced.

| model | student_positive_rate | student_mean_probability | student_brier | student_ece_10_bin | student_log_loss |
|---|---|---|---|---|---|
| hist_gradient_boosting | 0.059018 | 0.112581 | 0.034467 | 0.053563 | 0.116687 |
| random_forest | 0.059018 | 0.118083 | 0.034788 | 0.059065 | 0.120731 |
| extra_trees | 0.059018 | 0.132868 | 0.038439 | 0.073850 | 0.133511 |
| mlp | 0.059018 | 0.121062 | 0.040856 | 0.062044 | 0.136118 |
| logistic_regression | 0.059018 | 0.153277 | 0.061133 | 0.094259 | 0.183448 |

- Mean probabilities exceed the natural student positive rate for every model, consistent with prior distortion. Probabilities should not be treated as calibrated HMM emissions without grouped calibration or prior correction.
- 6,398 training observations have candidates but no positive row because the assigned reference arc is missing; another 15,286 observations have no candidates and never enter classifier rows. Candidate metrics reward all-negative missed observations, while observation Top-1 necessarily counts them wrong when present.
- Trips `00000033` and `00000054` contribute 6,974 sampled rows but only 2,184 positives. They were excluded from radius calibration, not from classifier fitting. Their approximate labels can bias the classifier toward negatives.
- Reference assignments are monotonic approximations from straight endpoint geometries, not independently verified ground truth. Label uncertainty is not used as a sample weight or filter.

## 4. Reproducibility of reported figures

- Saved model artifacts regenerated all student predictions without refitting. Maximum absolute difference from the three reported metrics checked per model was 0.000e+00.
- SHA-256 hashes of regenerated float64 probability arrays are stored in `outputs/csv/reliability_model_summary.csv`.
- Initial raw prediction arrays and out-of-fold predictions were not saved. Student metrics are reproducible from artifacts; historical CV scores can be checked for group integrity and logs, but exact fold scores require refitting because fold artifacts/predictions were not retained.

## 5. Corrections needed, ranked

1. **Critical:** perform negative sampling inside each training fold only (or use all rows/class weights), evaluate validation folds at natural prevalence, and persist fold assignments plus out-of-fold probabilities.
2. **High:** calibrate probabilities with grouped out-of-fold predictions or apply an explicit prior correction before HMM emissions; report Brier/ECE alongside log loss.
3. **High:** exclude, down-weight, or separately model uncertain/missing-reference observations and trips `00000033`/`00000054`; report sensitivity with and without them.
4. **High:** report per-trip macro summaries, worst-trip performance, candidate coverage, and route continuity for every model, not only pooled candidate metrics.
5. **Medium:** treat student-test as development data explicitly; use nested/grouped training validation for selection and reserve a separate untouched partition for final generalization claims.
6. **Medium:** persist raw student predictions, OOF predictions, model/config hashes, and exact sampled row identifiers.

## Evidence files

- `outputs/csv/reliability_model_summary.csv`
- `outputs/csv/reliability_student_per_trip.csv`
- `outputs/csv/reliability_cv_fold_integrity.csv`
- `outputs/csv/reliability_training_trip_quality.csv`
