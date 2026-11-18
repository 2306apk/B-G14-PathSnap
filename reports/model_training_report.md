# PathSnap — Model Training & Architectural Analysis Report

**Date:** 2026-11-18 13:56:21

## Compulsory Model Analysis for Undergraduate Viva

### 1. Logistic Regression (Linear & Probabilistic)
- **How it works:** Models log-odds as a linear combination of features: $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$.
- **Why suitable:** Serves as the natural linear reference baseline. Produces calibrated posterior probabilities directly suitable for HMM emission weighting.
- **Assumptions:** Linear relationship between features and log-odds; independent observations.
- **Overfitting Controls:** L2 ridge regularization ($C=1.0$), fold-local feature scaling with `StandardScaler`.
- **Strengths & Limitations:** Very fast training and low memory; cannot capture joint threshold interactions (e.g. high speed AND sharp angle).

### 2. Gaussian Naive Bayes (Generative Probabilistic)
- **How it works:** Applies Bayes' theorem assuming conditional feature independence given class $y$: $P(\mathbf{x} \mid y) = \prod_j \mathcal{N}(x_j \mid \mu_{yj}, \sigma_{yj}^2)$.
- **Why suitable:** Fast probabilistic generative classifier that computes class likelihoods.
- **Assumptions:** Conditional feature independence within each class; normal distribution of features.
- **Overfitting Controls:** `var_smoothing=1e-7` to prevent zero-variance singularities.
- **Strengths & Limitations:** Extremely fast closed-form parameter estimation; assumption severely violated by spatial correlation between `dist_to_arc`, `dist_to_from_node`, and `dist_to_to_node`.

### 3. Decision Tree (Non-Linear Rule Induction)
- **How it works:** Recursively partitions feature space using orthogonal axis splits that maximize Gini impurity reduction.
- **Why suitable:** Captures non-linear thresholds (e.g. `heading_diff > 90` and `dist_to_arc < 10m`) without feature scaling.
- **Assumptions:** Axis-aligned rectangular decision regions.
- **Overfitting Controls:** Constrained tree depth (`max_depth=12`), `min_samples_split=25`, `min_samples_leaf=12`.
- **Strengths & Limitations:** Highly interpretable; high variance and susceptible to noisy GPS jitter without ensembling.

### 4. Random Forest (Bagging Ensemble)
- **How it works:** Builds an ensemble of $B$ decorrelated decision trees using bootstrap aggregation (bagging) and random feature subsets ($\sqrt{D}$) at each split.
- **Why suitable:** Drastically reduces variance of single trees; resilient to label noise and GPS drift.
- **Assumptions:** Aggregated trees have independent errors.
- **Overfitting Controls:** `n_estimators=80`, `max_depth=14`, `min_samples_split=12`, `min_samples_leaf=6`, `class_weight='balanced'`.
- **Strengths & Limitations:** High generalization accuracy; higher inference latency than single linear or tree models.

### 5. Histogram Gradient Boosting (Sequential Boosting)
- **How it works:** Iteratively fits regression trees to negative gradients of binary cross-entropy loss, binning continuous features into 256 integer histograms.
- **Why suitable:** Optimized for large datasets; learns nuanced residual corrections across spatial distances and vehicle speeds.
- **Assumptions:** Additive tree approximation to loss surface.
- **Overfitting Controls:** `l2_regularization=0.5`, `max_leaf_nodes=31`, `min_samples_leaf=25`, `learning_rate=0.1`, `max_iter=80`.
- **Strengths & Limitations:** Outstanding Top-1 accuracy and fast training; requires tuning learning rate and tree depth.
