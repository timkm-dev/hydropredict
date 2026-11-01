# CARE model-family comparison for mentor review

29 October 2026; related issue #21. User requested a broader set of fitted models.
Freeze this plan before inspecting new model results. Development evidence only.

Already fitted on CARE: Isolation Forest normality detection, Extra Trees
temperature regression with residual alarms, and histogram gradient boosting
classification of residual trajectories. The latter two use tree ensembles but
solve different learning tasks; this is not yet a broad family comparison.

New candidates to fit, one fixed configuration each:

| Model | Reason | Configuration |
|---|---|---|
| Logistic regression | Regularized linear reference; tests whether complexity hurts | StandardScaler, L2, C=0.1, lbfgs, max_iter=2000 |
| Gaussian naive Bayes | Simple probabilistic/generative alternative | StandardScaler, var_smoothing=0.01 |
| Random forest classifier | Bagged nonlinear interactions, unlike sequential boosting | 100 trees, depth=10, leaf minimum=20, max_features=sqrt, two workers, seed=42 |
| Approximate RBF-kernel SVM | Smooth nonlinear decision boundaries outside tree families | StandardScaler, Nystroem RBF 128 components, gamma=1/feature_count, LinearSVC C=0.1, dual=False, max_iter=5000, seed=42 |

The kernel model is explicitly an approximate RBF SVM, not exact kernel SVC.
Its decision margin is mapped through logistic sigmoid only to use the common
threshold grid; neither it nor the other outputs are claimed to be calibrated
physical-failure probabilities. Normal-only threshold calibration handles scores.

All four reuse the same frozen v3 causal residual trajectories, cached feature
values, per-file sampled rows, equal file/class weights, 14 outer held-turbine
folds, and 46 inner excluded-pair fits. Fit scaling only on the relevant fitting
turbines, with fitting sample weights. No tuning on outer outcomes. All 37 A/B
files retained; C remains performance-uninspected. No automatic model promotion.

Use exactly the v3 normal-control calibration grid and six-reading persistence.
Keep the same gates: >=50% early anomaly detections, <=1 normal episode/30 observed
days, <=1% alerted normal rows, >=80% valid rows per file, separately by farm.
Early means >=24h before annotated interval end, not verified physical failure.
Vectorized alarm bookkeeping must match prior logic on randomized cases and saved
v3 outputs before comparison. This optimization changes speed, not alert semantics.

Save every case, fitted outer models, training/held-out file-weighted sampled-row
AUC (undefined for one-class held-out turbines), thresholds, exclusions, warnings,
source/cache/code/protocol hashes and runtime. No post-outcome hyperparameter search.
If a model cannot fit, record the failure rather than silently substituting another.
Do not choose a winner based on training accuracy. New comparisons use identical
features/splits; earlier unsupervised approaches are contextual references, not
controlled comparisons of classifier architecture alone.

Deferred candidates, NOT represented as tested: autoencoders, LSTM/GRU, temporal
convolutional networks, exact RBF SVC, and survival/time-to-failure models. Neural
models need a justified training/validation design with few independent fault
events; survival models need validated censoring/failure endpoints. They remain
mentor discussion options, not evidence that all modelling possibilities are exhausted.
