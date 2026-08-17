# Experiment 15 Preregistration: Geometry-Anchored Budget Density

Frozen on 2026-08-15 at 14:52:13 KST, before any Experiment 15 feature cache,
model checkpoint, cross-validation prediction, or result existed. At freeze
time, Experiment 14's `model_lock.json`, sealed-test data directory, and test
feature manifest were all absent.

## Reason for the new experiment ID

Experiment 14 is closed as a validation failure and will not be retuned. Its
hybrid learned/distance spatial selector improved validation recovery, but the
pooled scalar budget head missed the registered error limit. Experiment 15 is
a new, finite learnability test for that isolated bottleneck.

The main design diagnosis is observable without GT: Experiment 14 retained
SAM2 ranks but discarded absolute temporal logits. Rank normalization preserves
where evidence is strongest while making each candidate-band score distribution
nearly uniform. This is suitable for pixel ordering but may erase information
about how much support is missing. Experiment 15 adds sigmoid-transformed
forward/backward SAM2 probabilities while retaining the rank channels.

## Data isolation

- Development pool: the 25 Experiment-14 training sources and 10 former
  validation sources, now all treated as development data.
- Five fixed source-group folds: each contains five former-train and two
  former-validation sources. Every OOF source is excluded from its model's
  training, epoch selection, and calibration.
- Fixed training length: eight epochs; held-out labels cannot early-stop a run.
- Sealed test: the same 30 never-used sources fixed for Experiment 14. Their
  synthetic data and features remain ungenerated until a final model lock.
- Existing Experiment 10--13 results remain forbidden for model selection.

The exact source assignments and deterministic hash rule are stored in
`configs/learned_budget_density_cv_splits.json`.

## Frozen candidates

1. `scalar_rank`: the original nine rank/geometry channels and pooled scalar
   regression, serving as the Experiment-14-style baseline.
2. `scalar_absolute`: the same scalar head with four absolute SAM2 probability
   channels added.
3. `ordinal_absolute`: nine fixed log-ratio bins plus expected-log regression,
   using all 13 channels.
4. `density_absolute`: a calibrated per-pixel residual-density head. Its
   probability mass divided by the radius-5 base area produces the budget.
5. `density_absolute_temporal`: the same density checkpoint with a fixed
   `[0.25, 0.50, 0.25]` log-budget temporal filter.

All models retain the same support-ranking head. Their full spatial method
rank-normalizes the learned score and distance score, mixes them 1:1 as fixed
by the prior experiment, and applies the same connectivity-aware exact-budget
frontier decoder.

For every candidate and fold, one log-budget bias is the median training-fold
residual and is applied to that fold's held-out predictions. No OOF target is
used for calibration. The temporal candidate is smoothed before its training-
fold bias is computed.

## Staged OOF gate

Budget evaluation uses all 35 source-group OOF predictions. A candidate must
have mean absolute log-ratio error `<= 0.45`, Spearman correlation `>= 0.30`,
and source-mean prediction standard deviation at least 25% of the target's.
Candidates failing this gate do not receive expensive full-resolution mask
evaluation.

Candidates passing the budget gate must then, on the same source-group OOF
predictions and exact predicted area:

1. beat distance recovery by at least `+0.05`;
2. have a positive source-bootstrap CI lower bound;
3. win at least 21/35 sources and have higher added precision;
4. beat frozen SAM2-mean recovery by at least `+0.015`;
5. have a positive CI lower bound and at least 21/35 wins versus SAM2 mean.

Among candidates passing every check, select maximum recovery delta versus
distance, then lower budget MAE, then earlier declared candidate order. If none
passes, stop Experiment 15 without final training or test generation.

## Conditional sealed test and downstream

Only an OOF PASS permits training the selected architecture on all 35
development sources for the same fixed eight epochs. Its checkpoint, one
development-derived median bias, feature definition, temporal rule, and code
hashes must be locked before generating any test artifact.

The 30-source mask gate remains exactly the Experiment-14 gate. ProPainter and
E2FGVI-HQ are run only after that sealed mask gate passes. A validation or OOF
gain alone is not a method claim.
