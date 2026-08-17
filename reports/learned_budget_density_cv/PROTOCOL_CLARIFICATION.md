# Experiment 15 Protocol Clarification

Recorded before any Experiment 15 model checkpoint or neural OOF prediction.
This resolves aggregation details left implicit in the preregistration; it does
not change a candidate, feature, threshold, fold, or decision rule.

- Budget log-MAE is computed within each source over its frames and then
  averaged over the 35 sources, so the twice-regenerated sources do not receive
  double weight.
- Budget Spearman correlation and prediction/target standard-deviation ratio
  are computed from the 35 source-mean predictions and targets.
- Each fold's median calibration residual gives every training source equal
  total weight across its frames.
- Mask metrics are averaged within source and then across sources. Bootstrap
  resampling and win counts operate on the 35 source-level paired differences.
- If OOF selection authorizes final training, its seed is the registered base
  seed plus 1,000 (`20261840`). It uses all development samples for exactly the
  same eight epochs and derives one source-balanced median residual afterward.
- Test-generation fields copied into the machine-readable config are exact
  operational duplicates of the already frozen Experiment-14 sealed protocol:
  30 unchanged sources, 16 frames, probe seed 20260823, SAM-mask seed 20260833,
  5%/10% jitter, 5% box expansion, and single-component donors. They add no new
  selection freedom and are recorded before model lock or test generation.
