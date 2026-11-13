# Warranty model results

Test units: 600; training units: 2400.
Random Forest MAE: $558.16; RMSE: $866.77; R²: 0.9205.
95% unit-bootstrap MAE interval: $509.73 to $609.54.
This interval conditions on the fitted model and does not measure retraining uncertainty.

MAE improvement over mean baseline: 78.56%.
MAE improvement over median baseline: 74.01%.
Negative improvement means the model is worse than that baseline.

Expensive claims use the training 90th percentile ($7,700.00); 64 test units qualify.
Mean actual-minus-predicted cost for that group: 1249.9935053571428.

These are recorded costs for represented units, with no confirmed observation horizon.
Random splits assess similar units, not future-production performance. No-claim units are not validated.
Permutation importance is predictive association, not causal evidence. Negative values are possible.
Use predictions to prioritize review; avoided costs require a separately evaluated intervention.
Keep the held-out set fixed and do not tune further using its diagnostics.
