# Equal-compute nominal-Q mechanism audit

Evidence label: `post_confirmation_fresh_rollout_mechanism_audit`. Formal results were seen before this fresh-seed audit; this is not blind confirmation.

| Comparison | Checkpoint | mean normalized delta | t95 | bootstrap95 | positive |
|---|---:|---:|---:|---:|---:|
| complete_minus_nominal_tuned | 1 | 8.019593 | [1.863205, 14.175981] | [2.011864, 13.962748] | 33/50 |
| complete_minus_nominal_tuned | 10 | 13.598710 | [6.919526, 20.277894] | [7.141191, 20.196525] | 34/50 |
| complete_minus_nominal_matched_eta_0.3 | 1 | 14.046706 | [9.062706, 19.030706] | [9.331686, 19.016505] | 41/50 |
| complete_minus_nominal_matched_eta_0.3 | 10 | 14.015805 | [7.732931, 20.298680] | [8.109989, 20.218809] | 36/50 |
| complete_minus_calibrated_identity_eta_0.3 | 1 | 7.680414 | [2.684510, 12.676318] | [2.877539, 12.515176] | 33/50 |
| complete_minus_calibrated_identity_eta_0.3 | 10 | 6.969665 | [-0.333054, 14.272384] | [-0.017140, 13.926845] | 30/50 |
| calibrated_identity_minus_nominal_tuned | 1 | 0.339179 | [-4.911893, 5.590250] | [-4.757674, 5.400341] | 31/50 |
| calibrated_identity_minus_nominal_tuned | 10 | 6.629045 | [1.240275, 12.017816] | [1.465839, 11.835640] | 35/50 |
| calibrated_identity_minus_nominal_matched_eta_0.3 | 1 | 6.366292 | [1.980787, 10.751796] | [2.149744, 10.611991] | 39/50 |
| calibrated_identity_minus_nominal_matched_eta_0.3 | 10 | 7.046140 | [1.539420, 12.552860] | [1.768266, 12.426237] | 34/50 |

Cross-checkpoint values below are descriptive means of two checkpoint-level means only; no training-seed interval or p-value is claimed.

- `complete_minus_nominal_tuned`: 10.809152
- `complete_minus_nominal_matched_eta_0.3`: 14.031256
- `complete_minus_calibrated_identity_eta_0.3`: 7.325040
- `calibrated_identity_minus_nominal_tuned`: 3.484112
- `calibrated_identity_minus_nominal_matched_eta_0.3`: 6.706216
