# Independent v2 verification: equal-compute nominal-Q audit

Status: `verified_complete`. Evidence label: `post_confirmation_fresh_rollout_mechanism_audit`.

This report was reconstructed from raw returns without importing the active v1 runner. It is a post-confirmation fresh-rollout mechanism audit, not blind confirmation.

## Launch-frozen provenance

| File | Expected SHA-256 | Observed SHA-256 |
|---|---|---|
| Protocol | `fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034` | `fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034` |
| Runner | `c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1` | `c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1` |

## Independently reconstructed development selection

| eta | raw score | normalized score | env steps | Q1 rows | backward calls |
|---:|---:|---:|---:|---:|---:|
| 0.010000 | 561.905068 | 12.204666 | 2154 | 34464 | 4308 |
| 0.030000 | 603.373240 | 13.107980 | 2275 | 36400 | 4550 |
| 0.100000 | 939.110700 | 20.421452 | 3261 | 52176 | 6522 |
| 0.300000 | 795.281007 | 17.288366 | 2940 | 47040 | 5880 |
| 1.000000 | 389.620348 | 8.451735 | 1794 | 28704 | 3588 |

Initial eta: `0.1`; endpoint extension: `None`; final eta: `0.1`.

## Absolute holdout controller scores and recomputed cost

| Controller | Checkpoint | eta | Baseline norm | Adapted norm | Delta | Adapted steps | Q1 rows | Backward calls | Adapted wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal_tuned | 1 | 0.100000 | 11.090044 | 24.125127 | 13.035083 | 18359 | 293744 | 36718 | 107.471799 |
| nominal_matched_eta_0.3 | 1 | 0.300000 | 11.090044 | 18.098014 | 7.007970 | 14814 | 237024 | 29628 | 86.177711 |
| calibrated_identity_eta_0.3 | 1 | 0.300000 | 11.090044 | 24.464305 | 13.374262 | 19205 | 307280 | 38410 | 144.870769 |
| complete_calibrated_inverse_eta_0.1 | 1 | 0.100000 | 22.582690 | 32.144720 | 9.562030 | 22982 | 367712 | 45964 | 207.915101 |
| nominal_tuned | 10 | 0.100000 | 10.835412 | 18.365012 | 7.529600 | 14351 | 229616 | 28702 | 84.295034 |
| nominal_matched_eta_0.3 | 10 | 0.300000 | 10.835412 | 17.947917 | 7.112505 | 14420 | 230720 | 28840 | 82.343017 |
| calibrated_identity_eta_0.3 | 10 | 0.300000 | 10.835412 | 24.994057 | 14.158645 | 19189 | 307024 | 38378 | 134.539053 |
| complete_calibrated_inverse_eta_0.1 | 10 | 0.100000 | 22.127577 | 31.963722 | 9.836145 | 22913 | 366608 | 45826 | 207.007392 |

All rows match one actor row, 16 Q1 forward/backward rows, and two backward calls per adapted decision. This is equal per-decision neural budget, not exact wall-time/FLOP equality: inverse anchoring adds 48 scalar bisection iterations, and total calls vary with episode horizon.

## Paired holdout comparisons

| Comparison | Checkpoint | mean normalized delta | t 95% | bootstrap 95% | positive |
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

Intervals use paired rollout cases within each fixed checkpoint. The two checkpoint means below are descriptive only; no training-seed interval, p-value, pooling to n=100, or multiple-comparison correction is claimed.

## Cross-checkpoint descriptive comparisons

| Comparison | Per-checkpoint means | Mean | Positive checkpoints |
|---|---|---:|---:|
| complete_minus_nominal_tuned | 8.019593, 13.598710 | 10.809152 | 2/2 |
| complete_minus_nominal_matched_eta_0.3 | 14.046706, 14.015805 | 14.031256 | 2/2 |
| complete_minus_calibrated_identity_eta_0.3 | 7.680414, 6.969665 | 7.325040 | 2/2 |
| calibrated_identity_minus_nominal_tuned | 0.339179, 6.629045 | 3.484112 | 2/2 |
| calibrated_identity_minus_nominal_matched_eta_0.3 | 6.366292, 7.046140 | 6.706216 | 2/2 |

## Complete input hash inventory

| Roles | Bytes | SHA-256 | Path |
|---|---:|---|---|
| launch_frozen_protocol | 8609 | `fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034` | `/root/hubl_research_20260914/code/equal_compute_nominal_control_protocol.json` |
| raw_evaluator_dependency:evaluate_channel_opex.py | 37240 | `cf659f47fddd8f848c84f5f114be1fcb9081a55470fe59e8c073d4f84af0bff1` | `/root/hubl_research_20260914/code/evaluate_channel_opex.py` |
| raw_evaluator_dependency:evaluation_controls.py | 8507 | `24318269a6696fd22c8112b2d96913bdceb830d83de9df31ca234e1505f1aec9` | `/root/hubl_research_20260914/code/evaluation_controls.py` |
| raw_evaluator_dependency:inverse_residual_core.py | 23777 | `5700efa8bbefac91c991a76878c3b78c5b3e84aea83568987f7a7e8919e50041` | `/root/hubl_research_20260914/code/inverse_residual_core.py` |
| canonical_formal_manifest_for_seed_freshness | 167153 | `57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807` | `/root/hubl_research_20260914/code/inverse_residual_frozen_confirmation_manifest.json` |
| launch_frozen_runner_source_not_imported | 23860 | `c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1` | `/root/hubl_research_20260914/code/run_equal_compute_nominal_control.py` |
| raw_evaluator_dependency:td3bc_core.py | 31956 | `e27cec131b533dd542e77ba9de2de492db4890c6fb9723796f59a02002d1f052` | `/root/hubl_research_20260914/code/td3bc_core.py` |
| raw_evaluator_dependency:train_inverse_residual_adapter.py | 57182 | `fb065682c78e6c058d1165fa68f680fb682d235ec8c9c3abe70b8098177ee0e4` | `/root/hubl_research_20260914/code/train_inverse_residual_adapter.py` |
| raw_evaluator_dependency:train_td3bc.py | 27912 | `28094abb38441633f3e52386a9d3ac0747b9df8a4f2bd03403bf46d1d5234737` | `/root/hubl_research_20260914/code/train_td3bc.py` |
| independent_verifier_source | 71334 | `053d3d823bf23475385b956426f67d78c213bd0cc74b482cf995d7c9ddf2da90` | `/root/hubl_research_20260914/code/verify_equal_compute_nominal_control.py` |
| channel_calibration | 28649 | `b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354` | `/root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001_calibration.json` |
| base_checkpoint:seed0 | 3491870 | `d29501edf9b6961225bcbc157d7d7034bc5b625856dedcbbf9421eea25898106` | `/root/hubl_research_20260914/results/channel_dev/hubl_constant_executed_beta05_25k_seed0/latest.pt` |
| runner_v1_aggregate_crosscheck | 10321 | `4d0e949706cf97ce900565b815e0bdd4dcb8026cc1262c2c68e3fa6e70082325` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/aggregate/equal_compute_nominal_control.json` |
| development_raw:nominal_dev:seed0 | 9065 | `d5449dca0f42404cd85bcb1acfb1562337df119e783fd1d883004d7844271677` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/development/nominal_q_identity_eta_0p01_seed0_dev10.json` |
| development_raw:nominal_dev:seed0 | 9051 | `28596d93027974c059d30cad5a4b0b1e526318cf9878caa71b03bb146f6c3056` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/development/nominal_q_identity_eta_0p03_seed0_dev10.json` |
| development_raw:nominal_dev:seed0 | 9057 | `0dd844bc5eed43fd5c9b6656e9f5b7b1200f2fff8d3bb9a5013b8fefa93afe59` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/development/nominal_q_identity_eta_0p1_seed0_dev10.json` |
| development_raw:nominal_dev:seed0 | 9050 | `576f63c174a7e309f40f2691f15429b716194c836a8d1644527523eeb09afe9e` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/development/nominal_q_identity_eta_0p3_seed0_dev10.json` |
| development_raw:nominal_dev:seed0 | 9055 | `dac33f19dc2b1f43f5f51d2effcc1423f62b9638e601ede10ea1cf4b5aa16637` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/development/nominal_q_identity_eta_1_seed0_dev10.json` |
| holdout_raw:calibrated_identity_eta_0.3:seed10 | 19841 | `2dd7a42d4b54f76f09e7b22856d8c2a663524c653020016977ea569c2f9e05cd` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/calibrated_identity_eta_0.3_seed10_fresh50.json` |
| holdout_raw:calibrated_identity_eta_0.3:seed1 | 19846 | `5a123d8cadffb7b47c77069482262ab82b9e7bbaded1da8b329ba4559247e8fe` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/calibrated_identity_eta_0.3_seed1_fresh50.json` |
| holdout_raw:complete_calibrated_inverse_eta_0.1:seed10 | 19903 | `553b7ff3bfca058be2f15684222fa4a2a101357c5f8e04e503596089fbae3dd6` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/complete_calibrated_inverse_eta_0.1_seed10_fresh50.json` |
| holdout_raw:complete_calibrated_inverse_eta_0.1:seed1 | 19912 | `a962af60ee972428b67ac4ae91fdb619399ab04394aeebf37804570c12670cdc` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/complete_calibrated_inverse_eta_0.1_seed1_fresh50.json` |
| holdout_raw:nominal_matched_eta_0.3:seed10 | 17896 | `6f10f56aeda12c847bc6ff0d934cba8e4aaa5b02f0f5bf08d92d004e5860b1ed` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/nominal_matched_eta_0.3_seed10_fresh50.json` |
| holdout_raw:nominal_matched_eta_0.3:seed1 | 17902 | `7e733cb667c6ae80f85a7353fcf5b28ad048a1de0b5107186795a08d1788a4e5` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/nominal_matched_eta_0.3_seed1_fresh50.json` |
| holdout_raw:nominal_tuned:seed10 | 17917 | `4f35cedf07cd7af5ebdbcb4bdd7d8a69a6eab46fb77b0a62ea50987fe914a1b3` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/nominal_tuned_seed10_fresh50.json` |
| holdout_raw:nominal_tuned:seed1 | 17929 | `8ababfa7389b67ebfd0fb4827e799880b97e71be2e08b7c15b1811018f299f40` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/nominal_tuned_seed1_fresh50.json` |
| runner_v1_selection_crosscheck | 2262 | `71af36661fbaa5d2340fbb23897abacc7398576343f742f66bd3ed31c5677ae4` | `/root/hubl_research_20260914/results/equal_compute_nominal_control/selection.json` |
| base_checkpoint:seed1 | 3492510 | `22b47965695bfa312533741abaa1e17e3cbbc091f3c5e6ba7fc8953fbc8ca6ae` | `/root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt` |
| base_checkpoint:seed10 | 3492510 | `eba1af040aa73823ceba28048e10d3af2eef54c59f09c1e0194f27aa0ed4944d` | `/root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt` |
