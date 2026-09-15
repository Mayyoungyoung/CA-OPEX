# TD3+BC experiment aggregate

Statistical unit for cross-run uncertainty: **independent training seed**. Evaluation episodes estimate a fixed trained policy and are never reported as training repetitions.

## Run provenance

| Run | Variant | Train seed | Updates | Status | Train wall (s) |
|---|---:|---:|---:|---|---:|
| hubl_constant_a039184_commanded_seed0 | hubl_constant | 0 | 50000 | complete | 520.1103 |
| hubl_constant_a039184_dual_seed0 | hubl_constant | 0 | 50000 | complete | 538.9911 |
| hubl_constant_a039184_seed0 | hubl_constant | 0 | 50000 | complete | 630.3289 |
| hubl_constant_a055495_seed0 | hubl_constant | 0 | 50000 | complete | 511.3643 |
| hubl_constant_seed0 | — | — | — | incomplete_provenance | — |
| hubl_horizon_c002_seed0 | hubl_horizon | 0 | 50000 | complete | 540.4508 |
| hubl_horizon_c004_seed0 | hubl_horizon | 0 | 50000 | complete | 522.3776 |
| hubl_horizon_c008_seed0 | hubl_horizon | 0 | 50000 | complete | 621.6752 |
| hubl_rank_seed0 | hubl_rank | 0 | 100000 | complete | 1078.6832 |
| td3bc_commanded_seed0 | td3bc | 0 | 50000 | complete | 491.8075 |
| td3bc_dual_seed0 | td3bc | 0 | 50000 | complete | 527.7632 |
| td3bc_seed0 | — | — | — | incomplete_provenance | — |

## Independent evaluations

| Run | Step | Seed | Variant | Pairing | HUBL α | c | Condition | β | Episodes | Return | Normalized score | Status |
|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
| hubl_constant_a039184_commanded_seed0 | 50000 | 0 | hubl_constant | commanded_commanded | 0.3918 | — | clean | 0.0000 | 50 | 3401.6248 | 74.0632 | complete |
| hubl_constant_a039184_commanded_seed0 | 50000 | 0 | hubl_constant | commanded_commanded | 0.3918 | — | persistent_action_noise | 1.0000 | 50 | 1630.4084 | 35.4802 | complete |
| hubl_constant_a039184_dual_seed0 | 50000 | 0 | hubl_constant | executed_commanded | 0.3918 | — | clean | 0.0000 | 50 | 3527.2148 | 76.7989 | complete |
| hubl_constant_a039184_dual_seed0 | 50000 | 0 | hubl_constant | executed_commanded | 0.3918 | — | persistent_action_noise | 1.0000 | 50 | 1425.8428 | 31.0241 | complete |
| hubl_constant_a039184_seed0 | 50000 | 0 | hubl_constant | executed_executed | 0.3918 | — | clean | 0.0000 | 50 | 3647.1968 | 79.4125 | complete |
| hubl_constant_a039184_seed0 | 50000 | 0 | hubl_constant | executed_executed | 0.3918 | — | persistent_action_noise | 1.0000 | 50 | 1894.2731 | 41.2280 | complete |
| hubl_constant_a055495_seed0 | 50000 | 0 | hubl_constant | executed_executed | 0.5550 | — | clean | 0.0000 | 50 | 3608.1704 | 78.5624 | complete |
| hubl_constant_a055495_seed0 | 50000 | 0 | hubl_constant | executed_executed | 0.5550 | — | persistent_action_noise | 1.0000 | 50 | 1518.2108 | 33.0362 | complete |
| hubl_constant_seed0 | 100000 | — | — | — | — | — | clean | 0.0000 | 50 | 3501.0135 | 76.2282 | incomplete_provenance |
| hubl_constant_seed0 | 100000 | — | — | — | — | — | persistent_action_noise | 1.0000 | 50 | 1649.1417 | 35.8883 | incomplete_provenance |
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | clean | 0.0000 | 50 | 3644.8753 | 79.3620 | complete |
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | persistent_action_noise | 1.0000 | 50 | 1664.8759 | 36.2310 | complete |
| hubl_horizon_c004_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0400 | clean | 0.0000 | 50 | 3584.1662 | 78.0395 | complete |
| hubl_horizon_c004_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0400 | persistent_action_noise | 1.0000 | 50 | 1556.5011 | 33.8703 | complete |
| hubl_horizon_c008_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0800 | clean | 0.0000 | 50 | 3668.5432 | 79.8775 | complete |
| hubl_horizon_c008_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0800 | persistent_action_noise | 1.0000 | 50 | 1542.9820 | 33.5758 | complete |
| hubl_rank_seed0 | 100000 | 0 | hubl_rank | executed_executed | 1.0000 | — | clean | 0.0000 | 50 | 3567.4101 | 77.6745 | complete |
| hubl_rank_seed0 | 100000 | 0 | hubl_rank | executed_executed | 1.0000 | — | persistent_action_noise | 1.0000 | 50 | 1352.3711 | 29.4236 | complete |
| td3bc_commanded_seed0 | 50000 | 0 | td3bc | commanded_commanded | 1.0000 | — | clean | 0.0000 | 50 | 3464.5826 | 75.4346 | complete |
| td3bc_commanded_seed0 | 50000 | 0 | td3bc | commanded_commanded | 1.0000 | — | persistent_action_noise | 1.0000 | 50 | 1129.2655 | 24.5637 | complete |
| td3bc_dual_seed0 | 50000 | 0 | td3bc | executed_commanded | 1.0000 | — | clean | 0.0000 | 50 | 3542.4869 | 77.1316 | complete |
| td3bc_dual_seed0 | 50000 | 0 | td3bc | executed_commanded | 1.0000 | — | persistent_action_noise | 1.0000 | 50 | 991.6497 | 21.5659 | complete |
| td3bc_seed0 | 100000 | — | — | — | — | — | clean | 0.0000 | 50 | 3981.5094 | 86.6950 | incomplete_provenance |
| td3bc_seed0 | 100000 | — | — | — | — | — | persistent_action_noise | 1.0000 | 50 | 697.6096 | 15.1608 | incomplete_provenance |

Episode-level returns are not duplicated here. Each row records its source JSON, SHA-256, and JSON pointer in the CSV/JSON outputs.

## Across-training-seed summaries

| Variant | Pairing | Step | Condition | β | n train seeds | Mean normalized score | Sample SD across seeds | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
| hubl_constant | commanded_commanded | 50000 | clean | 0.0000 | 1 | 74.0632 | — | complete |
| hubl_constant | commanded_commanded | 50000 | persistent_action_noise | 1.0000 | 1 | 35.4802 | — | complete |
| hubl_constant | executed_commanded | 50000 | clean | 0.0000 | 1 | 76.7989 | — | complete |
| hubl_constant | executed_commanded | 50000 | persistent_action_noise | 1.0000 | 1 | 31.0241 | — | complete |
| hubl_constant | executed_executed | 50000 | clean | 0.0000 | 1 | 79.4125 | — | complete |
| hubl_constant | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 41.2280 | — | complete |
| hubl_constant | executed_executed | 50000 | clean | 0.0000 | 1 | 78.5624 | — | complete |
| hubl_constant | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 33.0362 | — | complete |
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 79.3620 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 36.2310 | — | complete |
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 78.0395 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 33.8703 | — | complete |
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 79.8775 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 33.5758 | — | complete |
| hubl_rank | executed_executed | 100000 | clean | 0.0000 | 1 | 77.6745 | — | complete |
| hubl_rank | executed_executed | 100000 | persistent_action_noise | 1.0000 | 1 | 29.4236 | — | complete |
| td3bc | commanded_commanded | 50000 | clean | 0.0000 | 1 | 75.4346 | — | complete |
| td3bc | commanded_commanded | 50000 | persistent_action_noise | 1.0000 | 1 | 24.5637 | — | complete |
| td3bc | executed_commanded | 50000 | clean | 0.0000 | 1 | 77.1316 | — | complete |
| td3bc | executed_commanded | 50000 | persistent_action_noise | 1.0000 | 1 | 21.5659 | — | complete |
| — | — | 100000 | clean | 0.0000 | 0 | — | — | invalid_missing_train_seed |
| — | — | 100000 | persistent_action_noise | 1.0000 | 0 | — | — | invalid_missing_train_seed |

## Paired episode comparisons

Differences are target minus the user-specified reference. The bootstrap resamples paired evaluation episodes, not training seeds.

| Target | Reference | Condition | Pairs | Positive | Δ normalized score | 95% paired bootstrap CI | Status |
|---|---|---|---:|---:|---:|---|---|
| hubl_constant_a039184_commanded_seed0 | td3bc_dual_seed0 | clean | 50 | 14 | -3.0684 | [-6.7597, 0.8051] | complete |
| hubl_constant_a039184_commanded_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 41 | 13.9143 | [9.3618, 18.6922] | complete |
| hubl_constant_a039184_dual_seed0 | td3bc_dual_seed0 | clean | 50 | 15 | -0.3327 | [-3.7738, 3.3267] | complete |
| hubl_constant_a039184_dual_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 38 | 9.4582 | [4.4309, 14.4567] | complete |
| hubl_constant_a039184_seed0 | td3bc_dual_seed0 | clean | 50 | 15 | 2.2809 | [-0.3325, 5.4650] | complete |
| hubl_constant_a039184_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 41 | 19.6621 | [13.4605, 25.9503] | complete |
| hubl_constant_a055495_seed0 | td3bc_dual_seed0 | clean | 50 | 10 | 1.4308 | [-1.2122, 4.6568] | complete |
| hubl_constant_a055495_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 32 | 11.4702 | [5.5574, 17.7614] | complete |
| hubl_constant_seed0 | td3bc_dual_seed0 | clean | 50 | 7 | — | — | paired_values_valid_but_provenance_incomplete |
| hubl_constant_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 35 | — | — | paired_values_valid_but_provenance_incomplete |
| hubl_horizon_c002_seed0 | td3bc_dual_seed0 | clean | 50 | 13 | 2.2304 | [-0.3967, 5.4251] | complete |
| hubl_horizon_c002_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 38 | 14.6651 | [8.7429, 20.6050] | complete |
| hubl_horizon_c004_seed0 | td3bc_dual_seed0 | clean | 50 | 7 | 0.9079 | [-1.6817, 4.0655] | complete |
| hubl_horizon_c004_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 34 | 12.3043 | [6.7171, 17.9516] | complete |
| hubl_horizon_c008_seed0 | td3bc_dual_seed0 | clean | 50 | 24 | 2.7459 | [0.0622, 5.9685] | complete |
| hubl_horizon_c008_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 37 | 12.0098 | [6.8479, 17.3822] | complete |
| hubl_rank_seed0 | td3bc_dual_seed0 | clean | 50 | 8 | 0.5429 | [-2.2504, 3.7763] | complete |
| hubl_rank_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 29 | 7.8577 | [2.4517, 13.5152] | complete |
| td3bc_commanded_seed0 | td3bc_dual_seed0 | clean | 50 | 9 | -1.6970 | [-4.9442, 1.7516] | complete |
| td3bc_commanded_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 27 | 2.9977 | [-1.0476, 7.1571] | complete |
| td3bc_seed0 | td3bc_dual_seed0 | clean | 50 | 49 | — | — | paired_values_valid_but_provenance_incomplete |
| td3bc_seed0 | td3bc_dual_seed0 | persistent_action_noise | 50 | 13 | — | — | paired_values_valid_but_provenance_incomplete |

## Input hashes

| Role | Path | Status | SHA-256 |
|---|---|---|---|
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_commanded_seed0\config.json | read | 6e616adedd07ffd616602d7e3b523007ffa93cc123eac9d0daa121f29e71be70 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_commanded_seed0\summary.json | read | eade78f3c22f70a1a003d71247075a9b96bb1390515b0826baf977dc0a2ab5cd |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_commanded_seed0\independent_eval_50.json | read | 02f2f113029933492db976e896b832d4c37caf622f5e55db8bfbe1d1120e99ec |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_dual_seed0\config.json | read | 74b833f5e543484ae898811b2d086c83fb0bc4e2f4440bcc19218a5359e72029 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_dual_seed0\summary.json | read | cad3b9f84f86b15eabbcab0facce0731be1c9de9e530877edf95ace29acae9e0 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_dual_seed0\independent_eval_50.json | read | 26cfc2d4d0a1844d126629d0094dfd9f98f9f16733d209402ae03cdf838d3f84 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_seed0\config.json | read | 89c446db79dca06a8a67f4a9310bff2c89632baa7981e1e8300d9fa64a4fa07a |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_seed0\summary.json | read | 75853724eb1b2adadfdd54035edce88ac3ff47dad4e4093a65b13f3f0e564724 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a039184_seed0\independent_eval_50.json | read | 4185657f24ab07917944b4507bbe91f6300f2fa8e0528c0287d7db0d5158cca6 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\config.json | read | d7a13459c87355d8001465892194e311ab071e4089a14bb0bc9bf40ba35102bc |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\summary.json | read | a9fb824b5248dd32f1ea010c3535d4b0705d5775c2ef0cd550c9cc023b61708c |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\independent_eval_50.json | read | 9934598255cbf38ca6b4d4484b4d4ee252dd43f3f47805a964bb68272a94066d |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_seed0\config.json | missing | — |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_seed0\summary.json | missing | — |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_seed0\independent_eval_50.json | read | 3989ee454c656aae28b8d1a9f10ed652333ff1f557ef606b60660b113baec729 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\config.json | read | 28d1383aecb8cdcbd96d75339bfd774ca7197583bb47a132c050ce5ed9c05de1 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\summary.json | read | c8a2f3a4538549a7a97052eafd2fcdb3aaa7ea14569876281c46a04f035011f8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\independent_eval_50.json | read | 0ae6e12ad73a3a8859d2f795bebc2f686cc19caa291662fc5783d6e17d2a6099 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\config.json | read | 1dbe93ae9596ddced6c67618777860b0e3c3eb675bebff8017e570db88ddcf01 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\summary.json | read | 95d7c4c7489898c4865e151678901441b4c672c03021e2959a5516b271bfdef6 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\independent_eval_50.json | read | 77c7941d7faa4deabd96de32114a2d2ab7d0ed33709a80c22e40b35f060353a7 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\config.json | read | 586dec041b60b2760f24a92297ac509a8a53a28e8ffd014d90d903b85acff9f1 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\summary.json | read | a519c41a5b460d863b279ecd3bdd3bedb7ceccf36c7fa4dbbd8fcb51826ebdf8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\independent_eval_50.json | read | 3fb467f1156c778e74e0f8282ed7042d89984b96af67da983452f3d8842c7c68 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_rank_seed0\config.json | read | 01fda6a1ae242183c4fb82d7c00a97e7fa04e4a51f6be10213db7f9cd161c8f4 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_rank_seed0\summary.json | read | 0b36415eb564e84defe1f23b0dbb2d2276b6d2384dc1c74cf110dff7faeba462 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_rank_seed0\independent_eval_50.json | read | f81878150fe02e30f274aee9918a3a9fc76d551f837e3d2372c2076fd42f448f |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_commanded_seed0\config.json | read | 6380a4de81b66a762ced9bd1bed3183a6a82a0756ffff5d4de482afaa25582cd |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_commanded_seed0\summary.json | read | cd45214de3ec1037efb65b6f45a31b59ff96af26591f97142442c962111d26e8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_commanded_seed0\independent_eval_50.json | read | 5c9ed31beea8e802debb087bdad02c9f0cb9d71185bc5ca67d5e4119d85151d9 |
| reference:config.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_dual_seed0\config.json | read | ca39d67016aa257ae5e86951ed1c29c8d8c771a4ad05acbd955e244516747853 |
| reference:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_dual_seed0\summary.json | read | 7f462d937a1f51e72becd981298efb3b29bc9f89e436f44711db1e13296d4902 |
| reference:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_dual_seed0\independent_eval_50.json | read | 3b35401f01f4345adce5d8fdd3e47c41aeea2d1994d72d67cb47b6230fa992d4 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\config.json | missing | — |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\summary.json | missing | — |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\independent_eval_50.json | read | 87b5207be0f3bd3ec5a44e254c873bda2a44a2692147bbbc2d992c236f5dc4b4 |
