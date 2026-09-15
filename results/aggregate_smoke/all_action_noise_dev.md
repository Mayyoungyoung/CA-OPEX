# TD3+BC experiment aggregate

Statistical unit for cross-run uncertainty: **independent training seed**. Evaluation episodes estimate a fixed trained policy and are never reported as training repetitions.

## Run provenance

| Run | Variant | Train seed | Updates | Status | Train wall (s) |
|---|---:|---:|---:|---|---:|
| hubl_constant_a039184_seed0 | hubl_constant | 0 | 50000 | complete | 630.3289 |
| hubl_constant_a055495_seed0 | hubl_constant | 0 | 50000 | complete | 511.3643 |
| hubl_constant_seed0 | — | — | — | incomplete_provenance | — |
| hubl_horizon_c002_seed0 | hubl_horizon | 0 | 50000 | complete | 540.4508 |
| hubl_horizon_c004_seed0 | hubl_horizon | 0 | 50000 | complete | 522.3776 |
| hubl_horizon_c008_seed0 | hubl_horizon | 0 | 50000 | complete | 621.6752 |
| hubl_rank_seed0 | hubl_rank | 0 | 100000 | complete | 1078.6832 |
| td3bc_seed0 | — | — | — | incomplete_provenance | — |

## Independent evaluations

| Run | Step | Seed | Variant | Pairing | HUBL α | c | Condition | β | Episodes | Return | Normalized score | Status |
|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
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
| td3bc_seed0 | 100000 | — | — | — | — | — | clean | 0.0000 | 50 | 3981.5094 | 86.6950 | incomplete_provenance |
| td3bc_seed0 | 100000 | — | — | — | — | — | persistent_action_noise | 1.0000 | 50 | 697.6096 | 15.1608 | incomplete_provenance |

Episode-level returns are not duplicated here. Each row records its source JSON, SHA-256, and JSON pointer in the CSV/JSON outputs.

## Across-training-seed summaries

| Variant | Pairing | Step | Condition | β | n train seeds | Mean normalized score | Sample SD across seeds | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
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
| — | — | 100000 | clean | 0.0000 | 0 | — | — | invalid_missing_train_seed |
| — | — | 100000 | persistent_action_noise | 1.0000 | 0 | — | — | invalid_missing_train_seed |

## Input hashes

| Role | Path | Status | SHA-256 |
|---|---|---|---|
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
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\config.json | missing | — |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\summary.json | missing | — |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\td3bc_seed0\independent_eval_50.json | read | 87b5207be0f3bd3ec5a44e254c873bda2a44a2692147bbbc2d992c236f5dc4b4 |
