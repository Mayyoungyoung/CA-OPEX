# TD3+BC experiment aggregate

Statistical unit for cross-run uncertainty: **independent training seed**. Evaluation episodes estimate a fixed trained policy and are never reported as training repetitions.

## Run provenance

| Run | Variant | Train seed | Updates | Status | Train wall (s) |
|---|---:|---:|---:|---|---:|
| hubl_horizon_c002_seed0 | hubl_horizon | 0 | 50000 | complete | 540.4508 |
| hubl_horizon_c004_seed0 | hubl_horizon | 0 | 50000 | complete | 522.3776 |
| hubl_horizon_c008_seed0 | hubl_horizon | 0 | 50000 | complete | 621.6752 |

## Independent evaluations

| Run | Step | Seed | Variant | Pairing | HUBL α | c | Condition | β | Episodes | Return | Normalized score | Status |
|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | clean | 0.0000 | 50 | 3644.8753 | 79.3620 | complete |
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | persistent_action_noise | 1.0000 | 50 | 1664.8759 | 36.2310 | complete |
| hubl_horizon_c004_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0400 | clean | 0.0000 | 50 | 3584.1662 | 78.0395 | complete |
| hubl_horizon_c004_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0400 | persistent_action_noise | 1.0000 | 50 | 1556.5011 | 33.8703 | complete |
| hubl_horizon_c008_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0800 | clean | 0.0000 | 50 | 3668.5432 | 79.8775 | complete |
| hubl_horizon_c008_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0800 | persistent_action_noise | 1.0000 | 50 | 1542.9820 | 33.5758 | complete |

Episode-level returns are not duplicated here. Each row records its source JSON, SHA-256, and JSON pointer in the CSV/JSON outputs.

## Across-training-seed summaries

| Variant | Pairing | Step | Condition | β | n train seeds | Mean normalized score | Sample SD across seeds | Status |
|---|---|---:|---|---:|---:|---:|---:|---|
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 79.3620 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 36.2310 | — | complete |
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 78.0395 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 33.8703 | — | complete |
| hubl_horizon | executed_executed | 50000 | clean | 0.0000 | 1 | 79.8775 | — | complete |
| hubl_horizon | executed_executed | 50000 | persistent_action_noise | 1.0000 | 1 | 33.5758 | — | complete |

## Paired episode comparisons

Differences are target minus the user-specified reference. The bootstrap resamples paired evaluation episodes, not training seeds.

| Target | Reference | Condition | Pairs | Positive | Δ normalized score | 95% paired bootstrap CI | Status |
|---|---|---|---:|---:|---:|---|---|
| hubl_horizon_c002_seed0 | hubl_constant_a055495_seed0 | clean | 50 | 38 | 0.7996 | [0.5082, 1.1010] | complete |
| hubl_horizon_c002_seed0 | hubl_constant_a055495_seed0 | persistent_action_noise | 50 | 31 | 3.1948 | [-4.7369, 10.3889] | complete |
| hubl_horizon_c004_seed0 | hubl_constant_a055495_seed0 | clean | 50 | 21 | -0.5229 | [-0.9380, -0.0843] | complete |
| hubl_horizon_c004_seed0 | hubl_constant_a055495_seed0 | persistent_action_noise | 50 | 21 | 0.8341 | [-6.6847, 8.1780] | complete |
| hubl_horizon_c008_seed0 | hubl_constant_a055495_seed0 | clean | 50 | 41 | 1.3151 | [0.8280, 1.8133] | complete |
| hubl_horizon_c008_seed0 | hubl_constant_a055495_seed0 | persistent_action_noise | 50 | 25 | 0.5396 | [-6.7963, 7.5190] | complete |

## Input hashes

| Role | Path | Status | SHA-256 |
|---|---|---|---|
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\config.json | read | 28d1383aecb8cdcbd96d75339bfd774ca7197583bb47a132c050ce5ed9c05de1 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\summary.json | read | c8a2f3a4538549a7a97052eafd2fcdb3aaa7ea14569876281c46a04f035011f8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\independent_eval_50.json | read | 0ae6e12ad73a3a8859d2f795bebc2f686cc19caa291662fc5783d6e17d2a6099 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\config.json | read | 1dbe93ae9596ddced6c67618777860b0e3c3eb675bebff8017e570db88ddcf01 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\summary.json | read | 95d7c4c7489898c4865e151678901441b4c672c03021e2959a5516b271bfdef6 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c004_seed0\independent_eval_50.json | read | 77c7941d7faa4deabd96de32114a2d2ab7d0ed33709a80c22e40b35f060353a7 |
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\config.json | read | 586dec041b60b2760f24a92297ac509a8a53a28e8ffd014d90d903b85acff9f1 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\summary.json | read | a519c41a5b460d863b279ecd3bdd3bedb7ceccf36c7fa4dbbd8fcb51826ebdf8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c008_seed0\independent_eval_50.json | read | 3fb467f1156c778e74e0f8282ed7042d89984b96af67da983452f3d8842c7c68 |
| reference:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\config.json | read | d7a13459c87355d8001465892194e311ab071e4089a14bb0bc9bf40ba35102bc |
| reference:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\summary.json | read | a9fb824b5248dd32f1ea010c3535d4b0705d5775c2ef0cd550c9cc023b61708c |
| reference:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_constant_a055495_seed0\independent_eval_50.json | read | 9934598255cbf38ca6b4d4484b4d4ee252dd43f3f47805a964bb68272a94066d |
