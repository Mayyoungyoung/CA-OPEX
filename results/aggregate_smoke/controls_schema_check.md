# TD3+BC experiment aggregate

Statistical unit for cross-run uncertainty: **independent training seed**. Evaluation episodes estimate a fixed trained policy and are never reported as training repetitions.

## Run provenance

| Run | Variant | Train seed | Updates | Status | Train wall (s) |
|---|---:|---:|---:|---|---:|
| hubl_horizon_c002_seed0 | hubl_horizon | 0 | 50000 | complete | 540.4508 |

## Independent evaluations

| Run | Step | Seed | Variant | Pairing | HUBL α | c | Cmd scale | Transform | Condition | β | Episodes | Return | Normalized score | Status |
|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | 1.0000 | identity | clean | 0.0000 | 50 | 3644.8753 | 79.3620 | complete |
| hubl_horizon_c002_seed0 | 50000 | 0 | hubl_horizon | executed_executed | 1.0000 | 0.0200 | 1.0000 | identity | persistent_action_noise | 1.0000 | 50 | 1664.8759 | 36.2310 | complete |

Episode-level returns are not duplicated here. Each row records its source JSON, SHA-256, and JSON pointer in the CSV/JSON outputs.

## Across-training-seed summaries

| Variant | Pairing | Cmd scale | Transform | Step | Condition | β | n train seeds | Mean normalized score | Sample SD across seeds | Status |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---|
| hubl_horizon | executed_executed | 1.0000 | identity | 50000 | clean | 0.0000 | 1 | 79.3620 | — | complete |
| hubl_horizon | executed_executed | 1.0000 | identity | 50000 | persistent_action_noise | 1.0000 | 1 | 36.2310 | — | complete |

## Input hashes

| Role | Path | Status | SHA-256 |
|---|---|---|---|
| run:config.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\config.json | read | 28d1383aecb8cdcbd96d75339bfd774ca7197583bb47a132c050ce5ed9c05de1 |
| run:summary.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\summary.json | read | c8a2f3a4538549a7a97052eafd2fcdb3aaa7ea14569876281c46a04f035011f8 |
| run:independent_eval_50.json | F:\Theory\hubl_backup\results\action_noise_dev\hubl_horizon_c002_seed0\independent_eval_50.json | read | 0ae6e12ad73a3a8859d2f795bebc2f686cc19caa291662fc5783d6e17d2a6099 |
