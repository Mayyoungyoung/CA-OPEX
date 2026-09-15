# CA-OPEX: channel-calibrated test-time offline policy extraction

The primary deliverable in this directory is CA-OPEX, a frozen-policy
deployment adapter for a calibrated stochastic command-to-action channel.  Start
with `FORMAL_CONFIRMATION_REPRO.md`,
`inverse_residual_frozen_confirmation_manifest.json`,
`run_frozen_channel_opex_confirmation.sh`, and
`aggregate_inverse_residual_results.py`; then see
`verify_equal_compute_nominal_control.py` and
`results/equal_compute_nominal_control/aggregate_v2/` for the supplemental
equal-neural-budget audit.  The later sections retain the
clean-room HUBL base learner and earlier mechanism branches so their data and
negative results remain auditable; they are not the final contribution.

## Preserved reward-noise mechanism audit

This directory contains a clean-room mechanism audit for testing whether a
trajectory score is robust to stochastic rewards.  It does not import or copy
the HUBL supplementary implementation.

`mechanism_audit.py` reads the standard flat D4RL HDF5 fields
`observations`, `actions`, `rewards`, `terminals`, and `timeouts`, reconstructs
complete trajectories, and injects seeded additive reward noise.  A small
heteroscedastic MLP predicts the time-zero return-to-go from reward-free
trajectory descriptors.  Every prediction is strictly out-of-fold at the
trajectory level; normalization is also trained only on the non-held-out
folds.  Within each outer training split, a disjoint calibration subset learns
a single uncertainty scale; it never sees the outer test trajectories or the
latent clean returns.  This guards against neural variance collapse while
preserving strict out-of-fold evaluation.  The same inner split selects an
early-stopped checkpoint by noisy-target NLL and then estimates a scalar
uncertainty correction; neither operation uses latent clean returns.

The saved `audit_arrays.npz` contains clean/noisy per-transition rewards, the
realized injected noise, latent clean trajectory returns, fold assignments,
raw noisy returns, cross-fitted mean/std, and the LCB score.  The JSON record
contains dataset SHA-256, all seeds and hyperparameters, exact train/test
trajectory ids per fold, wall time, ranking metrics, MSE, and uncertainty
coverage against the latent clean return.

Quick synthetic integration run:

```bash
python mechanism_audit.py make-synthetic \
  --output results/synthetic/source.hdf5 --trajectories 160 --seed 17
python mechanism_audit.py audit \
  --dataset results/synthetic/source.hdf5 \
  --output-dir results/synthetic/audit \
  --noise-std 3 --noise-seed 20260914 \
  --folds 5 --fold-seed 31 --model-seed 47 \
  --target-source noisy --epochs 300 --device cpu
pytest -q tests/test_mechanism.py
```

For a public D4RL file, replace `--dataset`.  `--target-source noisy` is the
deployable setting; `clean` is provided only as an oracle audit control.  This
module measures the proposed scoring mechanism and is not itself an offline-RL
performance result.

The checked-in synthetic record at
`results/synthetic/audit_final/audit_metrics.json` used 160 trajectories,
five outer folds, and Gaussian reward noise with per-transition standard
deviation 3.  On the server GPU, the raw noisy-return / OOF-mean results were:
MSE 65.95 / 8.54, Spearman 0.656 / 0.966, and top-20% precision 0.531 / 0.875.
The LCB score was worse than the mean-only score on this construction, so the
record supports cross-fitted denoising but does not by itself support a
risk-adjusted ranking claim.

## Episode-fold Q/V predictor for CA-HUBL

`qv_predictor.py` fits two return regressors on every non-held-out episode
fold: `V(s) -> G_t` and `Q(s,a) -> G_t`.  Their targets are discounted returns
computed from the seeded noisy rewards; clean returns are saved only for
auditing.  Observation, action, and target normalization statistics are
computed anew from each outer training fold.  Predictions for an episode are
therefore always produced by models that did not train on any transition from
that episode.

The transition convention exactly matches `train_iql.py` for flat D4RL files:
episodes are split on `terminal | timeout`; the final true-terminal row is
kept, whereas the final timeout or incomplete row is dropped.  The NPZ stores
the original HDF5 row of every kept transition and a SHA-256 digest over the
ordered `(observation, action, next observation, episode id, row id)` tuple.
`crossfit_next_mean` is the out-of-fold `V(next_state)` in raw reward units and
can be consumed directly by the `*_h` IQL variants.  At episode level,
`crossfit_mean` averages this denoised next-state value, while
`controllable_score` averages `Q(s,a)-V(s)` to isolate action-dependent return
quality from state visitation quality.  With more than one ensemble member,
the corresponding `*_std` arrays are trajectory-bootstrap epistemic proxies.

Fast deterministic integration check (linear regressors):

```bash
python qv_predictor.py \
  --dataset /root/hubl_backup_data/walker2d_medium-v2.hdf5 \
  --output results/walker/qv_ridge.npz \
  --backend ridge --folds 5 --ensemble-size 3 \
  --iid-noise-scale 1 --episode-noise-scale 1
pytest -q tests/test_qv_predictor.py
```

GPU MLP run with the same data contract:

```bash
python qv_predictor.py \
  --dataset /root/hubl_backup_data/walker2d_medium-v2.hdf5 \
  --output results/walker/qv_mlp.npz \
  --backend mlp --device cuda --folds 5 --ensemble-size 3 \
  --train-steps 3000 --batch-size 1024 \
  --iid-noise-scale 1 --episode-noise-scale 1
```

Each run also writes a JSON sidecar containing the archive checksum, dataset
checksum, exact train/held-out episode ids, fold-local normalization hashes,
all seeds and hyperparameters, and measured wall time.  The ensemble spread is
not calibrated aleatoric uncertainty; it is deliberately labeled as an
epistemic proxy.

## Clean-room IQL/HUBL training harness

`iql_core.py` and `train_iql.py` implement an equal-budget IQL comparison
without importing the unlicensed HUBL supplement.  The loader accepts both the
original flat D4RL HDF5 schema and episodic Minari conversions, reconstructs
the exact trajectory-level HUBL rank, discards timeout-final transitions as in
the D4RL q-learning protocol, and records the source SHA-256, noise realization,
configuration, raw progress, checkpoints, and every evaluation episode.

Available controls are base IQL, constant HUBL, realized-return rank HUBL,
cross-fitted point-estimate rank, LCB rank, and posterior-expected rank.  The
`*_h` variants additionally require an aligned out-of-fold next-state value
array and replace the noisy Monte-Carlo heuristic itself.  Posterior-expected
rank uses

```text
rho_i = mean_{j != i} Phi((mu_i-mu_j) / sqrt(sigma_i^2+sigma_j^2)),
lambda_i = alpha * rho_i.
```

This shrinks uncertain extreme ranks toward the middle; it is distinct from a
risk-sensitive `mu-kappa*sigma` lower quantile.  It remains an experimental
control until it beats both mean-only and constant HUBL.

Server smoke command (actually run against the official one-million-transition
Walker2d-medium-v2 file):

```bash
python train_iql.py \
  --dataset /root/hubl_backup_data/walker2d_medium-v2.hdf5 \
  --output-dir /root/hubl_research_20260914/results/smoke_iql \
  --variant iql --max-episodes 10 --updates 20 \
  --batch-size 32 --hidden-dim 64 --eval-period 20 \
  --eval-episodes 1 --device cuda
```

Full comparison runs use the same data/noise seed and evaluation episode seeds
for every variant.  Reward-noise experiments specify the standard deviation in
units of the clean per-transition reward standard deviation with
`--iid-noise-scale`; they are a controlled corruption protocol and must not be
described as HUBL's action-noise Walker2d protocol.

## Action-noise counterfactual replay audit

`counterfactual_audit.py` consumes the flat HDF5 written by
`collect_stochastic.py`.  For each episode it resets `Walker2d-v4` to the exact
logged initial `infos/qpos` and `infos/qvel`, then performs two deterministic
replays.  The first uses the logged executed actions and checks stored rewards
and next observations.  The second substitutes that trajectory's stored
`clean_policy_actions` open loop.  The first replay is a validity check; the
second return is a mechanism-audit label for how this particular action
sequence changes when injected action noise is removed.

Importantly, the clean-action replay is not a behavior-policy expected return
and not a closed-loop policy evaluation.  It does not recompute policy actions
at the counterfactual states, and it stops if the counterfactual Walker episode
ends before the logged horizon.

```bash
python counterfactual_audit.py \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --output results/walker_action_noise/counterfactual_audit.json \
  --max-episodes 10 --strict
pytest -q tests/test_counterfactual_audit.py
```

The JSON contains the source-file SHA-256, per-episode noisy and clean-action
returns, episode lengths, replay errors, early terminations, tolerances, and an
aggregate pass/fail flag.  `--strict` writes the complete record before exiting
non-zero when logged action replay fails.

Development result (not confirmation):
`results/action_noise_dev/counterfactual_audit_256ep.json` audits the first 256
episodes without outcome-based filtering (79,639 transitions, 7.96% of the
dataset).  Logged-action replay passed for all 256 episodes, with worst reward
and next-observation absolute errors of `2.39e-7` and `4.77e-7`.  In contrast,
the clean-action open-loop replay ended early for 251/256 episodes; its return
had approximately 0.096 Pearson and 0.029 Spearman correlation with logged
return.  We therefore reject this open-loop return as a usable policy-quality
or counterfactual-return label.  The replay validator and raw measurements are
reusable, but this failed label must not be used to tune or support the method.

## Dual-action TD3+BC controls

`train_td3bc.py` keeps the stochastic collector's `actions` (executed actuator
input) and `clean_policy_actions` (behavior-policy command) as distinct batch
fields. `--action-pairing` selects one equal-budget control:

- `executed_executed`: standard TD3+BC, with executed actions for critic
  regression and actor behavior cloning;
- `commanded_commanded`: naive relabeling, with commanded actions for both;
- `executed_commanded`: executed actions for the critic and commands for BC.

The latter is a mechanism/protocol ablation, not currently claimed as a novel
method: commanded/executed action distinctions already appear in the MAMDP
literature. All three settings leave reward preprocessing, network size, and
update budget unchanged. Each scheduled evaluation runs both a clean policy
rollout and a persistent actuator-noise rollout; the latter adds independently
seeded `Uniform[-beta,beta]` noise to every command before clipping, with
`--eval-action-noise-beta 1` by default. Episode environment and noise seeds
are written to `progress.jsonl` and `summary.json`.

```bash
python train_td3bc.py \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --output-dir results/dual_action/executed_commanded \
  --variant td3bc --action-pairing executed_commanded \
  --eval-action-noise-beta 1 --eval-seed 9000 --eval-noise-seed 19000
pytest -q tests/test_td3bc.py
```

## Episode-OOF action-noise Q/V audit

`action_qv_audit.py` joins the stochastic Walker HDF5 with a prediction archive
from `qv_predictor.py`. It verifies the dataset checksum, exact retained-row
order, episode ids, lengths, and saved action deltas before writing
`episode_metrics.csv`. `audit_metrics.json` deterministically derives the full
signal/target correlation grid, top/bottom-quintile tables, deciles, top-set
overlap, and a seeded paired episode bootstrap against raw HUBL ranking. An
optional `counterfactual_audit.py` JSON is joined by episode id; its open-loop
return remains labelled as a mechanism diagnostic, not expected policy return.

```bash
python action_qv_audit.py \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --predictions results/action_qv_audit/qv_predictions.npz \
  --counterfactual results/action_noise_dev/counterfactual_audit_256ep.json \
  --output-dir results/action_qv_audit/analysis \
  --bootstrap-replicates 500
```

The 2026-09-14 development records under `results/action_qv_audit/` cover all
3,255 episodes for both a 5-fold, 3-member ridge predictor and a 5-fold,
single-member 128-wide MLP. They are mechanism-screening runs, not frozen
offline-RL confirmation results.

## Inverse-residual result aggregation

`aggregate_inverse_residual_results.py` consumes only manifest-referenced raw
adapter evaluation and frozen-Q audit JSONs. It writes deterministic JSON, CSV,
and Markdown summaries with per-training-seed values and fixed-checkpoint paired
episode intervals. `inverse_residual_pre_split_dev_manifest.json` labels all
existing adapter evidence as development and retains the failed pre-update
launches. `inverse_residual_frozen_confirmation_manifest.template.json`
uses the fail-closed `inverse-residual-manifest-v2` schema. It predeclares seeds
1/10 for five controls: inverse/sampled, direct/sampled, inverse/Q-at-channel-
mean, wide direct/sampled, and nominal-Q beta-zero direct/sampled. The wide
controls explicitly match `penalty / delta_max^2 = 0.16` and
`learning_rate * delta_max = 7.5e-5`. Every run locks the exact training config,
dataset/base/calibration hashes, split hashes, evaluation seed arrays, independent
K=64 audit protocol at physical beta 1.25, checkpoint format/linkage, and actual
train/core/evaluate/audit implementation hashes. Every paired comparison also
requires the same training seed and base-checkpoint hash. The legacy development
manifest remains readable but is labeled `legacy_v1_not_fail_closed`.

The template deliberately contains implementation-hash placeholders for the
seven adapter files plus the OPEX evaluator and TD3+BC trainer, and cannot be
aggregated. Freeze it exactly once, after code and
input artifacts are final but before any of its ten output directories exists:

```bash
python freeze_inverse_residual_manifest.py \
  --results-root /root/hubl_research_20260914/results \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --calibration /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001_calibration.json \
  --base-seed1 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt \
  --base-seed10 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt
```

The freeze mode computes hashes from those files, requires their hash set to
equal the predeclared dataset/base/calibration set, independently recomputes the
whole-episode train/audit split (including all three index hashes), verifies all
output directories are absent, and atomically creates a new manifest without overwrite.
The wrapper delegates to `aggregate_inverse_residual_results.py`'s canonical
freezer. Strict aggregation inventories each runner-created sibling
`<training_dir>.train.console.log`; only legacy manifests may fall back to the
older in-directory `launch.log` layout.
After every predeclared raw file is complete, aggregate with:

```bash
python aggregate_inverse_residual_results.py \
  --manifest inverse_residual_frozen_confirmation_manifest.json \
  --output-dir /root/hubl_research_20260914/results/inverse_residual_confirm/aggregate \
  --stem frozen_confirmation
```

The commands above document the immutable one-shot chronology; do not rerun the
freeze or execution steps over the completed paths.  The first materialized
manifest (`4b35590d...`) retained placeholder tokens in provenance-map *keys*.
The first actual OPEX invocation rejected it during preflight before writing any
formal raw output.  After the freezer was corrected to scan keys as well as
values, the canonical manifest was created with SHA-256
`57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807`.
The `--dry-run` runner mode only renders commands; it does not validate manifest
semantics.  Actual preflight is the check that detected the invalid first
manifest.

The completed strict aggregation has status `complete` under
`strict_fail_closed_v2`: 10/10 adapter runs, 14 external evaluations, 56 paired
comparisons, and 117 referenced records (97 unique files).  Its immutable output
hashes are:

- JSON: `bbe5d2908df1ec6499a5afd2b4cc7db64b9fc9a83b175b0860a912dc63ba7795`
- CSV: `9c3d896c4a1c9738afeb2be6a7a6ec1607b957d76d2848f30672cbdb70066753`
- Markdown: `c04ee00a1f148c7b70086df84c763b699ffb69b695d17cd0adfce9e68e513d58`

The frozen manifest's `evidence_scope` is deliberately a pre-run statement and
therefore says that the then-template was unusable until its placeholders were
resolved.  That condition was subsequently fulfilled by the canonical manifest;
the sentence is retained verbatim to avoid post-hoc mutation and is not the
current run status.  `formal_confirmation_repro_bundle_v3.tgz` (SHA-256
`d816ecb7b3b70a80e48cdb90156a04dafc35c4185950bb0554735b16ef895d9f`)
supersedes the earlier result-only bundles: it includes all 97 strict-aggregation inputs,
checkpoints, sibling training logs, calibration artifacts, executable code, and
tests plus `requirements-research.txt`.  Extract it at
`/root/hubl_research_20260914`, run
`source /root/hubl_backup_env/bin/activate`, then
`cd /root/hubl_research_20260914/code`.  See
`FORMAL_CONFIRMATION_REPRO.md` for a non-overwriting reaggregation command; the
command above records the original canonical output and must not be rerun over it.

Audit output records both the checkpoint model/calibration beta and the actual
audit beta. Frozen audits must pass `--audit-action-noise-beta 1.25`; this is
essential for the nominal-Q beta-zero training control. Audit JSON/CSV/Markdown
retain means and medians for Q1, Q2, min-twin, and twin-disagreement diagnostics.
Adapter training cost is split into base-actor precompute rows, Q-scale Q1
forward rows, optimizer Q1 forward/backward rows, adapter-MLP forward/backward
rows, and total Q1 forward rows. Adapter rollout cost derives each arm's base-
actor and adapter-MLP rows exactly from raw episode lengths; only paired-total
wall time is available, so no per-arm wall time is imputed. Every resume
invocation recomputes the base commands and Q-scale calibration, so the
aggregator multiplies those row counts by
`resume_count + 1`. `preprocessing_seconds_this_invocation` is only the latest
invocation and is never mislabeled as lifetime time; wall and optimizer time are
cumulative. Exact resume means state-complete same-runtime continuation. The
uninterrupted-versus-resumed bitwise regression is CPU-only, so CUDA bitwise
resume equivalence remains explicitly unverified. Each run also reports its own
frozen Q-scale and estimator label; the Q-at-mean comparison therefore changes
both estimator placement and its estimator-consistent denominator.

The v2 manifest has a separate `external_evaluations` interface; external
controls are never coerced into adapter training runs. `evaluate_td3bc_v1`
entries select either a raw `clean` or `persistent_action_noise` arm and lock the
raw JSON SHA, actual checkpoint and adjacent training-config SHA, transform,
scale, base training variant/action pairing/seed/update count, evaluator hashes,
and exact rollout protocol. The frozen template includes raw-linked identity,
scalar-1.2, oracle-beta inverse, and dual-action controls for base seeds 1/10.
Same-base comparisons enforce identical checkpoint SHA; dual-policy comparisons
are explicitly labeled `different_trained_policy` and lock both distinct hashes.

`channel_opex_v1` entries support the paired `arms.baseline_only` and
`arms.adapted` schema emitted by `evaluate_channel_opex.py`, including T=2. They
lock the full controller dict (baseline transform, selected step size, T, K,
model beta/calibration mode, independent gradient-noise seed, Q reducer and
trust region), base/config/calibration hashes, implementation hashes, three seed
streams, and exact Q-forward/Q-backward accounting. Because a future
confirmation JSON cannot have a predeclared content hash, freeze mode instead
requires its path to be absent; aggregation records its actual source SHA after
the non-overwriting evaluator creates it. Pre-existing TD3+BC controls do have
their content hashes locked and are fully revalidated during freeze.

The frozen template predeclares original-structure OPEX and the dev-selected
inverse- and identity-anchor CA-OPEX controls for base seeds 1/10. OPEX cost is
reported with `adapted_arm_deployment_controller_only` scope: adapted-arm
environment steps, actor rows, Q rows, backward calls, and wall time are the
deployment cost. Separate baseline-arm and paired-evaluation totals preserve
the full experimental expense. The formal rollout path is required to be absent
at freeze time; unlike already-existing TD3+BC records, its future raw SHA is
recorded by aggregation rather than guessed in advance.

All six formal OPEX runs share the gradient-noise seed block
`69300..69349`, so base seeds 1/10 use common random numbers for gradient noise
as well as the common environment/action-noise rollout cases; the beta-zero
original controller records but does not consume that stream. The manifest
predeclares, in fixed order, selected inverse-anchor CA-OPEX minus identity-
anchor CA-OPEX, minus original-structure OPEX, and minus its inverse-only arm
for base-policy checkpoint seeds 1 and 10. JSON/CSV/Markdown report each
checkpoint-level paired difference and the descriptive mean across exactly two
base-policy checkpoints. No training-seed interval is computed, and the common
50-case block is never pooled into 100 independent pipeline repetitions.

CA-OPEX development selection is locked through the v3 chronology record, its
post-evaluation provenance-amendment protocol, the preserved pre-evaluation
protocol/selector/selection, and the original-grid protocol/selector/selection.
The canonical preflight recomputes both arm means and paired differences from
each raw return vector, checks paired environment/noise seeds, and inventories
all ten candidate files by path and SHA. It also requires every candidate's full
implementation map to equal the frozen OPEX implementation, not merely the
evaluator hash. All ten candidate sources are emitted under
`freeze_provenance.verified_opex_development_candidates`.

“Confirmation” here has a deliberately narrow meaning: the frozen adapter and
CA-OPEX versions are not run or tuned on the 50 formal episodes, and base seeds
1/10 are predeclared rather than performance-screened. Some external baseline
arms on the same environment/noise seeds already exist; they are retained in
full and the protocol does not describe the entire episode block as blind.
Cross-file baseline-equivalence declarations then require elementwise identical
returns whenever checkpoint, transform semantics, and rollout seed streams say
the baseline controller is the same (including inverse sampled-vs-Q-at-mean,
identity narrow/wide/nominal, and OPEX/TD3+BC baseline paths).

### Post-confirmation equal-Q-row/backward-call nominal audit

This additional audit has evidence label
`post_confirmation_fresh_rollout_mechanism_audit`. It was designed and frozen
after the formal-confirmation results existed, so it is explicitly **not** a
blind confirmation. Its 50-case holdout uses fresh environment seeds
`79300..79349`, action-noise seeds `89300..89349`, and gradient-noise seeds
`99300..99349`; the independent verifier checks that these do not overlap the
canonical formal manifest. The results therefore provide post-confirmation
mechanism evidence and must not be merged with the formal-confirmation evidence.

The predeclared seed-0 development sweep used ten rollout cases per point and
produced the following independently recomputed adapted scores:

| eta | Raw score | D4RL-normalized score |
|---:|---:|---:|
| 0.01 | 561.905068 | 12.204666 |
| 0.03 | 603.373240 | 13.107980 |
| 0.10 | 939.110700 | 20.421452 |
| 0.30 | 795.281007 | 17.288366 |
| 1.00 | 389.620348 | 8.451735 |

The selector therefore froze `eta=0.1`. This was an interior grid point, so the
predeclared endpoint rule did not launch either outward point (`0.003` or
`3.0`). With that selection frozen, the four holdout controllers had these
absolute D4RL-normalized scores, recomputed directly from their raw returns:

| Controller | Seed 1 baseline | Seed 1 adapted | Seed 10 baseline | Seed 10 adapted |
|---|---:|---:|---:|---:|
| nominal tuned (`identity`, eta 0.1) | 11.090044 | 24.125127 | 10.835412 | 18.365012 |
| nominal matched (`identity`, eta 0.3) | 11.090044 | 18.098014 | 10.835412 | 17.947917 |
| calibrated identity (eta 0.3) | 11.090044 | 24.464305 | 10.835412 | 24.994057 |
| complete calibrated inverse (eta 0.1) | 22.582690 | 32.144720 | 22.127577 | 31.963722 |

The three identity-controller baseline return and length vectors are
elementwise identical within each checkpoint, not merely equal in their means.
The inverse controller has a different baseline transform and is not subject to
that identity-baseline equality claim.

The ten intervals below are paired over the same 50 rollout cases **within one
fixed base-policy checkpoint**. `t 95%` is the paired t interval and `bootstrap
95%` is the seeded paired bootstrap interval for the normalized return
difference.

| Comparison | Checkpoint | Mean difference | t 95% | Bootstrap 95% |
|---|---:|---:|---:|---:|
| complete - nominal tuned | 1 | +8.019593 | [1.863205, 14.175981] | [2.011864, 13.962748] |
| complete - nominal tuned | 10 | +13.598710 | [6.919526, 20.277894] | [7.141191, 20.196525] |
| complete - nominal matched eta 0.3 | 1 | +14.046706 | [9.062706, 19.030706] | [9.331686, 19.016505] |
| complete - nominal matched eta 0.3 | 10 | +14.015805 | [7.732931, 20.298680] | [8.109989, 20.218809] |
| complete - calibrated identity eta 0.3 | 1 | +7.680414 | [2.684510, 12.676318] | [2.877539, 12.515176] |
| complete - calibrated identity eta 0.3 | 10 | +6.969665 | [-0.333054, 14.272384] | [-0.017140, 13.926845] |
| calibrated identity - nominal tuned | 1 | +0.339179 | [-4.911893, 5.590250] | [-4.757674, 5.400341] |
| calibrated identity - nominal tuned | 10 | +6.629045 | [1.240275, 12.017816] | [1.465839, 11.835640] |
| calibrated identity - nominal matched eta 0.3 | 1 | +6.366292 | [1.980787, 10.751796] | [2.149744, 10.611991] |
| calibrated identity - nominal matched eta 0.3 | 10 | +7.046140 | [1.539420, 12.552860] | [1.768266, 12.426237] |

Both interval constructions exclude zero at both checkpoints for complete
versus either nominal controller. For complete versus calibrated identity,
seed 1 excludes zero but seed 10 narrowly includes it in both constructions.
For calibrated identity versus nominal tuned, seed 1 includes zero whereas seed
10 excludes it. The clean fixed-controller test of explicit channel expectation
is calibrated identity versus nominal matched eta 0.3: identity transform,
`eta=0.3`, `delta_max=2`, K, and T are held fixed, and both intervals exclude
zero at both checkpoints. In contrast, complete versus calibrated identity
changes anchor transform, eta, and `delta_max` together; it is an entire-design
comparison, **not** a single-factor anchor ablation.

The corresponding two-checkpoint mean differences are respectively
`+10.809152`, `+14.031256`, `+7.325040`, `+3.484112`, and `+6.706216` in the
table's comparison order. These are descriptive summaries over exactly two
base-policy checkpoints. They are not a training-seed confidence interval or
p-value; the 50 paired cases per checkpoint are not pooled into `n=100`, and no
multiple-comparison-corrected claim is made. In particular, `n=2` is not treated
as training-seed significance.

Every controller uses `K=8`, `T=2`, 16 Q1 forward/backward rows, two backward
calls, and one actor-forward row per adapted decision. Thus the comparison
matches the nominal per-decision neural row/call budget. It is not exact FLOP or
wall-time equality: inverse anchoring additionally performs 48 deterministic
scalar bisection iterations, calibrated controllers sample channel noise while
nominal controllers repeat zero noise, and total rows/calls differ when episode
lengths differ. The v2 verifier independently recomputes all totals from the raw
length vectors.

The independent v2 report has status `verified_complete`, reconstructs eight
absolute controller rows and all ten paired comparisons from raw returns, and
verifies a 29-file input-hash inventory. Frozen executable provenance is:

- protocol `equal_compute_nominal_control_protocol.json`:
  `fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034`;
- runner `run_equal_compute_nominal_control.py`:
  `c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1`;
- independent verifier `verify_equal_compute_nominal_control.py`:
  `053d3d823bf23475385b956426f67d78c213bd0cc74b482cf995d7c9ddf2da90`.

The immutable v2 output SHA-256 values are:

- JSON: `3ca8b8c1e342d623f43ec73a3b16bc9ed9a51cad40e662867771110f2b8acf26`;
- CSV: `358a571aeadf8166273b054667a5ef40adea7a525579db5d9f8be1c310ef032d`;
- Markdown: `4b9dd5227f39f86ce537824ff729f045f077a4865cd7263b972fd4b7799c3433`.

`equal_compute_nominal_control_bundle.tgz` has SHA-256
`f7fa50b705171e141e85b84a4632d395ad679141bfc4f5fd159ab8b32792c84e`.
It is an overlay for the formal v3 reproduction tree described above. After
extracting both into `/root/hubl_research_20260914`, the completed immutable run
can be validated without overwriting it:

```bash
source /root/hubl_backup_env/bin/activate
cd /root/hubl_research_20260914/code
python run_equal_compute_nominal_control.py --dry-run
python run_equal_compute_nominal_control.py --resume-missing
python verify_equal_compute_nominal_control.py --check-only
pytest -q
```

After adding the scale-up safeguards, the final suite passed as `232 passed`
with exit code 0.  The retained record includes command, cwd, Python version,
and relevant source hashes: `results/final_validation_20260915.log` (SHA-256
`aa0bde3d5749cc8000be1f58f5ae586a304e97091a7dac0a2ade9956080f841e`).
For a genuinely fresh execution,
the create-only runner requires the protocol's output root to be absent, and the
verifier similarly creates v2 outputs only when its target directory is absent.
Use `--resume-missing` and `--check-only` for the packaged completed result
rather than attempting to overwrite canonical files.

### Fail-closed Walker2d seed expansion

`run_ca_opex_walker_scaleup.sh` is the executable expansion entry point.  It
requires an entirely absent per-seed root, rejects development/formal training
seeds `0/1/10`, rejects overlap with both completed 50-case RNG blocks, and
verifies the source dataset and calibration hashes.  It trains the frozen 25k
HUBL configuration, evaluates complete CA-OPEX plus calibrated identity,
equal-neural-budget nominal-Q, original OPEX, and inverse-only on one common
block, then invokes `aggregate_ca_opex_walker_scaleup.py`.  The aggregator
independently validates the checkpoint, calibration file, implementation hashes,
five controller contracts, seed pairing, costs, and raw-return statistics before
create-only writing `aggregate.json`.

```bash
bash run_ca_opex_walker_scaleup.sh \
  --seed 2 --env-seed 131300 --noise-seed 231300 --grad-seed 331300 \
  --dry-run

# Execute only after reviewing the seven rendered commands.
bash run_ca_opex_walker_scaleup.sh \
  --seed 2 --env-seed 131300 --noise-seed 231300 --grad-seed 331300
```

The driver SHA-256 is
`9f78809ac62d7e3a931ded3130d3a91f979baef5d65553a06977d61625d7911d`;
the aggregator SHA-256 is
`20b8ddfe74ab253d29a74be7b0c39bb58423a74950b369516958a040975e3cac`.
Their Linux-targeted tests passed 36/36.  The exact seed-2 dry-run above returned
0 and left its output root absent; its validation log is
`results/scaleup_seed2_dry_run_validation_20260915.log`, SHA-256
`c45d1510096448a47bed10ea9430866624f8863d6459a24a5ef51e3fdd9bf369`.

### Walker2d five-seed expansion (completed 2026-09-15)

Seeds 2/3/4 were subsequently executed and completed (UTC launch timestamps
`01:31:20`/`01:54:58`/`03:17:26`, driver PIDs `13893`/`14533`/`15028`; each run
about 18 minutes, all stage logs retained).  The five-training-seed aggregate
(`aggregate_ca_opex_walker_multiseed.py`, SHA-256
`952ac1b092f6b5189609f5c1969711ddacce09d24b4fb6ba8bb380b7e20fb1b1`) and its
independent verifier (`verify_ca_opex_walker_multiseed.py`, SHA-256
`8182ea311cd9f4b45be685d96d8e43457ceb5c2073f14f272eb709116189aced`, 96 checks
zero deviations) produce the per-seed and cross-seed statistics recorded in
`../RESEARCH_STATE.md` and `../RESEARCH_PACKAGE.md` section 4.  Output SHA-256:
`multiseed_aggregate.json`
`86f66982367216c069872f6e70431605854e284708749fb9aed9d719e53d39ad`;
`multiseed_verify.json`
`df0f7cf335b92b56a9a9ae4fd023858672c38f951e893dffbf1af172ebeec0b1`.

Additional cross-task and severity tooling added the same day:
`run_ca_opex_hopper_crosstask.sh` + `aggregate_ca_opex_hopper_crosstask.py`
(transformed deterministically from the frozen Walker2d pair by
`make_hopper_crosstask_tools.py`; dataset and calibration SHA placeholders are
filled after data creation) and `run_ca_opex_walker_betagrid.sh` +
`aggregate_ca_opex_walker_betagrid.py` + `betagrid_run_all.sh` for the frozen
beta grid `{0.5, 0.9, 1.1, 1.4}`.  Their guard tests pass on server and
Windows.

The seed-2 dry-run itself started no training, but the later real seed-2
launch was verified from PID `13893`, its stage log, raw controls, aggregate,
and 25k checkpoint.  Seeds 3 and 4 were also subsequently completed under the
same frozen settings.  Hopper-v4 then completed the cross-task replication;
HalfCheetah remains unrun.

### Hopper-v4 cross-task replication (completed 2026-09-15)

Source `hopper_medium-v2.hdf5` (SHA-256
`5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0`) passed the
embedded-policy audit (Pearson `0.9999999983`); the one-shot pipeline
`run_hopper_collect_and_calibrate.sh` produced the smoke set, the 1M-transition
dataset (`hopper_v4_beta1_seed2201.hdf5`, SHA-256
`19bff1e68edef8fdd3ef8b4911a74d0c898e7b81970ca73975b6a0c9f04b9ee1`), and the
512-pair calibration (`beta_mle=1.2499450445175158`).  Training seeds 2/3/4 ran
the frozen five-controller protocol; an initial seed-2 aggregation failure
(Walker's pinned beta constant inherited by the transformed aggregator; no
output written) is preserved in
`results_hopper/hopper_seed2_aggregate.console.log` and was recovered by a
labeled retry (`results_hopper/hopper_seed2_aggregate.retry1.console.log`)
after the constant was corrected.  Three-seed aggregate SHA
`eebe852bbb0ec9ad566e214e5841a8d49585dd4b0417881a5732ed777eb54209`,
independent verifier `verified_complete` with 57 checks and zero failures.
Result: complete CA-OPEX beats original OPEX on 3/3 seeds (mean `+2.25`,
interval excludes zero) and has positive point means over nominal-Q and
identity, but shows no detectable incremental gain over the calibrated
inverse-only anchor (mean `-0.57`); the value-correction stage is therefore
Walker-specific evidence with an explicit Hopper counterexample.  A bounded
zero-GPU diagnostic (termination-dominated episodes, `0/10` reaching the
1000-step horizon; weaker base policy ~`33.6-39.5` vs Walker `80.6` clean)
is the leading explanation.  Full numbers in `../RESEARCH_STATE.md` and
`../RESEARCH_PACKAGE.md` section 4.

### Beta-severity grid (running 2026-09-15)

`betagrid_run_all.sh` runs the frozen grid `{0.5, 0.9, 1.1, 1.4}` on
checkpoints 1/2/10 (12 create-only units, sequential on the single GPU); each
beta generates its own 512-pair calibration and its own 50-case block and no
K/T/eta/delta is reselected.  At the time of writing the chain is running;
aggregate with `aggregate_ca_opex_walker_betagrid.py` after completion.  No
beta-grid result is claimed yet.

The chain was checked as genuinely active at takeover under PID `18625`; its
launch record is `results/betagrid_chain_20260915.launch.info` and its progress
log is `results/betagrid_chain_20260915.nohup.log`.  Completed partial units
are preserved and create-only.  The updated beta aggregator rejects unexpected
duplicate unit directories and overlapping seed blocks; its independent
verifier and guard tests must pass before any beta result is reported.

The guarded beta aggregator SHA-256 is
`7ec29fda1e4c1f9ed8c0fc64c69c79b6d4ff4d8c608b082bdc8a406e6c7c9c7a`; the
independent beta verifier SHA-256 is
`23ea6a55954dabff303e4e7ccaa3689afc5d19aeff51156a2efb4e4f4e0ece49`.

The canonical new completed-evidence delivery is `results/ca_opex_scaleup_bundle_v3.tgz`
(19 MB, 275 payload files, SHA-256
`e2f889d2e69e7a12d964aa7a185dd797bce3d3ceb4f3359c9baad61e2e47766b`).  It
contains the Walker2d/Hopper-v4 checkpoints, raw records, calibration
artifacts, code, tests, and logs; source HDF5 files are excluded and formal v3
is a separate bundle.  The fresh takeover test log
`results/final_validation_takeover_20260915.log` records `55 passed` with exit
code 0 (SHA-256
`20a2d5ad5fb735d91d4d03e9b2100c367895ef4fdcf4bf9c50d1dc1df026850a`).
The corrected full suite from the canonical `code/` entry point records
`251 passed` in `results/final_validation_takeover_full_v2_20260915.log`
(SHA-256
`a574edb1c2ddc1244c0f9f70635b57cc5486992cbea53f73b1c4e1de8ff3cec8`).
The earlier v1/v2 archives are retained as noncanonical snapshots.

The final supplemental archive `ca_opex_delivery_overlay_v2.tgz` (stored under
`results/` in the local delivery) has SHA-256
`7701de19d50478522084c3f3d0a3ed22888d4c8164898b0ea73e31426f4cff17`.
Its 51 entries contain a README snapshot current through the scale-up
instructions above, the complete equal-budget raw audit, both new scale-up tools
and tests, and the final/dry-run validation logs.  This checksum paragraph was
necessarily added after the archive was frozen.  The archive is an overlay for
`formal_confirmation_repro_bundle_v3.tgz`, not a standalone copy of the base
checkpoints.
