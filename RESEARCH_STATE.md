# Research state

Last updated: 2026-09-15 (Asia/Shanghai)

## Status

`WALKER2D FIVE-SEED EXPANSION COMPLETE — HOPPER-V4 CROSS-TASK COMPLETE — BETA-SEVERITY GRID RUNNING`

The Walker2d expansion (training seeds 2/3/4) and its five-training-seed
statistics are complete and independently verified.  The Hopper-v4 cross-task
replication is complete: calibrated mean inversion transfers, while the
frozen-critic value-correction increment is not detectable on Hopper (full
numbers and the bounded diagnostic below).  The beta-severity grid
`{0.5,0.9,1.1,1.4}` x checkpoints `{1,2,10}` is running as of
`2026-09-15T06:11:15Z` (chain PID 18625, `betagrid_run_all.sh`; driver log
`results/betagrid_chain_20260915.driver.log`, unit progress
`results/betagrid_chain_20260915.nohup.log`; units in
`results/betagrid/beta<beta>_seed<seed>/`).  No beta-grid result is claimed
yet.  To resume or verify: units are create-only — inspect the chain log; once
all 12 units exist, run `python aggregate_ca_opex_walker_betagrid.py --output
results/betagrid/betagrid_aggregate.json`; if the chain stopped on a failed
unit, diagnose the preserved unit directory and restart the chain only after
inspection (completed units will refuse reuse).

## 2026-09-15 takeover audit

At takeover, the requested Walker2d scale-up was already complete on the
server.  The retained seed-2 dry-run validation records the seven rendered
commands, hash and RNG-block checks, and absent output root; the subsequent
real launch was PID `13893` and its seed-2 raw controls, 25k checkpoint, stage
logs, and aggregate are present.  Seeds 3 and 4 are likewise complete.  No
duplicate seed-2 run was started because the create-only protocol rejects its
existing output root.

An independent takeover verifier re-read the Walker raw records and reproduced
the five-seed result with 96 checks and zero failures.  The corresponding
Hopper verifier reproduced the three-seed cross-task result with 57 checks and
zero failures.  These verifier outputs are new audit artifacts; formal v3 and
the prior canonical aggregates were not modified.

The optional beta-severity chain is a real sequential single-GPU run, not a
dry-run: chain PID `18625`, launch record
`results/betagrid_chain_20260915.launch.info`, and progress log
`results/betagrid_chain_20260915.nohup.log`.  Its partial units remain
create-only and no beta statistic is claimed until all 12 units are complete,
then aggregated and independently verified.

The latest verified snapshot at `2026-09-15T07:32:08Z` has four complete units
(`beta0.5_seed1/2/10` and `beta0.9_seed1`, each 4/4 controller arms).  The
next create-only unit `beta0.9_seed2` has started; its driver child PID is
`22544`, with the complete arm in progress.  This is still not a beta result.

## 2026-09-15 Walker2d seed-expansion completion

Seeds 2/3/4 each trained an identical 25k-update HUBL base policy and evaluated
five controllers on one 50-case common-random-number block: complete CA-OPEX
(inverse anchor, calibrated, K=8, T=2, eta=0.1, delta=0.25), calibrated identity
CA (eta=0.3), equal-neural-budget nominal-Q (K=8, T=2, eta=0.1), original OPEX
(K=1, T=1), and calibrated inverse-only.  No hyperparameter was retuned.

Normalized (D4RL-reference) mean scores per training seed — complete CA-OPEX:
seed 2 `33.2987`, seed 3 `30.0550`, seed 4 `38.7492` (raw `1530.26/1381.36/
1780.48`).  Per-seed complete-minus-control normalized deltas: vs tuned
nominal-Q `+11.53/+7.35/+16.49`, vs original OPEX `+12.92/+7.14/+18.87`, vs
inverse-only `+11.93/+6.20/+14.77`, all with paired t and bootstrap intervals
excluding zero on every new seed.  Vs calibrated identity CA the deltas are
`+3.56/-0.19/+12.36`: seed 2's interval crosses zero and seed 3 is slightly
negative, so the identity-anchor margin is seed-sensitive and is reported as
the weakest comparison.

Five-training-seed statistics (one mean per independent base checkpoint, n=5;
the 250 rollout episodes are never pooled): complete minus tuned nominal-Q
`+11.3984` t95 `[+6.6454,+16.1514]` (5/5 seeds positive), minus original OPEX
`+12.6304` `[+7.4201,+17.8407]` (5/5), minus inverse-only `+10.1345`
`[+6.0483,+14.2207]` (5/5), minus identity command `+20.2371`
`[+14.7976,+25.6765]` (5/5), minus calibrated identity CA `+6.1597`
`[+0.0080,+12.3114]` (4/5).  Complete CA-OPEX per-seed normalized means across
seeds `1/2/3/4/10` are `31.2110/33.2987/30.0550/38.7492/27.6997`, mean
`32.2028` (sd `4.1812`).  Seeded 20k-bootstrap per-seed intervals are retained
in the aggregate JSON.

Artifacts (server `/root/hubl_research_20260914/results/scaleup/`):
`multiseed_aggregate.json` SHA-256
`86f66982367216c069872f6e70431605854e284708749fb9aed9d719e53d39ad`;
independent verifier `multiseed_verify.json`
`df0f7cf335b92b56a9a9ae4fd023858672c38f951e893dffbf1af172ebeec0b1`
(status `verified_complete`, 96 checks, 0 failures, no aggregator import).
Per-seed aggregates: seed 2 `75f28e6391b63c3af6a0febeb2ce09422bd8011bcf28cda1e145132c274f2091`,
seed 3 `5814cacc0120b6a484de1e76dc7d13b5ac5add2986c5710be17e6a96f9e4edcf`,
seed 4 `1ff48653562df23a0effddaf4be47d3b31748fb1c3e73cb3e8ec08cc018f580f`.
New base checkpoints: seed 2
`d94b05044c9071c2a987aa28bd2dabbf04a91a7164d4215f851f37aaff06a75e`, seed 3
`d788b745c3a2bfb53409f31e1da1061e09c0ea62e42dd658d055f0789e0168ae`, seed 4
`ed6bdf90d2f82beea16eba3ba4107e4719bf45245cd3a4db6faf60cea90d9da2`.
Recorded per-seed base-training wall times are `271.6/274.5/269.4 s`; total
recorded base-training time for all five seeds is `1350.9 s` and total recorded
rollout wall time across the five controller arms on all five seeds is
`3521.8 s`.  "Equal compute" for the controller comparisons remains
per-decision neural rows/backward calls (1 actor row, 16 Q1 rows, 2 backwards
for all K=8/T=2 controllers), not FLOP or wall-time equality; inverse
anchoring adds 48 scalar bisection iterations per decision.

New tools (server code dir, mirrored locally):
`aggregate_ca_opex_walker_multiseed.py` SHA-256
`952ac1b092f6b5189609f5c1969711ddacce09d24b4fb6ba8bb380b7e20fb1b1`;
`verify_ca_opex_walker_multiseed.py`
`8182ea311cd9f4b45be685d96d8e43457ceb5c2073f14f272eb709116189aced`.
Their 8 guard tests (missing arm, tampered pinned hash, duplicate training
seed, reserved-block reuse, scaleup-block overlap, overwrite refusal) pass on
both Windows and the server.  Server run evidence: launch records
`results/scaleup_driver_seed{2,3,4}_20260915.launch.info` (UTC timestamps and
PIDs 13893/14533/15028), driver logs
`results/scaleup_driver_seed{2,3,4}_20260915.nohup.log`, and the interim full
suite result (240 passed, 2 skipped) in
`results/interim_pytest_20260915.log`.

Hash-check note: the formal v3 bundle is stored at the project top level
(`/root/hubl_research_20260914/formal_confirmation_repro_bundle_v3.tgz`) rather
than under `results/inverse_residual_confirm/` as older text suggested; its
SHA-256 still matches `d816ecb7...`.  The supplemental overlay also lives at
the project top level and matches `7701de19...`.  No content discrepancy was
found; only the stored path differs.

## Hopper-v4 cross-task status (COMPLETE — negative sub-result on the value-correction stage)

The official D4RL source `hopper_medium-v2.hdf5` finished downloading on
2026-09-15 (152,756,557 bytes; SHA-256
`5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0`).  The
embedded tanh-Gaussian SAC policy passes the log-probability audit (10k-transition
smoke: Pearson `0.9999999983`, MAE `4.8e-5`).  The one-shot pipeline
`run_hopper_collect_and_calibrate.sh` then produced, all create-only:

- smoke set `hopper_v4_beta1_smoke10k.hdf5` (10k transitions, SHA
  `f1e266dd7d626e69595e3449ec322f1a5fe58bd74d79174c6cf5bb8d53bd2d05`);
- full 1M-transition beta-1 dataset `hopper_v4_beta1_seed2201.hdf5`
  (237,689,585 bytes; SHA
  `19bff1e68edef8fdd3ef8b4911a74d0c898e7b81970ca73975b6a0c9f04b9ee1`; collected
  04:54:06--05:15:45Z at ~780 steps/s, RNG env/noise/policy/audit seeds
  2201/2210/2211/2212);
- 512-pair calibration `hopper_beta125_n512_seed27002_calibration.json` (SHA
  `d7f273b39ae935bb31e62d12f41861ca216f89c8c5086899f355b6ebfd7941f0`), fitted
  `beta_mle = 1.2499450445175158` against the known 1.25 construction value.

Normalization references `random=-20.272305, expert=3234.3` were verified from
`d4rl/infos.py` (GitHub master, SHA-256
`4460cbb9d49e05d5b494a94f922bac1ed3ee559dbf298fd14a658f70f8c86732`) before any
Hopper outcome existed; the same Gymnasium-vs-D4RL-v2 mismatch caveat as
Walker2d applies, so raw returns remain the primary metric (all reported
comparison signs are identical in raw and normalized units).  Frozen choices
carried over: HUBL base recipe (25k updates, constant lambda
`0.391843318939209`), controller K/T/eta/delta, rollout blocks seed 2
`41300/51300/61300`, seed 3 `41400/51400/61400`, seed 4 `41500/51500/61500`;
training seed 0 reserved for development only.  Driver SHA
`832fb93913a5881adc40e61a6eb111f2826782ea65b6a0c699f3c2be8adcd2e0`;
aggregator SHA after the fix below
`2ce3fbdebab4e337da51e3a17c8f6c1a87a515e8aa6f788581ffc1a4a634a968` (the
pre-fix transformed copy `6e13339f...` was uploaded, filled, and then corrected
in place; see chronology).

**Engineering fix chronology (recorded, not hidden).**  The transformed
aggregator initially inherited Walker's pinned `CALIBRATED_BETA` constant
`1.2498948872089386`, so the seed-2 post-run aggregation step failed at
05:35:39Z with `controls.complete.model_beta mismatch` before writing any
output.  The five seed-2 controller records were complete and valid; only the
aggregation step failed.  The failed console log is preserved as
`hopper_seed2/aggregate.console.log`.  The constant was corrected to the
Hopper calibration value `1.2499450445175158` and the aggregation was re-run
as a clearly labeled retry (`aggregate.retry1.console.log`), creating the
canonical `hopper_seed2/aggregate.json`.  Seeds 3/4 ran with the fixed
aggregator and aggregated on first attempt.  Driver launch UTC timestamps
`05:24:43/05:41:57/05:55:40` (PIDs `17249`/child of `17720`/`18055`); seed 3
and 4 completed at `05:51:59` and `06:07:03`.

**Three-seed results (50 paired beta-1.25 rollout cases per seed; one mean per
independent base-policy checkpoint, n=3; the 150 episodes are never pooled).**
Normalized controller means per seed (2/3/4) — complete CA-OPEX
`20.5232/20.7687/19.9657` (mean `20.4192`, sd `0.4115`); calibrated identity CA
`10.6601/3.8647/17.9089` (mean `10.8112`, sd `7.0233` — strongly
seed-dependent); tuned nominal-Q `14.7047/13.1143/20.0339` (mean `15.9510`);
original-structure OPEX `17.9971/18.0831/18.4345` (mean `18.1716`); calibrated
inverse-only `21.2334/20.5417/21.1981` (mean `20.9911`, sd `0.3896`).

Cross-seed paired deltas (normalized, complete minus control):

| Comparison | Per-seed deltas (2/3/4) | Mean | 95% t interval | Positive seeds |
|---|---|---:|---:|---:|
| complete - original OPEX | +2.5261/+2.6856/+1.5312 | +2.2476 | [+0.6936, +3.8016] | 3/3 |
| complete - nominal-Q (K=8,T=2) | +5.8185/+7.6544/-0.0683 | +4.4682 | [-5.5541, +14.4906] | 2/3 |
| complete - calibrated identity CA | +9.8631/+16.9040/+2.0568 | +9.6080 | [-8.8415, +28.0574] | 3/3 |
| complete - calibrated inverse-only | -0.7102/+0.2270/-1.2325 | **-0.5719** | [-2.4089, +1.2652] | 1/3 |

Independent verifier `multiseed_verify.json` reports `verified_complete` with
57 checks and 0 failures against aggregate SHA
`eebe852bbb0ec9ad566e214e5841a8d49585dd4b0417881a5732ed777eb54209`
(`results/crosstask_hopper/multiseed_aggregate.json`); per-seed aggregates are
`hopper_seed2/3/4/aggregate.json`.

**Interpretation.**  On Hopper-v4 the calibrated mean-inversion part of
CA-OPEX transfers: complete beats original OPEX robustly (3/3 seeds, interval
excludes zero) and has positive point means over nominal-Q and identity CA
(the latter interval crosses zero because the identity control itself is
extremely seed-dependent).  The frozen-critic expected-value correction stage
does **not** show a detectable incremental gain over its own inverse anchor on
this task (mean `-0.57`, 1/3 seeds positive, interval crosses zero).  The
cross-task claim therefore narrows to: channel calibration plus mean inversion
generalizes to Hopper; the value-correction increment remains
Walker-specific evidence with this honest Hopper counterexample recorded.

**One bounded diagnostic (zero-GPU, existing records).**  Base-policy internal
evaluations (10 episodes; clean / persistent beta-0.5 noise) show: Hopper clean
normalized `39.5/39.0/33.6` versus Walker seed-2 `80.6`; Hopper mean episode
length `310-410` steps with `0/10` episodes reaching the 1000-step horizon in
every seed, versus Walker `898-988` steps with `7-9/10` full-horizon episodes.
Hopper is therefore termination-dominated with a comparatively weaker base
policy, so the one-step frozen-Q correction has fewer effective steps and its
valuations near fall-termination are dominated by survival risk rather than by
physical-action-noise curvature.  This supports "the value-correction
increment does not transfer to the termination-dominated regime" over
alternatives such as an environment/termination bug (plumbing validated by the
policy audit and consistent seed-to-seed base evals) or a normalization
artifact (raw and normalized comparison signs agree everywhere).  No
confirmation seed was used to retune anything; K/T/eta/delta remain frozen.

## 2026-09-15 formal confirmation result

The canonical fail-closed manifest is
`inverse_residual_frozen_confirmation_manifest.json` (SHA-256
`57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807`).
Strict aggregation completed with 10/10 adapter runs, 14 external controls, 56
comparisons, and 117 referenced records / 97 unique inputs.  An independent
audit recomputed all 56 return comparisons and 690 cost/mechanism quantities
from raw records with zero mismatch.

On 50 paired beta-1.25 rollout cases per checkpoint, inverse-anchor CA-OPEX
scored `31.2110/27.6997` normalized for base seeds 1/10.  Tuned equal-Q-call
identity-anchor CA scored `21.5616/22.2849`, original-structure OPEX
`18.5048/16.1928`, and inverse-only `22.2513/18.8868`.  The two-checkpoint
descriptive margins are therefore `+7.5322`, `+12.1066`, and `+8.8863`, with
all three comparisons positive on 2/2 checkpoints.  Per-checkpoint paired
bootstrap intervals exclude zero in all six primary rows.  The seed-10
inverse-versus-identity paired-t interval is `[-0.0048,10.8346]`, so no claim is
made that both interval procedures exclude zero everywhere.  Cross-checkpoint
`n=2` supports descriptive means only; rollout episodes are not pooled into a
training-seed CI.

The learned inverse residual adapter scored `31.0258/24.5847`, positive over
inverse-only but below the unamortized primary method by `0.1852/3.1150` points.
Q-at-mean versus sampled has mixed sign, and clean-channel regressions for the
target-specific adapters also have mixed sign.  K=64 held-out-state audits show
that wide/nominal adapters obtain larger frozen-Q proxy gains while producing
worse closed-loop return, which supports the tight inverse-centered guardrail
and rejects Q gain as a standalone success metric.

CA-OPEX costs one base-actor row, 16 Q1 rows, and two backward calls per adapted
decision; original OPEX costs one Q row and one backward.  The calibrated
identity controller matches CA-OPEX's per-step neural-call budget, but inverse
adds 48 scalar bisection steps and the two controllers use independently selected
`eta/delta`; it therefore tests the selected inverse-centered stack rather than
isolating the anchor as a single factor.  The later nominal-Q audit resolves the
neural-call confound and provides a fixed-identity single-factor expectation
comparison, as recorded below.  It cannot retroactively make the formal
experiment blind.

The complete reproduction archive is
`results/inverse_residual_confirm/formal_confirmation_repro_bundle_v3.tgz`
(SHA-256
`d816ecb7b3b70a80e48cdb90156a04dafc35c4185950bb0554735b16ef895d9f`).
It contains all 97 aggregate inputs, raw records, checkpoints, sibling logs,
calibration artifacts, executable code, tests, and dependency pins.  The large
source HDF5 is referenced by hash rather than duplicated.

## 2026-09-15 post-confirmation equal-neural-budget audit

The launch-frozen audit tuned only nominal-Q `eta` on the retained seed-0
development block.  Scores for `.01/.03/.1/.3/1` were
`12.2047/13.1080/20.4215/17.2884/8.4517`; the selected `.1` is interior and no
endpoint extension was triggered.  It then used fresh, formal-disjoint rollout
seeds `79300../89300../99300..` for 50 paired cases on base checkpoints 1 and
10.  This evidence was designed after reading the formal result and is labeled
`post_confirmation_fresh_rollout_mechanism_audit`, not blind confirmation.

All four controllers use one actor row, `K=8,T=2`, 16 Q1 rows, and two backward
calls per adapted decision.  Fresh normalized scores for seeds 1/10 were:
nominal tuned `24.1251/18.3650`, nominal matched-eta `.3`
`18.0980/17.9479`, calibrated identity eta `.3` `24.4643/24.9941`, and complete
calibrated inverse `32.1447/31.9637`.  Complete minus independently tuned
nominal is `+8.0196/+13.5987`; paired t and 20k-bootstrap intervals exclude zero
at both fixed checkpoints.  At fixed identity anchor, eta `.3`, delta `2`, K,
and T, calibrated expectation minus nominal-Q is `+6.3663/+7.0461`, again with
both interval types excluding zero in both rows.  Allowing nominal-Q its own eta
makes calibrated identity minus nominal only `+0.3392/+6.6290`, with seed 1
intervals crossing zero.  Complete versus calibrated identity also changes
anchor, delta, and selected eta, so it is not labeled an anchor-only ablation.

The independent verifier reconstructed the result from raw returns without
importing the v1 runner.  It passed 29/29 input hashes, all 8 absolute records,
10 paired comparisons, identity-baseline equality, seed disjointness, and cost
identities with zero deviation.  After adding the scale-up safeguards, the final
server suite passed 232 tests with exit code 0.  Its retained provenance log is
`results/final_validation_20260915.log`, SHA-256
`aa0bde3d5749cc8000be1f58f5ae586a304e97091a7dac0a2ade9956080f841e`.  V2
hashes are JSON `3ca8b8c1e342d623f43ec73a3b16bc9ed9a51cad40e662867771110f2b8acf26`,
CSV `358a571aeadf8166273b054667a5ef40adea7a525579db5d9f8be1c310ef032d`,
and Markdown `4b9dd5227f39f86ce537824ff729f045f077a4865cd7263b972fd4b7799c3433`.
The supplemental overlay archive, to be extracted on top of the formal v3 tree, is
`results/equal_compute_nominal_control_bundle.tgz`, SHA-256
`f7fa50b705171e141e85b84a4632d395ad679141bfc4f5fd159ab8b32792c84e`.
Cross-checkpoint summaries remain descriptive `n=2`; no training-seed interval
or pooled-episode significance claim is made.

The Walker2d expansion entry point is now
`run_ca_opex_walker_scaleup.sh` (SHA-256
`9f78809ac62d7e3a931ded3130d3a91f979baef5d65553a06977d61625d7911d`).
It atomically claims a new per-seed directory, rejects training seeds `0/1/10`
and both used rollout blocks, runs the five required controllers, and calls the
strict raw-return aggregator `aggregate_ca_opex_walker_scaleup.py` (SHA-256
`20b8ddfe74ab253d29a74be7b0c39bb58423a74950b369516958a040975e3cac`).
The two tools passed 36 targeted Linux tests.  The exact seed-2 dry-run returned
0 and left the output root absent; retained log SHA-256 is
`c45d1510096448a47bed10ea9430866624f8863d6459a24a5ef51e3fdd9bf369`.
That dry-run did not launch training; the later real scale-up launch and its
completion are recorded in the 2026-09-15 takeover audit above.  The final
51-entry supplemental delivery
overlay is `results/ca_opex_delivery_overlay_v2.tgz`, SHA-256
`7701de19d50478522084c3f3d0a3ed22888d4c8164898b0ea73e31426f4cff17`;
it must be extracted over, not substituted for, the formal v3 checkpoint tree.

## 2026-09-14 latest decision: CA-OPEX becomes the primary confirmation candidate

The strongest matched development evidence now favors a test-time, channel-aware
policy-extraction method rather than the first amortized CCIRA adapter.  On one
predeclared 10-episode development block (base-policy training seed 0, environment
seeds 28300--28309, actuator-noise seeds 38300--38309, target beta 1.25), the
calibrated inverse anchor scored `24.8733`.  Original-structure OPEX
(`identity`, model beta 0, K=1, T=1, eta=.1) scored `20.3798`, and the
channel-aware identity-anchor control (`K=8`, `T=2`, eta=.1) scored `19.5429`.
In contrast, channel-aware OPEX initialized at the calibrated generalized inverse
scored `38.7645`, a paired raw-return gain of `+637.70` over its inverse-only
anchor.  Thus neither online Q gradients alone nor channel marginalization around
the uncorrected command explains the signal; the current hypothesis is that mean
channel inversion moves commands into the correct basin and expected-Q gradients
then correct value curvature left by mean matching.

The eta value was selected separately for inverse and identity anchors by a
fully retained grid `{.01,.03,.1}` and an automatic largest-mean rule.  After a
code audit corrected zero-noise OPEX's boundary gradient and result metadata, the
entire positive-beta grid was replayed under a protocol that locks evaluator SHA
`cf659f47...`; it reproduced the same scores and selected eta `.1` for both
anchors.  The final development selection is
`results/channel_opex_dev/selection.json` on the server (SHA
`35821ef5...`).  The selected CA-OPEX and adapter confirmation outputs did not
yet exist and were not read during this selection.  External baseline arms on
the same formal seed identifiers did already exist and had been observed; those
predeclared baselines are therefore controls, not blind outcomes.

Because both anchors initially selected the upper grid boundary `.1`, a second
predeclared development-only extension evaluated `.3` and `1.0` without changing
any other field.  The five-point inverse scores were
`22.9299/26.5033/38.7645/37.8107/20.0936`, so inverse remained at the now-interior
`.1`.  The stronger identity competitor selected `.3`, with five-point scores
`13.5830/13.8171/19.5429/31.0982/15.0658`; inverse therefore retained a `+7.6663`
normalized-score development margin over the tuned identity channel-aware
control.  An audit then found that the boundary selector imported the original
grid validator without hashing the live dependency.  The original pre-evaluation
protocol (`cba9c405...`), selector (`22d3eb49...`), selection (`603d5a70...`),
and all raw runs remain immutable.  A post-evaluation provenance-only amendment
now verifies that full chain and recomputes all ten candidate records exactly;
its scoped final v3 record is
`results/channel_opex_dev_extension/selection_provenance_revalidated_v3_final.json`
(SHA `b35bb5fc...`).  This amendment is explicitly not described as a new
pre-registration.  It also records the accurate chronology: no selected CA-OPEX
or adapter confirmation output had been read, but external baseline arms on the
same confirmation seed identifiers already existed and had been observed, so no
global blindness claim is made.

The first amortized inverse-residual adapter is not promoted on this evidence.
On the same matched development block it scored `22.0069`, below its own inverse
baseline (`24.8733`), despite positive results on an earlier development block.
This exposes seed/block sensitivity rather than a confirmation win.  A single
mechanism-targeted replacement was therefore tested: distill the strong per-state
channel-aware OPEX commands into the same bounded residual network at equal Q-row
budget.  The result below partially recovers the teacher signal, but CA-OPEX
remains the primary method; the first direct adapter remains a failed
amortization attempt rather than being hidden or relabeled.

The targeted teacher-distillation diagnostic is now complete and remains
development-only.  A 128x2 residual student trained for 5,000 updates in 73.6 s
to imitate fixed `K=8,T=2,eta=.1,delta=.25` CA-OPEX commands.  On the same ten
matched development episodes it scored `31.0162`, versus `24.8733` for the
inverse anchor and `38.7645` for the teacher, recovering `44.22%` of the
teacher's mean-return improvement with zero critic rows at deployment (training
used 20.48M teacher Q1 input rows).  The frozen single-use runner stopped only
after training because two expected split hashes were manually transcribed
incorrectly: the first failing train-index declaration had 65 characters
(`...71ff477...` versus `...71f477...`), and the next audit-index declaration
was also wrong (`e122...` versus deterministic `e127...`).  Training was not
rerun.  Because the driver stopped before evaluation, one diagnostic evaluation
was then written once; the separate post-training recovery script did not
reevaluate, but recomputed the split from the source dataset and verified every
frozen artifact before writing comparison v2 (SHA `3c0b15f2...`).
This positive amortization signal is explicitly PA-RL-like and is not promoted
as the core novelty or as independent confirmation evidence.

## 2026-09-14 route that produced the final method: calibrated value-aware channel compensation

Training a channel-marginalized TD3+BC policy from scratch did not provide the
required signal.  At 25k equal updates for a source `beta=1` dataset and target
`beta=1.25`, resampled-antithetic `K=2` TD3+BC scored `13.15` and its HUBL version
`16.57` over 50 independent noisy episodes.  The direct executed-action HUBL
control scored `12.62`, but the causally cleaner executed-critic/command-BC HUBL
control scored `20.35`.  Thus bare marginalization is not the retained method;
it is a completed direct competitor/ablation.

The strongest current non-learned signal is post-hoc compensation of a strong
executed-action HUBL policy.  With no target rewards or target transitions, the
analytic generalized inverse of the clipped-Uniform channel mean scored
`22.55 +/- 15.32` over 50 target-`beta=1.25` episodes, compared with `12.62` for
identity commands and `17.05` for a hand-selected `1.1` scale.  A fair, fixed
development grid over scales `1.2/1.4/1.6` gave `25.86/21.12/25.19` on the ten
predeclared development episodes; `1.2` is now frozen as the selected scalar
control.  It was subsequently evaluated without retuning in the formal block,
where it scored `21.2931/16.6657` for base seeds 1/10.

The retained hypothesis is that mean inversion preserves the desired mean physical
action, but misses action-value curvature: in general
`Q(s,E[a]) != E[Q(s,a)]`.  Starting from the generalized inverse `u0`, a bounded
small residual adapter `delta_psi(s)` was designed to maximize a common-random-number,
antithetic estimate of `E_epsilon[Q_e(s,clip(u0+delta+epsilon))]`, while the base
actor and physical-action critic remain frozen in the first diagnostic.  The
decisive controls are inverse-only, direct residual of equal capacity around the
unmodified actor, and the selected scalar multiplier.  The method is not a new
generic Q-guided perturbation: PLAS+P and PA-RL are close structural competitors.
Its potentially defensible distinction is calibrated stochastic actuator-channel
inversion plus distributional value correction on a frozen offline policy.

The main unresolved risk remains continuation mismatch.  A critic trained with the
source/clean continuation may reward a one-step channel perturbation without
representing the policy under persistent target noise.  A frozen-Q adapter was
therefore treated as a diagnostic and required to improve real
closed-loop return without relying on held-out Q gain.  The later formal audit
showed that larger frozen-Q proxy gains can correspond to worse rollout return;
channel-consistent fitted-Q remains a future alternative if broader-channel
tests invalidate the frozen source critic.

The pre-split development screen now supplies a positive mechanism signal, but
is not confirmation.  With the fixed 512-pair estimate, `K=8`, 5k updates,
`delta_max=.25`, and the predeclared penalties `{.01,.1}`, the inverse-residual
`penalty=.01` version scored `32.85` on the ten development episodes versus
`25.50` for its exact learned-beta inverse baseline, `25.86` for the selected
scalar, and `23.22` for the equal-capacity direct residual.  On the previously
used 50-episode development block it scored `33.92 +/- 21.82`, versus
`19.69 +/- 11.17` for the paired learned-beta inverse and `24.16 +/- 18.43` for
direct residual; the adapted-minus-inverse raw-return difference was `+653.0`
with 36/50 positive pairs.  A fresh-`K=64` source-state audit found mean
min-twin gain `+0.2112` (93.8% positive) and only `+0.0101` mean disagreement
change.  This audit used fresh channel noise but not a strictly held-out state
split, because the first trainer sampled from all source observations.  The
result is therefore retained as development evidence only.  Before freezing,
the trainer/auditor was changed to a recorded disjoint state split.  The formal
K=64 audit then covered 100,060 held-out source states and is summarized in the
final package.

The target channel can be estimated from reward-free `(command, executed_action)`
pairs.  A censored clipped-Uniform MLE at true `beta=1.25` produced estimates
`1.24779`, `1.24989`, and `1.24950` from 128, 512, and 2048 pairs respectively.
The completed main experiment used the 512-pair estimate `1.2498949`.
Because this endpoint estimator is non-regular, the current percentile-bootstrap
interval under-covers the true boundary and is not used as a calibrated confidence
claim.

## 2026-09-14 route decision: stop realized ERR-HUBL before training

The clean-room ERR-HUBL implementation and its alignment tests are complete, but a
full-dataset mechanism audit showed that the proposed signal has almost no
information beyond the already-falsified horizon schedule.  At `c=.01`, the
transition-wise residual and horizon lambda schedules have correlation `0.98135`,
mean absolute difference `0.00635`, and exactly matched mean `0.391843`.  For
remaining horizons 200--499, residual `D` has mean/std `50.53/3.11`, while the
deterministic horizon proxy is `49.96/0.26`.  This agrees with the analytic result
that normalizing homogeneous IID action-noise energy by its dataset mean removes
the global noise scale and leaves `E[D|H]=sum gamma^(2j)`.  The matched-mean horizon
run had already scored `80.81/35.97` clean/persistent versus `79.41/41.23` for the
constant control.  Training ERR-HUBL would therefore repeat a negative comparison
with only a tiny perturbation, rather than test a genuinely distinct mechanism.

The route is stopped without spending confirmation seeds.  Its implementation and
36 passing server tests are retained as a transparent negative ablation.  The
additional scientific concern is outcome-dependent weighting: the same realized
future disturbance generates both `D` and the MC return, so the method cannot be
described as an unbiased conditional-variance estimator.

The then-active backup was **execution-channel-marginalized TD3+BC**.  For paired logs
`(s,u_command,a_exec,r,s')`, the critic remains grounded in the causal physical
action `Q(s,a_exec)`, behavior cloning uses `u_command`, and both actor improvement
and the next-state Bellman value integrate `Q(s, clip(u+epsilon))` over the known or
estimated execution channel.  This directly targets noisy deployment rather than
using command labels as if they caused the observed transition or learning an
executed-action policy that is perturbed a second time at deployment.  The closest
controls are executed/executed, commanded/commanded, executed/commanded without
marginalization, and an equal-compute one-sample channel estimator.  Novelty and
exact-collision checking were completed; MAMDP remains a required conceptual
citation, not something to omit.

### Archived channel-factorization hypothesis and development test

For a policy command `u=pi(s)` and physical execution
`a=g(u,epsilon)=clip(u+epsilon)`, define a physical-action critic and target-channel
command value

`Q_e(s,a) = r(s,a) + gamma E[V_K(s') | s,a]`,
`V_K(s) = E_{epsilon~K}[min_i Q_{e,i}(s,g(pi(s),epsilon))]`.

The implemented clipped-double default first averages each twin over `K` and then
takes their minimum; expectation-of-the-pointwise-minimum is retained only as an
extra-pessimism ablation.  The actor maximizes the channel-marginalized first twin
while its BC term imitates logged commands.  The decisive comparison was against
the already completed executed-critic/command-BC baseline, which uses identical
data and networks but does not integrate the deployment channel.  This was tested
as a development screen with train seed 0, fixed evaluation seeds, and `K=2`
antithetic points.  It was stopped after the target-beta-1.25 results reported
above: marginalized TD3+BC/HUBL scored `13.15/16.57`, while the direct
executed-critic/command-BC HUBL control reached `20.35`.  No untouched training
seeds were spent on this rejected branch.

## Preserved prior directions

1. Direction 2, computation-aware diffusion path residual correction (`README.md`, `research_report.md`, `draft_experiment/`): stopped after exact/toy and RAM-style screens failed equal-budget gates.
2. Direction 4, BRIDGE-RRL (`direction4_research/`): stopped because the finite-sample certificate was vacuous, the useful marginalized estimator lacked a nuisance-bias certificate, and the main factorization overlapped options-OPE.
3. Remote `sampler_robust_20260912`: multiple Diffusion Policy sampler/residual variants were explored. The latest state explicitly says `METHOD_WEAK`, not `PAPER_CORE_READY`. Its checkpoints, Robomimic wrappers, raw trajectories, and exact-reproduction utilities are reusable; its development seeds must not be relabeled as independent confirmation.

## Verified resources

- GPU: NVIDIA GeForce RTX 2080 Ti; 22,528 MiB total, driver 580.119.02, CUDA driver API 13.0. The final 2026-09-15 check reported 0 MiB used, 0% utilization, and no compute process.
- CPU/RAM: host exposes 56 logical CPUs (2 x 14 cores with SMT), 251 GiB RAM, no swap. Service banner advertises 12 CPU cores / 58 GB, so runs will use conservative thread limits.
- Storage: the final 2026-09-15 check reports `/` as 30 GiB with 28 GiB free and `/root/rivermind-data` as 99 GiB with 50 GiB free. The data volume was only 49 GiB and full at initial inspection, so its later expansion is recorded rather than assumed retroactively.
- Runtime: Ubuntu 20.04; `/root/hubl_backup_env/bin/python` is Python 3.10.13 with PyTorch 2.1.0, NumPy 1.26, SciPy 1.15.3, h5py 3.16, Gymnasium 0.29.1, MuJoCo 3.13, and pytest 8.4.2. No standalone `nvcc` is installed.
- Existing tasks: no GPU research job; only JupyterLab/code-server infrastructure.

## Candidate decision

Focused searches produced seven concrete candidates: (1) target-preserving gradient-optimal domain allocation; (2) uncertainty-calibrated HUBL for stochastic offline RL; (3) conditional-ESS IQL extraction; (4) reliable/controllable latent metrics for frozen world-model planning; (5) action-spectrum-matched diffusion noising; (6) hard counterfactual conditioning for diffusion policies; and (7) frozen few-step diffusion prior calibration. Candidate 7 was the first implementation target, but exact collision checking found that scalar initial-noise calibration is already covered by Efficient Diffusion Policies and is closely surrounded by DDSS, LD3, Golden Ticket, AYS, RTI-DP, and other frozen-sampler methods. A defensible structured-covariance version would require beating zero/scalar noise and schedule calibration, with uncertain closed-loop transfer. It is therefore retained only as a backup, not pursued by sunk cost.

The first selected direction, **GOVA-DR: Gradient-Optimal Variance Allocation for Domain Randomization**, and the learned cross-fitted Q/V variant have been stopped. The pure horizon-only HUBL branch and its realized-residual refinement have now also failed their mechanism screens. Bare **execution-channel-marginalized offline control** was implemented and fairly tested, but did not beat the direct dual-action HUBL control. The final main method is **unamortized CA-OPEX**; the calibrated inverse-residual adapter is a secondary amortization/mechanism control. Existing MAMDP, SlateQ, PLAS+P, PA-RL, UAN, GARAT, mediator, and action-projection work make broad novelty claims untenable; the retained claim concerns low-data actuator-channel calibration and post-hoc expected-value correction of a frozen physical-action policy.

## Archived HUBL hypothesis and evidence

HUBL treats a long realized Monte-Carlo suffix as no less reliable than a short one, and its rank version changes trust according to trajectory desirability rather than target uncertainty. In the special case of independent unbiased MC/bootstrap errors with `Var(e_B)=tau^2` and `Var(e_MC)=v0+sigma^2 S(H)`, inverse-MSE blending gives
`lambda_t = alpha / (1 + c S(H_t))`, `S(H)=sum_{j=0}^{H-1} gamma^(2j)`, `alpha=tau^2/(tau^2+v0)`, and `c=sigma^2/(tau^2+v0)`. Here `H_t` is the number of rewards in the next-state suffix. Under actuator noise the assumptions are not exact, so this is called a cheap horizon reliability prior rather than calibrated variance. The implementation changes neither data, model size, update count, nor evaluation calls.

If this mechanism is correct, horizon weighting should preserve the robust-deployment gain already seen from Monte-Carlo blending while recovering clean performance relative to constant HUBL. It must beat an **equal-mean constant lambda** control; otherwise any gain is explained by changing average heuristic strength rather than allocating trust by reliability. Raw rank-HUBL is the closest published control. Development uses training seed 0 and fixed evaluation seeds; only after fixing `alpha,c` will training seeds 1 and 10 and untouched evaluation seeds be used for confirmation.

The first alpha=1 screen showed that an aggressive schedule reaching lambda=1 near terminals is not sufficient: at 50k updates and 50 paired evaluation episodes, `c=.02/.04/.08` produced clean/persistent scores `79.36/36.23`, `78.04/33.87`, and `79.88/33.58`. Equal-mean constants for the first two were `78.56/33.04` and `79.41/41.23`; thus `c=.02` weakly beats its exact matched mean, while `c=.04` clearly fails, and the best global constant still dominates. This is evidence against freezing the alpha=1 version, not a result to hide.

The state-restoration audit independently found large horizon growth but invalidated the IID interpretation: 48 restored anchors all passed one-step replay, beta=1 conditional return variance rose from `0.01` at H=1 to `425.5` at H=200, correlation with `S(H)` was `0.844`, but geometric-fit R2 was only `0.643` and off-diagonal reward covariance contributed `95.1%` of H=200 variance. Based on that evidence, the next and final development modification uses nonzero base MC uncertainty (`alpha<1`) and a gentler schedule. It sets `c=.01` and `alpha=.5524354153`, chosen algebraically so its transition-weighted mean is exactly the already-tested best constant `0.3918433189`; this is a one-run structural comparison, not a new grid.

That capped, equal-mean schedule did **not** dominate the constant: its 50-episode clean/persistent result was `80.81/35.97` versus `79.41/41.23` for constant `lambda=.391843`. A separately motivated rank-times-horizon product also fell to `77.41/27.42` in the 10-episode development evaluation at 50k. The horizon-only hypothesis is therefore stopped rather than tuned further. Its code and negative controls remain useful ablations.

ERR-HUBL replaces the homoscedastic length proxy by the actually observed applied disturbance. For raw logged transition `k`, let `e_k=mean_dim[(a_exec,k-u_cmd,k)^2]` and normalize by its full-dataset mean. For the HUBL next-state MC suffix, compute `D_{t+1}=sum_{k=t+1}^{T-1} gamma^{2(k-t-1)} e_k/E[e]` and set `lambda_t=alpha/(1+c D_{t+1})`. The dropped timeout transition is retained inside this suffix exactly as it is for `raw_mc_next`. Under homoscedastic perturbations, `E[D|H]=S(H)`, so this is a trajectory-specific generalization of the falsified horizon proxy, not an unrelated added module. It uses no learned model and no additional training or inference calls. The decisive controls are exact-mean constant, shuffled residual lambdas, horizon-only lambdas, and commanded-action TD3+BC/HUBL.

## Completed GOVA-DR falsification and reusable artifacts

- The clean implementation remains in local `tpadr/` and remote `/root/tpadr_research_20260914`; it contains target-preserving estimators, PPO, diagnostics, raw result aggregation, and tests.
- Fixed-policy development diagnostic, 16 Pendulum domain strata, 64 estimator repetitions and identical 64-trajectory budgets: gradient-trace Neyman component MSE was `1.060x` equal stratification; return-Neyman was `0.945x`. The proposed estimator therefore did not beat the simple proxy.
- Closed-loop development, Pendulum mass range 0.25--2.5, 720k environment steps, seed 1301: equal final mean/worst-decile return `-233.1/-676.4`; GOVA `-259.1/-726.9`; IID `-250.6/-705.6`; return-Neyman `-239.2/-677.4`. GOVA lost both primary comparisons.
- A final continuous-within-bin control-variate diagnostic (4 strata, independent 256/stratum pilot, 512/stratum reference, 64 repetitions) found CV-only MSE `1.063x` equal. Joint CV+allocation reached `0.888x`, but raw gradient allocation was already `0.884x`; hence the CV added no measurable benefit. No runs were excluded.
- Novelty audit found the allocation formula directly covered by Salmani et al. (2025 arXiv) and closely instantiated in 2026 rollout-allocation work. Together with the negative closed-loop evidence, this is a scientific rather than merely engineering reason to stop.

## Current implementation and data status

- Isolated server environment: `/root/tpadr_env`; Gymnasium 0.29.1, MuJoCo 3.13.0, PyTorch 2.1.0; headless step tests passed for Pendulum-v1, Hopper-v4, HalfCheetah-v4, and Walker2d-v4.
- Official HUBL ICLR 2024 paper, supplement, and source were inspected. Its stochastic Walker2d appendix reports rank-HUBL degrading under strong action noise and explicitly notes sensitivity to heuristic estimation; the relabeling equations and preprocessing details have been recovered.
- Official D4RL Walker2d-medium-v2 is validated at `/root/hubl_backup_data/walker2d_medium-v2.hdf5`: 1,000,000 transitions, 222 MiB, SHA-256 `cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda`. It contains the SAC behavior-policy weights, qpos/qvel, action log-probabilities, and next observations. Hugging Face was reached through `hf-mirror.com`; no incomplete file is in use.
- The embedded tanh-Gaussian SAC policy was reconstructed and checked on 10,000 logged transitions: log-probability MAE `2.55e-4`, RMSE `0.00707`, Pearson `0.999998`. A 1,000-step Walker2d-v4 action-noise collector smoke passed and saved the exact executed action/noise convention.
- The full 1,000,000-transition `beta=1` action-noise development dataset completed at `/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5` (SHA-256 `159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a`). It contains 3,254 complete episodes plus one collector-truncated segment, was collected at 1,609.7 transitions/s, and has executed-versus-command action RMSE `0.4634` with a `34.45%` action clipping rate. The first launch was deliberately restarted after 70k unsaved steps because unconstrained PyTorch CPU threads were starving concurrent training; the replacement used one BLAS/OpenMP thread. This follows HUBL's `clip(a + beta * Uniform[-1,1])` mechanism but uses Gymnasium Walker2d-v4/MuJoCo 3.13 rather than legacy Walker2d-v2/MuJoCo 2.1; that difference remains explicit.
- Clean-room Gymnasium/PyTorch code is under `hubl_backup/`; upstream supplement has no obvious license and is retained only as a formula/reference snapshot. The final server suite passes 232 tests, retained with provenance in `results/final_validation_20260915.log`. The clean IQL 100k-update development sanity run completed with a final 10-episode normalized score of `21.93`; it is an undertrained sanity result, not a paper comparison. On the action-noise dataset, equal-budget seed-0 TD3+BC completed 100k updates in 963 s and HUBL-constant in 991 s. A paired independent 50-episode evaluation with identical environment/noise seeds gives TD3+BC `86.69 +/- 0.31` clean and `15.16 +/- 5.14` under persistent beta=1 noise, versus HUBL-constant `76.23 +/- 0.05` clean and `35.89 +/- 20.91` persistent-noise score. The paired persistent-noise raw-return difference is `+951.5` (episode-level 95% t interval `[661.8, 1241.3]`, 40/50 positive pairs), but this is still one training seed and hence an exploratory signal. The historical HUBL-rank launch is no longer active and is not part of the final CA-OPEX evidence.
- The cross-fitted trajectory mechanism audit passed seven server tests. On a synthetic 160-trajectory audit, OOF mean improved noisy-return MSE `65.95 -> 8.54`, Spearman `0.656 -> 0.966`, and top-20 precision `0.531 -> 0.875`; LCB was worse than mean-only. On real Walker2d, OOF mean was worse at reward-noise sigma 1, but better at sigma 3 (MSE `449.3 -> 219.5`, Spearman `0.794 -> 0.873`) and sigma 10 (`4992.1 -> 594.6`, `0.453 -> 0.596`). Thus denoising has a high-noise boundary, while the LCB uncertainty claim is not supported.
- Novelty checking found the ICASSP 2026 PHQL/Progressive Heuristic Blending paper already uses learnable heuristic state-value estimates; Adaptive TD already uses learned MC-return uncertainty to select TD versus MC, and ESPER addresses lucky realized returns. The cross-fitted `V/Q` branch was tested and stopped; its `Q-V` interpretation is also closely related to ACT (AAAI 2024). It is not the final contribution.
- A deterministic replay audit reproduced all logged rewards and states for 256/256 audited trajectories, but the proposed open-loop clean-action counterfactual terminated early in 251/256 trajectories and had only about `0.029` Spearman correlation with logged return. It is therefore recorded as an invalid mechanism label and will not be used for selection or claims. Raw audit data is in `hubl_backup/results/action_noise_dev/`.
- The strict episode-level OOF Q/V audit is negative and the branch is stopped: ridge and MLP episode-level `Q` and `V` ranks correlate at `0.999947` and `0.999939`; raw HUBL return has clean-return Spearman `0.9344`, versus MLP `V=0.8515` and `Q-V=0.1738`. No favorable subset was selected. Learned V is retained only as a PHQL-like direct control, not as the proposed method.
- A paired command/execution action split was tested only as a protocol ablation. Langlois and Everitt's AAAI 2021 Modified-Action MDP already covers the core command-versus-execution semantics, and an executed-action critic is aligned only when the deployment target is the clean/unmodified underlying MDP. Persistent actuator noise instead requires marginalizing the execution channel. The simple split is not the final contribution.

## Continuation status and bounded follow-up

The required Walker2d scale-up and Hopper-v4 cross-task replication are
complete, raw-record based, and independently verified.  The new server test
log `results/final_validation_takeover_20260915.log` records `55 passed` and
exit code 0; the pre-existing formal validation log remains immutable at
`232 passed`.

The first root-level full-suite collection attempt is preserved at
`results/final_validation_takeover_full_20260915.log` (two collection errors
from a duplicate test-module path; SHA-256
`6d0278d3b4fe91fa42e7efd7c53b1c19a6739138b05fe20b692a4ba43062ec0a`).  Its
wrapper's exit marker was invalidly masked by a trailing timestamp command, so
it is not a pass record.  The misplaced duplicate was moved, not deleted, to
`results/diagnostic_quarantine_20260915/`.  The corrected canonical full-suite
run from `code/` is
`results/final_validation_takeover_full_v2_20260915.log`: `251 passed`, exit
code 0, SHA-256
`a574edb1c2ddc1244c0f9f70635b57cc5486992cbea53f73b1c4e1de8ff3cec8`.

The canonical new delivery is `results/ca_opex_scaleup_bundle_v3.tgz` with
275 payload files, including checkpoints, raw records, calibration artifacts,
code, tests, diagnostics, and logs.  Its SHA-256 is
`e2f889d2e69e7a12d964aa7a185dd797bce3d3ceb4f3359c9baad61e2e47766b`;
source HDF5 files are intentionally excluded and formal v3 remains separate.
The earlier v1/v2 archives are preserved as noncanonical packaging snapshots.

The only active follow-up is the optional beta-severity grid.  It is a real
single-GPU chain, but incomplete and therefore not part of completed results;
after all 12 units finish it must be aggregated and independently verified.
HalfCheetah, Gaussian, biased, delayed, correlated, and state-dependent
channels remain unrun scientific follow-ups and are not claims.
