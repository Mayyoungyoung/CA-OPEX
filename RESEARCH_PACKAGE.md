# CA-OPEX research package

Status: **paper-core prototype complete at a deliberately narrow scope; Walker2d five-seed expansion complete; Hopper-v4 cross-task replication complete with a negative sub-result on the value-correction stage**.  The
selected controller was frozen and then evaluated on two predeclared base-policy
training checkpoints, with 50 paired rollout cases per checkpoint, and the
Walker2d base-policy seed set was then expanded to five independent checkpoints
(seeds 1/10 formal blocks plus new scale-up seeds 2/3/4) with training-seed-level
statistics.  Strict
fail-closed aggregation completed with 10/10 adapter runs, 14 external controls,
56 comparisons, and no missing raw record.  A later fresh-rollout mechanism
audit also compared the complete controller with a separately tuned nominal-Q
controller at the same
per-decision neural evaluation budget.  The Hopper-v4 replication confirms that
channel calibration plus mean inversion transfers (complete beats original OPEX
on 3/3 seeds) but finds no detectable incremental gain from the frozen-critic
expected-value correction stage over its own inverse anchor on that
task.  Neither experiment is described as globally
blind: several formal baseline arms had already been observed before the
selected-method outputs, and the supplemental audit was designed after reading
the formal result.

The 2026-09-15 takeover audit found that the requested Walker2d seeds 2/3/4
and the Hopper-v4 cross-task replication had already completed on the server.
The seed-2 dry-run and real launch records, raw controls, checkpoints, and
stage logs were checked without rerunning the create-only seed-2 root.  New
independent verifiers reproduced the Walker result with 96 checks and zero
failures and the Hopper result with 57 checks and zero failures.  Formal v3 and
the canonical aggregates remain untouched.  The optional beta-severity chain
is genuinely running sequentially on the single GPU (chain PID 18625); its
partial output is not treated as a result.

The takeover verifier files are
`results/scaleup/takeover_verification_20260915/multiseed_verify.json`
(`df0f7cf335b92b56a9a9ae4fd023858672c38f951e893dffbf1af172ebeec0b1`) and
`results/crosstask_hopper/takeover_verification_20260915/multiseed_verify.json`
(`5c2ee75923c84447cc5789f17ddb6725274106f562f8e6e5c66472044a6eca4e`).

## 1. Research problem and core idea

### Problem

An offline controller is often trained as if its action were executed exactly.
At deployment, actuator electronics, low-level control, latency, clipping, or
wear can instead induce an execution channel

\[
  a_{\mathrm{exec}}=g_\beta(u,\epsilon)
  =\operatorname{clip}(u+\epsilon,-1,1),\qquad
  \epsilon_i\overset{iid}{\sim}\mathcal U[-\beta,\beta],
\]

where `u` is the high-level command and `a_exec` is the physical action.  A
frozen offline policy may therefore output a sensible desired physical action
but the wrong command distribution.  Retraining is undesirable, and in the
target domain we assume access only to a small reward-free and state-free set of
`(command, executed_action)` pairs.  “State-free” refers only to calibration:
the deployed controller still observes its ordinary current state.

### CA-OPEX

**Channel-Aware OPEX (CA-OPEX)** combines a calibrated channel inverse with
critic-guided, test-time command extraction:

1. Fit `beta` from 512 `(u, a_exec)` pairs with the likelihood of the censored
   clipped-Uniform channel.  No target reward, task state, next state, or target
   transition is used.
2. Let `a* = pi0(s)` be the frozen actor's desired physical action.  Compute an
   inverse-mean command

   \[
     u_0=m_{\hat\beta}^{\dagger}(a^*),\qquad
     m_\beta(u)=\mathbb E_\epsilon[g_\beta(u,\epsilon)].
   \]

   The expectation is analytic.  For

   \[
   F(x)=\begin{cases}
   -x-\tfrac12,&x\le -1\\
   \tfrac12x^2,&-1<x<1\\
   x-\tfrac12,&x\ge1,
   \end{cases}
   \]

   `m_beta(u) = [F(u+beta)-F(u-beta)]/(2 beta)` elementwise; its monotone
   generalized inverse is found by 48 deterministic bisection steps.
3. Starting from `u0`, take `T` projected ascent steps through the calibrated
   physical channel:

   \[
   \widehat J_s(u)=\frac1K\sum_{k=1}^{K}
       Q_1\!\left(s,g_{\hat\beta}(u,\epsilon_k)\right),
   \]
   \[
   u_{t+1}=\Pi_{[-1,1]^d\cap B_\infty(u_0,\delta)}
       \left[u_t+\eta\nabla_u\widehat J_s(u_t)\right].
   \]

   The implementation uses fresh antithetic Uniform samples at every gradient
   step.  The selected inverse-anchor configuration is `K=8`, `T=2`,
   `delta=0.25`, `eta=0.1`.  Actor and critic parameters never change.

In plain language, mean inversion first compensates the actuator's systematic
clipping/noise bias.  Expected-Q ascent then corrects what mean matching cannot:
in general `Q(s, E[a]) != E[Q(s,a)]`.

### Testable hypothesis

The hypothesis fixed before confirmation is:

> Under strong clipped actuator noise, ordinary OPEX optimizes a nominal command
> as though it were the physical action.  Mean inversion should move the command
> into the right physical-action basin, and channel-marginalized Q gradients
> should improve the remaining value curvature.  If this mechanism is real, the
> complete method should beat inverse-only, ordinary OPEX, and an equal-Q-call
> channel-aware identity-anchor control—not merely show a lower training loss.

The primary outcome is closed-loop Walker2d return at target `beta=1.25`.
Secondary mechanism outcomes are paired return differences, clipping/boundary
statistics, and Q-call cost.  Additional parameters, target rewards, target
transitions, or extra rollout retries are not granted to the complete method.

## 2. Innovation positioning and direct competitors

The scoped contribution is the combination under this target-information
regime, not any individual ingredient.  Test-time Q ascent, action modification
MDPs, inverse dynamics, bounded residual correction, and robust action-noise
training all predate this work.  A focused primary-source search through
2026-09-15 found no work that directly combines action-pair-only stochastic
channel identification, clipped mean inversion, and expected-physical-action Q
extraction on a frozen offline actor/critic.  This is a scoped literature result,
not a priority or “first method” claim.

| Work | Verified status and links | Core mechanism | Substantive difference from CA-OPEX | Locally reproduced? |
|---|---|---|---|---|
| Is Value Learning Really the Main Bottleneck in Offline RL? | NeurIPS 2024 main conference; [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/8ffb4e3118280a66b192b6f06e0e2596-Paper-Conference.pdf) | OPEX performs a test-time frozen-Q action gradient step. | The closest ancestor.  OPEX treats its optimized action as the executed action; it neither identifies nor integrates an execution channel. | Its published benchmark is not reproduced; its exact one-step structure is implemented as a local direct control. |
| Improving Offline RL by Blending Heuristics | ICLR 2024; [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b40cd8bb212ad8ad0a8c8a43c4da5f0b-Abstract-Conference.html), [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/b40cd8bb212ad8ad0a8c8a43c4da5f0b-Paper-Conference.pdf) | HUBL blends MC heuristics into bootstrapped offline RL; its appendix also retrains on action-noise data. | HUBL needs complete reward-bearing transitions and retraining.  It is the local base learner, not the deployment adaptation. | Clean-room HUBL+TD3BC base is trained locally; settings differ from the paper and numbers are not cross-paper comparisons. |
| PLAS: Latent Action Space for Offline Reinforcement Learning | CoRL 2020, PMLR volume published 2021; [PMLR](https://proceedings.mlr.press/v155/zhou21b.html), [code](https://github.com/Wenxuan-Zhou/PLAS) | PLAS+P learns an offline, bounded action perturbation layer. | No target actuator channel, pair calibration, or test-time expected-channel search. | No. |
| Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone | arXiv 2024 and ICLR 2025 workshop, not ICLR main conference; [arXiv](https://arxiv.org/abs/2412.06685), [code](https://github.com/MaxSobolMark/PolicyAgnosticRL) | Q reranking/local action optimization followed by supervised distillation. | Generic action optimization/distillation is already covered; it does not identify a command/execution channel.  Our distilled student is therefore only a PA-RL-like engineering control. | No. |
| SPAR: Support-Preserving Action Rectification | ICML 2026 poster/PMLR 306; [ICML](https://icml.cc/virtual/2026/poster/63368), [OpenReview](https://openreview.net/forum?id=XiaqvkSnWk), [arXiv](https://arxiv.org/abs/2605.27877) | Offline local residual rectification around a frozen BC anchor. | The most dangerous structural competitor.  SPAR trains from full reward-bearing transitions and has no calibrated target actuator channel or test-time channel expectation. | No public author code found by the audit; not reproduced. |
| Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model | ICLR 2025; [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html), [code](https://github.com/tongzhoumu/policy_decorator) | Online SAC learns a bounded residual around a frozen imitation policy. | Requires target task rewards and online interaction; CA-OPEX freezes all weights and calibrates only an action channel. | No. |
| How RL Agents Behave When Their Actions Are Modified | AAAI 2021 technical track; [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17378), [code](https://github.com/edlanglois/mamdp) | Modified-Action MDP separates policy-specified and executed actions. | Supplies the general problem formalism, not pair estimation, mean inversion, or frozen-Q command extraction. | No. |
| Action Robust Reinforcement Learning and Applications in Continuous Control | ICML 2019; [PMLR](https://proceedings.mlr.press/v97/tessler19a.html), [code](https://github.com/tesslerc/ActionRobustRL) | Trains policies against probabilistic/adversarial action perturbations. | Minimax retraining under a specified uncertainty set, rather than reward-free identification and risk-neutral post-hoc compensation. | No. |
| Transfer from Simulation to Real World through Learning Deep Inverse Dynamics Model / Grounded Action Transformation / FADA | Christiano et al. [arXiv 2016](https://arxiv.org/abs/1610.03518); GAT [AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/11044); FADA [arXiv 2026](https://arxiv.org/abs/2606.28476), [code](https://github.com/LeCAR-Lab/FADA), proceedings status not yet verified | Learns target inverse/forward dynamics from target state or observation transitions. | Much broader dynamics adaptation, but requires target trajectories.  CA-OPEX's inverse is only a state-independent actuator mean inverse and uses action pairs. | No. |
| Residual Policy Learning / Residual RL for Robot Control | [arXiv 2018](https://arxiv.org/abs/1812.06298), [code](https://github.com/k-r-allen/residual-policy-learning); ICRA 2019 [paper](https://ieeexplore.ieee.org/document/8794127/) | Learns additive residual control with online task reward. | Confirms residual correction is not novel; target reward and interaction requirements differ. | No. |

The two main reviewer challenges are therefore addressed directly in the local
experiment: “just OPEX through noise” is tested by ordinary OPEX and the
channel-aware identity anchor; “just inverse compensation” is tested by the
inverse-only arm.

## 3. Experimental setting and evidence labels

### Task and data

- Environment: Gymnasium `Walker2d-v4`, MuJoCo 3.13.
- Source log: one million transitions collected with additive clipped Uniform
  command noise `beta=1`; commands and executed actions are stored separately.
- Base learner: clean-room HUBL+TD3BC with the critic and actor trained on
  executed physical actions, 25k updates.
- Target deployment channel: `beta=1.25`.
- Calibration: 512 synthetic software-channel pairs, selected without
  replacement from source commands and re-executed through the target channel.
  The pair file contains no state, transition, or reward field.  True beta is
  known only to construct/audit the simulator; the method uses MLE
  `beta_hat=1.2498948872`.
- Normalization: D4RL-style references `(1.629008, 4592.3)` applied to
  `Walker2d-v4`.  Because the runtime and dataset are not the original D4RL-v2
  stack, normalized values are only comparable within this package.

The actuator shift is synthetic, not hardware evidence.  It preserves the core
causal distinction between command and executed action, but it does not establish
robustness to latency, state-dependent dynamics, sensor error, or a real robot.

### Evidence separation

- **Development:** training seed 0; environment seeds `28300..28309`; action-noise
  seeds `38300..38309`.  These ten episodes were used to choose one step size per
  anchor and cannot become confirmation evidence.
- **External baseline controls:** training seeds 1 and 10; 50 episodes on
  `39300..39349` / `49300..49349`.  These baseline arms existed and were observed
  before the selected-method confirmation.  They are fair fixed controls, but
  not blind outcomes.
- **Selected-method confirmation:** the selected CA-OPEX and adapter outputs on
  those 50-episode blocks did not exist and were not read during selection.  No
  global blindness claim is made.
- **Post-confirmation mechanism audit:** after reading the formal result, a
  nominal-Q `K=8,T=2` control was tuned only on the retained seed-0 development
  block and evaluated on fresh environment/action/gradient seeds
  `79300../89300../99300..`.  These seeds are disjoint from the canonical formal
  manifest.  This audit is useful fresh-rollout evidence, but it is not a second
  prospective or blind confirmation.  Its machine-readable evidence label is
  `post_confirmation_fresh_rollout_mechanism_audit`.

The boundary extension and its chronology are cryptographically retained.  The
original pre-evaluation protocol/selector/selection have hashes `cba9c405...`,
`22d3eb49...`, and `603d5a70...`.  A later audit noticed the selector imported an
unhashed validation helper.  A clearly post-evaluation, provenance-only v3
amendment hashes that dependency, checks the original chain and every one of the
ten immutable raw candidates, and reproduces the same choices.  It is not
described as a second pre-registration.

## 4. Completed preliminary results

### Development selection

All entries below use the same base seed 0 and ten paired development episodes.

| Controller | Anchor | Channel model in gradient | K x T | eta | Normalized score |
|---|---|---|---:|---:|---:|
| Inverse only | inverse mean | none | 0 | — | 24.8733 |
| Original-structure OPEX | identity | beta=0 | 1 x 1 | 0.1 | 20.3798 |
| CA-OPEX, identity control | identity | calibrated beta | 8 x 2 | **0.3** | 31.0982 |
| **CA-OPEX, complete** | inverse mean | calibrated beta | 8 x 2 | **0.1** | **38.7645** |

The complete method's paired raw-return improvement over inverse-only was
`+637.70`; it beat inverse-only in 6/10 episodes.  The stronger tuned identity
control is `7.6663` normalized points below the complete method.  This is a
promising mechanism signal, not a training-seed estimate.

The retained step-size curves are:

| eta | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---:|---:|---:|---:|---:|---:|
| inverse anchor | 22.9299 | 26.5033 | **38.7645** | 37.8107 | 20.0936 |
| identity anchor | 13.5830 | 13.8171 | 19.5429 | **31.0982** | 15.0658 |

Thus neither selected value is an untested upper boundary, and the direct
identity competitor received its own reasonable tuning opportunity.

### Preexisting 50-episode baseline controls

These are local values under the exact confirmation rollout seeds, but they were
observed before selected-method confirmation and are labeled accordingly.

| Controller | Train seed 1 | Train seed 10 | Mean across checkpoints |
|---|---:|---:|---:|
| Identity command | 12.4784 | 9.9889 | 11.2337 |
| Selected scalar `1.2 x actor` | 21.2931 | 16.6657 | 18.9794 |
| Exact inverse mean | 21.9933 | 18.4220 | **20.2076** |
| Direct dual HUBL | 16.0750 | 14.6310 | 15.3530 |

Each cell is one fixed trained checkpoint evaluated for 50 episodes; 50 rollout
episodes are not 50 independent training seeds.

### Zero-critic amortization diagnostic

A 128x2 residual student was trained for 5,000 updates to imitate frozen
CA-OPEX teacher commands on the source training-state split.  This is a PA-RL-like
engineering extension, not the primary novelty and not held-out confirmation.

| Development controller | Normalized score | Deployment Q1 rows/step |
|---|---:|---:|
| Inverse only | 24.8733 | 0 |
| Distilled CA-OPEX student | 31.0162 | 0 |
| CA-OPEX teacher | 38.7645 | 16 |

The student recovered `44.22%` of the teacher's mean-return gain over inverse.
Training took 73.6 s and consumed 20.48M teacher Q1 input rows.  The original
single-use driver finished training and then rejected two manually mistyped
split declarations: the first failing train-index hash was 65 characters
(`...71ff477...` instead of `...71f477...`), and its next audit-index hash was
also wrong (`e122...` instead of deterministic `e127...`).  Training was not
rerun.  The driver had stopped before evaluation, so one diagnostic evaluation
was then written once; `recover_opex_distillation_dev.py` did not reevaluate, but
recomputed the split from the raw HDF5, verified both original errors plus all
frozen artifacts, and published comparison v2 (SHA `3c0b15f2...`).  This
chronology and all failed/successful validation logs are retained.

### Frozen confirmation

The canonical manifest
(`57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807`)
was materialized before any selected CA-OPEX or adapter confirmation output
existed.  The first attempted manifest is retained separately: a freezer bug left
placeholder tokens in provenance-map *keys*, and the first actual OPEX runner
rejected it at preflight before producing any formal raw output.  The corrected
freezer checked both keys and values, produced the canonical manifest, and the
one-shot runners then completed.  This chronology is preserved rather than
silently rewriting the failed preflight.

Each cell below is the mean of 50 paired target-channel rollouts for one frozen
base-policy checkpoint.  The last column is only the descriptive mean of the two
checkpoints (`n_checkpoint=2`); the 100 rollout episodes are never pooled as 100
independent training repeats.

| Controller | Seed 1 raw / normalized | Seed 10 raw / normalized | Two-checkpoint descriptive mean raw / normalized |
|---|---:|---:|---:|
| **CA-OPEX, inverse anchor** | **1434.4251 / 31.2110** | **1273.2322 / 27.6997** | **1353.8286 / 29.4554** |
| CA-OPEX, tuned identity anchor | 991.4510 / 21.5616 | 1024.6537 / 22.2849 | 1008.0524 / 21.9232 |
| Original-structure OPEX | 851.1230 / 18.5048 | 744.9876 / 16.1928 | 798.0553 / 17.3488 |
| Calibrated inverse only | 1023.1147 / 22.2513 | 868.6581 / 18.8868 | 945.8864 / 20.5690 |

The six predeclared primary comparisons use episode-case pairing within each
fixed checkpoint.  Intervals are conditional on that checkpoint, not
training-seed confidence intervals.  Bootstrap intervals use 20,000 fixed-seed
paired resamples.

| Comparison | Checkpoint | Normalized delta | Paired t 95% CI | Paired bootstrap 95% CI | Positive cases |
|---|---:|---:|---:|---:|---:|
| inverse CA - identity CA | 1 | +9.6494 | [4.2674, 15.0315] | [4.5242, 14.9260] | 37/50 |
| inverse CA - identity CA | 10 | +5.4149 | [-0.0048, 10.8346] | [0.3739, 10.7785] | 30/50 |
| inverse CA - original OPEX | 1 | +12.7062 | [6.2728, 19.1397] | [6.4506, 18.8265] | 40/50 |
| inverse CA - original OPEX | 10 | +11.5069 | [6.4490, 16.5649] | [6.7717, 16.5506] | 39/50 |
| inverse CA - inverse only | 1 | +8.9597 | [2.4395, 15.4799] | [2.7519, 15.3169] | 29/50 |
| inverse CA - inverse only | 10 | +8.8130 | [3.5529, 14.0730] | [3.9303, 14.1152] | 30/50 |

Across the two checkpoints, the descriptive normalized margins are `+7.5322`
over equal-Q-call identity CA, `+12.1066` over original OPEX, and `+8.8863` over
inverse-only; all three comparisons are positive on 2/2 checkpoints.  The seed-10
inverse-versus-identity t interval very slightly crosses zero even though its
bootstrap interval does not, so the package does not claim that both interval
procedures reject zero in every row.

### Frozen learned-adapter controls

The same manifest also ran five bounded 128x2 residual-adapter variants for
5,000 updates per checkpoint.  These are mechanism and amortization controls,
not the main method.

| Adapter (target beta 1.25) | Seed 1 | Seed 10 | Two-checkpoint descriptive mean |
|---|---:|---:|---:|
| inverse anchor, sampled expected-Q | 31.0258 | 24.5847 | 27.8053 |
| identity anchor, narrow bound | 23.6273 | 17.5921 | 20.6097 |
| inverse anchor, Q at channel mean | 28.2760 | 27.9938 | **28.1349** |
| identity anchor, wide bound | 19.0192 | 14.4877 | 16.7535 |
| identity anchor, nominal beta-zero Q | 14.8570 | 14.2217 | 14.5394 |

Inverse sampled beats its own inverse-only baseline by `+8.7745/+5.6980`
normalized points and beats the narrow identity adapter by `+7.3985/+6.9927`.
It does **not** beat unamortized CA-OPEX: its margins are `-0.1852/-3.1150`.
The Q-at-mean minus sampled deltas are `-2.7498/+3.4091`, so these runs do not
support a sampled-estimator superiority claim.  Wide-minus-narrow is
`-4.6080/-3.1044`, supporting the tight inverse-centered trust region rather than
unrestricted frozen-critic exploitation.  For equal wide settings, nominal
beta-zero minus calibrated beta-1.25 is `-4.1622/-0.2660`; this is only weak
adapter-level evidence for explicit channel modeling, not yet a test-time
CA-OPEX single-factor result.

Target-specific adapters were also evaluated after removing the deployment
channel without recalibration.  Inverse-adapter clean deltas against its own
baseline were `-5.8008/+6.8809`; direct-adapter deltas were
`+2.3151/-11.6252`.  The signs disagree, so clean safety is not established.
Operationally a detected clean channel should select the beta-zero controller;
the package does not claim that a beta-1.25 adapter can be left on after the
channel disappears.

### Post-confirmation equal-neural-budget nominal-Q audit

This audit answers the largest compute confound left by the formal comparison.
All four online controllers use one actor row, `K=8,T=2`, 16 Q1 rows, and two
action-gradient backward calls per adapted decision.  Only the nominal-Q step
size was tuned, using the already designated seed-0 development block.  Its
complete curve was retained:

| nominal-Q eta | 0.01 | 0.03 | **0.1** | 0.3 | 1.0 |
|---:|---:|---:|---:|---:|---:|
| Development normalized score | 12.2047 | 13.1080 | **20.4215** | 17.2884 | 8.4517 |

The selected `eta=0.1` is an interior value.  No endpoint extension was
triggered.  The following are fresh 50-case holdout means; the same base-policy
checkpoints are used, but every rollout, actuator-noise, and gradient-sampling
seed is new relative to the formal block.

| Controller | Seed 1 raw / normalized | Seed 10 raw / normalized | Two-checkpoint descriptive normalized mean |
|---|---:|---:|---:|
| Nominal-Q identity, independently tuned `eta=0.1` | 1109.1342 / 24.1251 | 844.7063 / 18.3650 | 21.2451 |
| Nominal-Q identity, matched `eta=0.3` | 832.4493 / 18.0980 | 825.5588 / 17.9479 | 18.0230 |
| Calibrated expected-Q identity, `eta=0.3` | 1124.7048 / 24.4643 | 1149.0239 / 24.9941 | 24.7292 |
| **Complete calibrated inverse, `eta=0.1`** | **1477.2873 / 32.1447** | **1468.9783 / 31.9637** | **32.0542** |

Paired intervals below are conditional on one fixed checkpoint.  The final
column is a descriptive mean of two checkpoint deltas, not a training-seed
confidence interval.

| Comparison | Seed 1 delta; t / bootstrap 95% CI | Seed 10 delta; t / bootstrap 95% CI | Descriptive mean |
|---|---:|---:|---:|
| complete - tuned nominal | +8.0196; [1.8632, 14.1760] / [2.0119, 13.9627] | +13.5987; [6.9195, 20.2779] / [7.1412, 20.1965] | +10.8092 |
| complete - matched-eta nominal | +14.0467; [9.0627, 19.0307] / [9.3317, 19.0165] | +14.0158; [7.7329, 20.2987] / [8.1100, 20.2188] | +14.0313 |
| complete - calibrated identity | +7.6804; [2.6845, 12.6763] / [2.8775, 12.5152] | +6.9697; [-0.3331, 14.2724] / [-0.0171, 13.9268] | +7.3250 |
| calibrated identity - tuned nominal | +0.3392; [-4.9119, 5.5903] / [-4.7577, 5.4003] | +6.6290; [1.2403, 12.0178] / [1.4658, 11.8356] | +3.4841 |
| calibrated identity - matched-eta nominal | +6.3663; [1.9808, 10.7518] / [2.1497, 10.6120] | +7.0461; [1.5394, 12.5529] / [1.7683, 12.4262] | +6.7062 |

Thus the complete stack beats the strongest separately tuned nominal-Q control
on both fresh checkpoints while matching neural rows and backward calls.  The
fixed-identity, fixed-`eta/delta/K/T` last comparison is the cleanest isolated
evidence for explicit channel expectation: both interval types exclude zero on
both checkpoints.  Giving nominal-Q its own best step size makes that isolated
comparison inconclusive on seed 1 and positive on seed 10, so no universal
single-factor expectation claim is made.  Complete versus calibrated identity
changes anchor, trust-region radius, and selected step size together; it remains
a selected-stack comparison rather than an anchor-only ablation.

### Five-training-seed Walker2d expansion (2026-09-15)

Three new fail-closed runs (training seeds 2/3/4) each trained the identical
25k-update HUBL base policy and evaluated five controllers on one fresh 50-case
common-random-number block with no hyperparameter changes: complete CA-OPEX
(inverse anchor, K=8, T=2, eta=0.1, delta=0.25, calibrated), calibrated
identity CA (eta=0.3), equal-neural-budget nominal-Q (eta=0.1), original OPEX
(K=1,T=1), and calibrated inverse-only.  The five-seed aggregate combines these
with seeds 1/10 from the frozen formal block (and, for the nominal comparison,
the supplemental block).  All numbers below are recomputed from the raw
per-episode returns by
`aggregate_ca_opex_walker_multiseed.py` and independently re-derived by
`verify_ca_opex_walker_multiseed.py` (96 checks, zero deviations).

Per-seed normalized controller means (seeds 1/2/3/4/10):

| Controller | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 10 | Five-seed mean (sd) |
|---|---:|---:|---:|---:|---:|---:|
| **Complete CA-OPEX (inverse)** | 31.2110 | 33.2987 | 30.0550 | 38.7492 | 27.6997 | **32.2028 (4.1812)** |
| Calibrated identity CA | 21.5616 | 29.7354 | 30.2426 | 26.3909 | 22.2849 | 26.0431 (4.0496) |
| Tuned nominal-Q (K=8,T=2) | 24.1251 | 21.7648 | 22.7064 | 22.2581 | 18.3650 | 21.8439 (2.1347) |
| Original-structure OPEX | 18.5048 | 20.3752 | 22.9123 | 19.8767 | 16.1928 | 19.5724 (2.4725) |
| Calibrated inverse-only | 22.2513 | 21.3658 | 23.8563 | 23.9812 | 18.8868 | 22.0683 (2.0920) |
| Identity command | 12.4784 | 12.8100 | 13.4474 | 11.1038 | 9.9889 | 11.9657 (1.3983) |

Cross-seed paired differences (one mean per independent training seed, n=5;
normalized; the 250 rollout episodes are never pooled):

| Comparison | Per-seed deltas (1/2/3/4/10) | Mean | 95% t interval | Positive seeds |
|---|---|---:|---:|---:|
| complete - tuned nominal-Q | +8.02/+11.53/+7.35/+16.49/+13.60 | +11.3984 | [+6.6454, +16.1514] | 5/5 |
| complete - original OPEX | +12.71/+12.92/+7.14/+18.87/+11.51 | +12.6304 | [+7.4201, +17.8407] | 5/5 |
| complete - inverse-only | +8.96/+11.93/+6.20/+14.77/+8.81 | +10.1345 | [+6.0483, +14.2207] | 5/5 |
| complete - identity command | +18.73/+20.49/+16.61/+27.65/+17.71 | +20.2371 | [+14.7976, +25.6765] | 5/5 |
| complete - calibrated identity CA | +9.65/+3.56/-0.19/+12.36/+5.41 | +6.1597 | [+0.0080, +12.3114] | 4/5 |

The four primary comparisons hold on 5/5 independent base checkpoints with
training-seed t intervals excluding zero.  The complete-minus-calibrated-
identity comparison is the weakest: seed 3 is slightly negative, seed 2's
per-seed interval crosses zero, and the five-seed interval only barely excludes
zero (+0.0080 lower bound).  This is reported as seed-sensitivity of the
inverse-anchor advantage over a separately tuned channel-aware identity
controller, not as a uniform win.

Cost accounting: recorded base-policy training times are `266.7-274.5 s` per
seed (five-seed total `1350.9 s`); recorded rollout wall time across the five
controller arms on all five seeds totals `3521.8 s`; the 512-pair calibration
wall time is not recorded in the frozen calibration JSON.  Equal neural budget
remains "one actor row, 16 Q1 rows, two backward calls per adapted decision"
for all K=8/T=2 arms; inverse anchoring adds 48 scalar bisection steps and no
FLOP/wall-time equality is claimed.  Artifact hashes are recorded in
`RESEARCH_STATE.md`.

### Hopper-v4 cross-task replication (2026-09-15)

A fully carried-over replication was executed on Hopper-v4.  The source was the
official D4RL `hopper_medium-v2.hdf5` (SHA-256
`5bdf1bc4a713c82941de44633df669b36c89850b652a25985166796d25cf71a0`), whose
embedded SAC policy passes an exact log-probability audit (Pearson
`0.9999999983`).  A fresh 1M-transition beta-1 executed-action dataset was
collected (SHA-256
`19bff1e68edef8fdd3ef8b4911a74d0c898e7b81970ca73975b6a0c9f04b9ee1`), the
512-pair beta-1.25 calibration reproduced the construction value
(`beta_mle=1.2499450445175158`), and training seeds 2/3/4 each trained the
identical 25k HUBL base with five controllers on one 50-case block.  No
hyperparameter was retuned.  Normalization references (`-20.272305`,
`3234.3`) came from the official `d4rl/infos.py`; raw and normalized comparison
signs agree, so raw returns are primary.

Cross-seed results (n=3 training seeds; normalized deltas of complete CA-OPEX
minus the control):

| Comparison | Mean delta | 95% t interval | Positive seeds |
|---|---:|---:|---:|
| complete - original OPEX | +2.2476 | [+0.6936, +3.8016] | 3/3 |
| complete - tuned nominal-Q | +4.4682 | [-5.5541, +14.4906] | 2/3 |
| complete - calibrated identity CA | +9.6080 | [-8.8415, +28.0574] | 3/3 |
| complete - calibrated inverse-only | **-0.5719** | [-2.4089, +1.2652] | 1/3 |

An independent verifier recomputed the report from raw records (57 checks, zero
failures).  The calibrated-inversion part of the method transfers: complete
beats original OPEX on every seed with an interval excluding zero.  However,
the frozen-critic expected-value correction stage shows **no detectable
incremental gain over the bit-matched inverse anchor** on Hopper (mean
`-0.57`), in contrast to Walker2d (`+10.13` over five seeds).  A bounded
zero-GPU diagnostic using existing records: Hopper episodes end at 310--410
steps (`0/10` reach the 1000-step horizon in every seed) versus Walker
898--988 steps (`7-9/10` full-horizon), and the Hopper base policy is roughly
half the Walker normalized level (`33.6-39.5` vs `80.6` clean).  The
termination-dominated regime gives the one-step correction few effective steps
and makes near-terminal valuations survival-dominated; this is the leading
explanation for the non-transfer, ahead of environment/normalization bugs
(ruled out by the audit and sign consistency).  One aggregation engineering
fix is on record: the transformed aggregator initially inherited Walker's
pinned beta constant, failed the seed-2 aggregation step before writing any
output, and was corrected and re-run as a labeled retry; the failed log is
preserved.

The direct consequence for claims: the cross-task statement is limited to
channel calibration plus mean inversion (and the improvements over original
OPEX / identity / nominal pipelines), while the value-correction increment is
currently supported on Walker2d only, with an explicit Hopper counterexample
retained.  

### Walker2d channel-severity grid (running; no result claimed yet)

The optional frozen grid is running as 12 create-only units: beta values
`{0.5, 0.9, 1.1, 1.4}` crossed with checkpoints `{1, 2, 10}`, sequentially on
the single RTX 2080 Ti.  Each unit has its own 512-pair calibration and paired
50-case rollout block, with mutually exclusive calibration/rollout RNG blocks;
K/T/eta/delta are unchanged.  The chain was verified as active under PID
`18625`, with launch record
`results/betagrid_chain_20260915.launch.info` and progress log
`results/betagrid_chain_20260915.nohup.log`.  No beta-grid mean, interval, or
claim is included until all units complete and the independent verifier passes.
At the latest check (`2026-09-15T07:32:08Z`), four units were complete at 4/4
arms (`beta0.5_seed1/2/10` and `beta0.9_seed1`); the next create-only unit
`beta0.9_seed2` had started under driver child PID `22544`.  Chain PID `18625`
remained present, and this is still not a beta result.

The guarded beta aggregator SHA-256 is
`7ec29fda1e4c1f9ed8c0fc64c69c79b6d4ff4d8c608b082bdc8a406e6c7c9c7a`; the
independent beta verifier SHA-256 is
`23ea6a55954dabff303e4e7ccaa3689afc5d19aeff51156a2efb4e4f4e0ece49`.

### Scale-up delivery and validation

The canonical new create-only delivery is `results/ca_opex_scaleup_bundle_v3.tgz`,
with 275 payload files and SHA-256
`e2f889d2e69e7a12d964aa7a185dd797bce3d3ceb4f3359c9baad61e2e47766b`.  It
includes Walker2d seeds 2/3/4 and Hopper-v4 seeds 2/3/4 checkpoints, raw
records, configurations, calibration artifacts, code, tests, and logs.  It
does not include the source HDF5 files or the formal v3 bundle.  The optional
beta chain is included only under `running_optional/` as partial provenance,
not as a completed result.  The fresh remote targeted test log
`results/final_validation_takeover_20260915.log` has SHA-256
`20a2d5ad5fb735d91d4d03e9b2100c367895ef4fdcf4bf9c50d1dc1df026850a` and
records `55 passed` with exit code 0.  The corrected full suite from the
canonical `code/` entry point is recorded in
`results/final_validation_takeover_full_v2_20260915.log` with SHA-256
`a574edb1c2ddc1244c0f9f70635b57cc5486992cbea53f73b1c4e1de8ff3cec8` and
`251 passed`.  The earlier v1/v2 archives are preserved as noncanonical
snapshots.  The bundled summary entry point has SHA-256
`2af45475437cfab91315baf8601b2a0b373208f558f68cd20774c5cdbf4e962c`.

## 5. Mechanism evidence, costs, and boundaries

The completed evidence supports six limited statements:

1. **Additional neural test-time compute does not explain the full gain.**  On
   fresh rollouts, complete CA-OPEX beats the independently tuned nominal-Q
   `K=8,T=2` controller by `+8.0196/+13.5987` normalized points, with both paired
   t and bootstrap intervals excluding zero at each checkpoint.  This controls
   Q rows and backward calls, not scalar bisection work or wall time.
2. **Explicit channel expectation has a controlled signal.**  With identity
   anchor, `eta=0.3`, `delta=2`, `K=8`, and `T=2` fixed, calibrated expected-Q
   beats nominal-Q by `+6.3663/+7.0461`; both interval types exclude zero in
   both rows.  Against independently tuned nominal-Q the result is mixed across
   checkpoints, so the stronger general statement is not supported.
3. **The selected inverse-centered stack matters.**  Inverse-only is better than
   identity commands, and complete inverse-anchor CA-OPEX is better than the
   independently tuned identity-anchor CA controller on both checkpoints.  The
   two CA variants match actor rows, Q rows, and backward calls per decision, but
   they are not exact wall/FLOP matches: inverse adds 48 analytic bisection steps,
   and their independently selected `eta/delta` values differ.  This is therefore
   a strong selected-controller comparison, not a single-factor anchor ablation.
4. **Value correction adds to inversion.**  Complete CA-OPEX beats the
   bit-matched inverse-only arm by `+8.9597/+8.8130` normalized points in real
   closed-loop return.  This is the clearest evidence for the second stage.
5. **A tight inverse-centered region is a guardrail.**  On 100,060 states held
   out from adapter SGD and Q-scale fitting, wide and nominal adapters report
   *larger* frozen min-twin Q gains (`0.4708` and `0.5380`) than inverse sampled
   (`0.2358`) while obtaining much worse rollout scores.  The ordering is
   essentially reversed.  The audit therefore demonstrates that maximizing the
   source critic is not itself evidence of control improvement; the inverse
   anchor and closed-loop evaluation are necessary.
6. **Amortization is feasible but not free.**  The formal learned inverse
   adapter uses no online critic and remains positive over inverse-only, but its
   two-checkpoint mean (`27.8053`) is below unamortized CA-OPEX (`29.4554`).
   The earlier teacher-distilled development student recovered only 44.22% of
   the teacher gain.  Zero-critic deployment is an engineering extension, not a
   stronger result than the primary method.

The held-out audit states are held out only from adapter training and Q-scale
fitting, not from the already trained base critic.  Mean twin-disagreement
change for inverse sampled is small (`+0.0032/-0.0085`), but that does not make
the Q estimate a target-channel continuation-value certificate.

### Measured cost

| Deployment controller | Base-actor rows / step | Q1 rows / step | action-gradient backward calls / rows | Parameter updates online |
|---|---:|---:|---:|---:|
| CA-OPEX inverse or identity (`K=8,T=2`) | 1 | 16 | 2 / 16 | 0 |
| Original-structure OPEX (`K=1,T=1`) | 1 | 1 | 1 / 1 | 0 |
| Learned residual adapter | 1 | 0 | 0 / 0 | 0; one small MLP forward |

CA-OPEX inverse measured `8.711/8.575 ms` adapted-controller wall time per
environment step for checkpoints 1/10; identity CA measured approximately
`7.221/6.849 ms`, and original OPEX `3.593/3.608 ms`.  Total Q-row counts differ
because better policies survive for different numbers of steps, so the fair
algorithmic budget is the per-step row/call count.  Each 5k adapter run used
899,940 base-actor precompute rows, 32,768 Q-scale rows, 20.48M optimizer Q1
forward rows, 10.24M Q1 backward rows, and 1.28M adapter rows in each direction.
Inverse-adapter training took `48.57/50.90 s`.  Its paired evaluation records
only combined arm wall time, so no per-arm latency is imputed.

Unsupported or deliberately bounded statements:

- One task and a synthetic IID clipped-Uniform channel do not establish broad
  robot robustness, vision robustness, or latency/state-dependent adaptation.
- CA-OPEX versus original OPEX is not equal compute.  The completed supplemental
  audit matches per-decision actor/Q rows and backward calls to a tuned nominal-Q
  controller, but inverse anchoring still adds 48 scalar bisection iterations;
  total episode steps and wall time differ with policy survival.  It is therefore
  an equal-neural-budget mechanism audit, not exact FLOP or wall-time equality.
- The Q-at-mean ablation changes both estimator placement and its
  estimator-consistent scale; its mixed sign cannot be presented as a pure
  Jensen-effect or sampled-estimator result.
- The endpoint MLE is accurate here, but percentile bootstrap under-covers this
  non-regular boundary estimator and is not used as a confidence claim.
- The frozen source critic may still be wrong about long-horizon target-channel
  continuation.  Closed-loop return, rather than Q gain, is the decision metric.
- The original formal confirmation has only two independently trained base
  checkpoints, so its intervals remain fixed-checkpoint episode-case intervals.
  The later Walker2d scale-up and Hopper-v4 replication add training-seed
  intervals separately; they do not retroactively make the formal block blind.

## 6. Reproduction entry points

### Environment and immutable inputs

Server environment:

```bash
source /root/hubl_backup_env/bin/activate
cd /root/hubl_research_20260914/code
python -c "import torch, gymnasium, mujoco; print(torch.__version__, torch.cuda.is_available())"
```

The local dependency record is `hubl_backup/requirements-research.txt`.  Core
input SHA256 values:

- source HDF5: `159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a`
- 512-pair calibration JSON: `b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354`
- base seed-1 checkpoint: `22b47965695bfa312533741abaa1e17e3cbbc091f3c5e6ba7fc8953fbc8ca6ae`
- base seed-10 checkpoint: `eba1af040aa73823ceba28048e10d3af2eef54c59f09c1e0194f27aa0ed4944d`

### Calibration and development selection

```bash
python generate_channel_pairs.py \
  --source /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --output /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001.npz \
  --pairs 512 --beta 1.25 --seed 27001
python calibrate_uniform_channel.py \
  --pairs /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001.npz \
  --output /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001_calibration.json \
  --bootstrap-replicates 1000 --bootstrap-seed 28001
python select_channel_opex_boundary_extension.py \
  --protocol channel_opex_boundary_extension_protocol.json \
  --output /tmp/selection_revalidation.json
```

The last command intentionally refuses an existing output.  Its current
post-evaluation protocol validates the preserved original chain and ten raw runs;
it is a provenance revalidation, not a new selection.

### Frozen confirmation

The original one-shot sequence was:

```bash
python freeze_inverse_residual_manifest.py \
  --results-root /root/hubl_research_20260914/results \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --calibration /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001_calibration.json \
  --base-seed1 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt \
  --base-seed10 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt

# These render commands only; actual runner preflight performs semantic checks.
bash run_frozen_channel_opex_confirmation.sh --dry-run
bash run_frozen_inverse_residual_confirmation.sh --dry-run

# Executed once. Both scripts refuse any existing target.
bash run_frozen_channel_opex_confirmation.sh
bash run_frozen_inverse_residual_confirmation.sh

# Canonical strict aggregate (already exists; do not overwrite).
python aggregate_inverse_residual_results.py \
  --manifest inverse_residual_frozen_confirmation_manifest.json \
  --output-dir /root/hubl_research_20260914/results/inverse_residual_confirm/aggregate \
  --stem frozen_confirmation
```

For independent checking, extract
`results/inverse_residual_confirm/formal_confirmation_repro_bundle_v3.tgz`
(SHA-256
`d816ecb7b3b70a80e48cdb90156a04dafc35c4185950bb0554735b16ef895d9f`)
at `/root/hubl_research_20260914` and aggregate to a *new* directory/stem as
shown in `FORMAL_CONFIRMATION_REPRO.md`.  The 16 MB archive contains all 97
unique inputs named by the strict report, the 10 sibling training logs, raw
returns, configs, checkpoints, calibration artifacts, executable clean-room
code, tests, and pinned dependency versions.  It excludes the 1M-transition
source HDF5 to avoid duplicating a 222 MB input; that file is needed for
retraining but not for exact report reconstruction.  An independent audit read
the files directly from the archive, found 0 missing/hash/size mismatches, and
rebuilt a JSON object exactly equal to the saved aggregate.

Frozen implementation hashes are aggregator
`f4fe3c58a13342e0722c9ee1fee46001a51ea1b98d6a8fbafd37065d9006d146`,
freezer wrapper
`87f39e82b999209e3b2a8f09e8e7454fce52fd09a9fbd3f76d40684b562e59b3`,
OPEX runner
`7a4dfdfbae6883382e27af90cb7d2091a257ac6b48508e80e838964981cd3260`,
and adapter runner
`7f1585820eeb787829b590758837205ee1b776a0bb8fae696ef46f2e36cd9c40`.
Output hashes are JSON
`bbe5d2908df1ec6499a5afd2b4cc7db64b9fc9a83b175b0860a912dc63ba7795`,
CSV `9c3d896c4a1c9738afeb2be6a7a6ec1607b957d76d2848f30672cbdb70066753`,
and Markdown
`c04ee00a1f148c7b70086df84c763b699ffb69b695d17cd0adfce9e68e513d58`.

`latest.pt` supports state-complete, same-runtime resume (adapter, optimizer,
minibatch/channel/global RNG states).  Bitwise uninterrupted-versus-resumed
equality is tested on CPU; universal CUDA bitwise replay is not claimed.

### Distillation diagnostic

The original single-use driver and its frozen development protocol are preserved:

```bash
bash run_opex_distillation_dev.sh --preflight-only
python recover_opex_distillation_dev.py \
  --protocol opex_distillation_recovery_protocol.json \
  --output /tmp/distillation_comparison.json
```

The recovery verifier performs no training or evaluation and checks the existing
artifacts by SHA.  To run a genuinely fresh distillation experiment, use a new
output directory and freeze a corrected protocol first; do not overwrite or
relabel this development run.

### Post-confirmation equal-neural-budget audit

The launch-frozen protocol, runner, independent verifier, and their hashes are:

- `equal_compute_nominal_control_protocol.json`: `fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034`
- `run_equal_compute_nominal_control.py`: `c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1`
- `verify_equal_compute_nominal_control.py`: `053d3d823bf23475385b956426f67d78c213bd0cc74b482cf995d7c9ddf2da90`

The immutable raw run is already complete.  Re-run only the independent verifier
in check-only mode, or write a new non-overlapping output stem.  The runner's
dry-run and completed-tree validation paths were also exercised:

```bash
python run_equal_compute_nominal_control.py --dry-run
python run_equal_compute_nominal_control.py --resume-missing
python verify_equal_compute_nominal_control.py --check-only
python verify_equal_compute_nominal_control.py \
  --output-dir /tmp/ca_opex_equal_compute_verify \
  --stem independent_reaggregation
```

The v2 verifier reconstructed the result directly from raw returns, did not
import the v1 runner, and passed 29/29 input hashes, seed-disjointness, 8 absolute
controller rows, 10 paired comparisons, and all cost identities.  Output hashes
are JSON `3ca8b8c1e342d623f43ec73a3b16bc9ed9a51cad40e662867771110f2b8acf26`,
CSV `358a571aeadf8166273b054667a5ef40adea7a525579db5d9f8be1c310ef032d`,
and Markdown `4b9dd5227f39f86ce537824ff729f045f077a4865cd7263b972fd4b7799c3433`.
`results/equal_compute_nominal_control_bundle.tgz` is a supplemental overlay for
the formal v3 tree and packages its protocol, driver/verifier code, tests, logs,
raw returns, and both aggregates; its SHA-256 is
`f7fa50b705171e141e85b84a4632d395ad679141bfc4f5fd159ab8b32792c84e`.
After adding the scale-up safeguards, the final server suite passed 232 tests
with exit code 0.  The retained log includes the command, cwd, Python version,
and hashes of the new/frozen drivers; it is
`results/final_validation_20260915.log`, SHA-256
`aa0bde3d5749cc8000be1f58f5ae586a304e97091a7dac0a2ade9956080f841e`.
The final supplemental delivery overlay is
`results/ca_opex_delivery_overlay_v2.tgz`, SHA-256
`7701de19d50478522084c3f3d0a3ed22888d4c8164898b0ea73e31426f4cff17`.
Extract it over the formal v3 tree; its 51 entries add the equal-budget raw
audit, a README snapshot, scale-up driver/aggregator and tests, plus both retained
validation logs.  It does not duplicate the base checkpoints in the formal
archive and is not a standalone replacement for that archive.

## 7. Scale-up checklist

### A. Routine expansion of a core that survived confirmation

1. **More training seeds on Walker2d** — extend base seeds from 2 to 5 while
   retaining 50 paired evaluation episodes and the fixed `eta/K/T/delta` values.
2. **More public locomotion tasks** — Hopper and HalfCheetah with separately
   generated command/execution logs, identical 512-pair information constraint,
   ordinary OPEX, inverse-only, tuned-on-development identity CA, and the
   equal-neural-budget nominal-Q control.
3. **Channel coverage** — without retuning, evaluate beta `{0.5, 0.9, 1.1, 1.4}`;
   then add Gaussian, biased, and state-dependent channels as clearly new model
   classes rather than pretending the Uniform MLE applies.
4. **Calibration sensitivity** — pair counts `{32, 128, 512, 2048}` and honest
   endpoint-estimator uncertainty; pair collection cost must be reported.
5. **Closest implementable baselines** — PLAS+P-like learned perturbation and a
   PA-RL action-optimization/distillation control on matching offline data and
   compute.  SPAR should be added when reproducible author code is available or
   an independently validated clean implementation is feasible.
6. **Visual/robotic evidence** — only after state-control results survive,
   evaluate a small image-based or real actuator task.  RSS positioning requires
   actual hardware or a substantially more realistic execution-channel test.

Walker2d seed expansion now has a single fail-closed entry point.  It refuses
training seeds `0/1/10`, rejects any 50-case RNG block overlapping the formal or
supplemental runs, verifies the fixed dataset/calibration hashes, and requires
the entire per-seed output root to be absent.  It then trains the exact 25k HUBL
base, runs complete CA-OPEX, calibrated identity, equal-neural-budget nominal-Q,
original OPEX, and inverse-only on one paired block, and create-only aggregates
their raw returns.  An interrupted directory is preserved for diagnosis and is
never silently resumed or overwritten.

```bash
bash run_ca_opex_walker_scaleup.sh \
  --seed 2 --env-seed 131300 --noise-seed 231300 --grad-seed 331300 \
  --dry-run

# Remove only --dry-run to execute after reviewing the rendered seven commands.
bash run_ca_opex_walker_scaleup.sh \
  --seed 2 --env-seed 131300 --noise-seed 231300 --grad-seed 331300
```

`run_ca_opex_walker_scaleup.sh` has SHA-256
`9f78809ac62d7e3a931ded3130d3a91f979baef5d65553a06977d61625d7911d`;
its strict raw-return aggregator has SHA-256
`20b8ddfe74ab253d29a74be7b0c39bb58423a74950b369516958a040975e3cac`.
The exact seed-2 dry-run above completed with exit code 0 and confirmed that no
output root was created; its retained validation log has SHA-256
`c45d1510096448a47bed10ea9430866624f8863d6459a24a5ef51e3fdd9bf369`.
The new tools passed 36/36 targeted Linux tests.  The dry-run itself started no
training, but the later real seed-2 launch was verified from PID `13893`, its
stage log, raw controls, aggregate, and 25k checkpoint.  Seeds 3 and 4 were
also subsequently completed under the same frozen settings.  Hopper-v4 then
completed the cross-task replication; HalfCheetah remains unrun.

Measured throughput on the RTX 2080 Ti gives the following planning scale.  A
50-episode paired complete CA-OPEX evaluation took `212--240 s`, identity CA
`127--132 s`, and original OPEX `55--61 s`, including each raw file's baseline
arm.  Thus the three-controller primary matrix is about 7 minutes per new base
checkpoint before I/O/launch margin.  One 5k inverse-adapter fit took about 50 s;
its target paired evaluation took `101--123 s`, while the K=64 audit evaluates
100,060 held-out source states.  A full new seed including base training should
reserve roughly 15--25 GPU-minutes; this is an extrapolation because base-policy
training and rollout length vary.  The 1M-transition collector measured
1,609.7 transitions/s (about 10.4 minutes per task dataset) and a prior 100k
HUBL run took about 991 s, providing the basis for multi-task estimates.

### B. Scientific questions that could still overturn the contribution

1. Does a source-channel critic remain useful when the target channel changes the
   entire continuation distribution?  If not, channel-consistent fitted-Q may be
   necessary and the current frozen-Q claim would narrow.
2. Is the inverse anchor's benefit specific to severe clipping at beta 1.25?
3. Can the action-pair model tolerate biased, correlated, delayed, or
   state-dependent execution channels without requiring target transitions?
4. Is the deployment return gain worth 16 Q rows and two backwards per step, or
   can the positive but partial student amortization be made reliable across
   independent checkpoints?

The original confirmation question—whether complete CA-OPEX beats inverse-only,
tuned equal-Q-call identity CA, and original OPEX on both held-out base
checkpoints—is resolved positively.  The later equal-neural-budget audit also
resolves the narrow question of whether merely raising nominal OPEX to the same
Q-row/backward-call budget explains the full gain; it does not.  Items B1–B4
above remain explicit until their corresponding evidence is collected; they are
not disguised as additional-seed chores.

## 8. Current maturity and target venue

The package now has the ingredients of a method-paper prototype: a precise
information regime and objective, clean implementation, action-pair estimator,
frozen selection record, direct OPEX and inverse controls, an independently
tuned equal-call control, two-checkpoint prospective selected-method evidence,
mechanism ablations, a fresh-rollout equal-neural-budget nominal-Q audit,
measured online/offline cost, raw records, checkpoints, tests, and strict
reproducibility bundles.  The completed continuation adds five Walker2d
training seeds and a three-seed Hopper-v4 cross-task replication, while the
formal selected-method confirmation remains the original two-checkpoint
block.  “Equal-call” here means the same
per-decision critic rows/backward calls, not identical wall time or all scalar
operations.  The formal result is internally credible but scientifically
narrow: its blind prospective block is one MuJoCo task, one severe synthetic
channel, two base-policy training seeds, and fixed-checkpoint rather than
training-seed intervals.  The later expansions are explicitly reported as
supplemental five-seed Walker2d and three-seed Hopper-v4 evidence, not as a
claim of broad robot robustness.

The primary target should be **ICLR**, with **AAAI** as the secondary venue.
The requested Walker2d seed expansion and Hopper-v4 cross-task coverage are
complete; beta-severity coverage is still an optional running audit, while
HalfCheetah remains unrun.  The method's combination of low-information
actuator calibration,
offline RL, and test-time policy extraction fits a learning-method venue better
than a broad robotics claim.  **RSS** would require hardware or substantially
stronger embodied validation.  The largest current risk is cross-task/channel
generalization, followed by whether a source critic remains reliable under
state-dependent or delayed execution.  No acceptance probability or guarantee
that the gains scale is claimed.
