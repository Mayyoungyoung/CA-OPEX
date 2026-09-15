# HUBL stochastic-backup execution notes

## Upstream status and what is safe to reuse

The authoritative paper is Sinong Geng, Aldo Pacchiano, Andrey Kolobov, and
Ching-An Cheng, [*Improving Offline RL by Blending Heuristics*](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b40cd8bb212ad8ad0a8c8a43c4da5f0b-Abstract-Conference.html),
published at ICLR 2024.  The official code is not a discoverable GitHub
repository.  It is the [ICLR supplementary ZIP](https://proceedings.iclr.cc/paper_files/paper/2024/file/b40cd8bb212ad8ad0a8c8a43c4da5f0b-Supplementary-Conference.zip):

- supplementary SHA-256:
  `8c11769b829ec4800deeb17fe2ba5b21c46f20d2f833da860711c5862b29a7b3`;
- paper PDF SHA-256:
  `4f9dbc67cbff0bc6a0009d188c2c11a0f67a7aa91b76ca7d3fcfc4a28e3eb9c6`;
- the ZIP has no `.git` metadata, commit identifier, `LICENSE`, or other
  license notice.  Consequently, this project does **not** vendor or modify
  that source.  The files in this directory are independent implementations
  from the published equations;
- GitHub repository searches by exact title, `HUBL offline RL`, and distinctive
  source identifiers returned no official repository as of 2026-09-14.  This
  is evidence of no located repository, not proof that none exists.

The upstream `install.sh` specifies the following environment:

| Component | Upstream pin/status |
|---|---|
| D4RL | `rail-berkeley/d4rl@6330b4e09e36a80f4b706a3885d59d97745c05a9` |
| PyTorch | `1.12.1` |
| TensorBoard / psutil / protobuf | `2.10.0` / `5.9.1` / `3.19.4` |
| lightATAC | GitHub install, **unlocked commit** |
| IQL-PyTorch | Git clone, **unlocked commit** |
| young-geng/CQL | Git clone, **unlocked commit** |
| Other Python | `wandb`, `ml_collections`, `dowel`, versions unlocked |
| Simulator | legacy MuJoCo 2.1.0 plus `mujoco_py` |
| Meta-World | authors state they used a private version; not reproducible from the ZIP |

Two upstream code defects matter for automation: the dataset retry branch calls
the nonexistent `time.wait(300)`, and the README command omits
`--lambda-method` although the default string `None` reaches a
`NotImplementedError`.  A faithful launch must explicitly pass, for example,
`--lambda-method traj_portion` (rank) or `constant`.

## Exact stochastic protocol in the paper

Appendix D.6.3 constructs a stochastic Walker2d environment by sending

\[
a_{\mathrm{env}} = \operatorname{clip}(a_{\mathrm{policy}} + \beta\epsilon,
-1, 1),\qquad \epsilon_j\overset{iid}{\sim}\mathcal U[-1,1]
\]

to the environment.  It evaluates
`beta = {0.0001, 0.001, 0.01, 1}`, regenerates an offline dataset with the
original `walker2d-medium-v2` behavior policy, and trains TD3+BC with HUBL.
The paper's global protocol uses training seeds `{0, 1, 10}`, one million
gradient updates, minibatches of 256, and ten evaluation episodes.  Table 13
reports HUBL relative improvements (sigmoid/rank/constant):

| beta | sigmoid | rank | constant | base normalized score |
|---:|---:|---:|---:|---:|
| 0.0001 | +6% | +6% | +4% | 66.99 |
| 0.001 | +6% | +8% | +7% | 66.73 |
| 0.01 | +4% | +4% | +6% | 64.23 |
| 1 | +4% | -4% | +5% | 28.09 |

The authors explicitly identify further degradation at larger noise as a HUBL
limitation and observe that constant blending is more robust because it does
not depend on noisy heuristic ranking.  This is the direct motivation for a
cross-fitted uncertainty-calibrated trajectory score.  Additive **reward**
noise is a useful mechanism audit, but it is not the same protocol: action
noise changes state visitation, next states, rewards, termination, and dataset
support.

The supplementary material does not include the stochastic-data collection
script or state whether `actions` stores the commanded or noise-perturbed
action.  This ambiguity must be frozen explicitly in a new protocol.  The
recommended primary convention is to store the action actually applied to the
environment and additionally store `infos/commanded_action` and
`infos/action_noise` for auditability.  A commanded-action ablation can test
the partial-observability interpretation.

## Verified server path

The existing `/root/rivermind-data/envs/sampler-robust` environment cannot run
Gymnasium MuJoCo: an installed but unconfigured `mujoco_py` raises a generic
exception while Gymnasium probes its backends.  It was left unchanged.

A clean environment was created on the system disk:

```bash
python3 -m venv --system-site-packages /root/hubl_backup_env
/root/hubl_backup_env/bin/python -m pip install \
  'gymnasium[mujoco]==0.29.1' scipy==1.15.3 h5py==3.16.0
```

Verified imports are PyTorch 2.1.0 (CUDA visible), Gymnasium 0.29.1, NumPy
1.26.0, and MuJoCo 3.13.0.  With no rendering and random actions, actual
10,000-step measurements were:

| Environment | Steps/second | Result |
|---|---:|---|
| `Hopper-v4` | 3,722 | passed |
| `Walker2d-v4` | 3,612 | passed |
| `HalfCheetah-v4` | 6,761 | passed |

This establishes simulator availability, not equivalence between modern v4
dynamics and the paper's legacy v2 environment.  Use the same environment
version for every method in our comparison and label it as a new protocol.
Run the exact noise wrapper smoke test with:

```bash
/root/hubl_backup_env/bin/python action_noise_smoke.py \
  --env Walker2d-v4 --beta 0.01 --steps 10000 \
  --env-seed 0 --noise-seed 1 --policy-seed 2 \
  --output results/action_noise_smoke.json
```

The script records independent RNG seeds, noise statistics, clipping rate,
completed returns, package versions, wall time, and throughput.  It is not an
offline-RL result.

## Dataset routes and current transfer

The official D4RL v2 URL is documented in the
[D4RL source](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/infos.py):

```text
https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium-v2.hdf5
```

It is 232,254,996 bytes.  The server reached only about 13 KiB/s from Berkeley,
so that route was stopped.  Direct Hugging Face access is blocked on the host,
but this mirror worked at 307,221 B/s over a measured 10 MiB range:

```text
https://hf-mirror.com/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5?download=true
```

The mirror advertises LFS object
`cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda`.
The repository page labels the mirror Apache-2.0, but that label does not
supersede upstream dataset rights; cite D4RL as the data source.  A download was
started as PID 1497 with these durable paths:

```text
/root/hubl_backup_data/walker2d_medium-v2.hdf5.part
/root/hubl_backup_data/download.log
/root/hubl_backup_data/walker2d_medium-v2.sha256
/root/hubl_backup_data/walker2d_medium-v2.hdf5
```

The `.hdf5` filename appears only after successful download, hashing, and an
atomic rename.  Verify rather than assuming completion:

```bash
ps -p 1497 -o pid,etime,cmd
tail -n 3 /root/hubl_backup_data/download.log
ls -lh /root/hubl_backup_data/walker2d_medium-v2.hdf5*
cat /root/hubl_backup_data/walker2d_medium-v2.sha256
```

A Minari-format alternative is documented by
[Farama Minari](https://huggingface.co/datasets/farama-minari/mujoco/tree/611ea1898caf8ad873fb3985a1915f3548af085a/walker2d/medium-v0/data):

```text
https://hf-mirror.com/datasets/farama-minari/mujoco/resolve/611ea1898caf8ad873fb3985a1915f3548af085a/walker2d/medium-v0/data/main_data.hdf5
```

Its HEAD response is 210,762,366 bytes.  Prefer original D4RL for comparison
with HUBL because conversion/version differences otherwise become a confound.

## Cheapest discriminative execution sequence

1. **Data integrity (minutes).** Verify the completed hash and HDF5 keys.
   D4RL v2 medium data should expose transition arrays plus
   `infos/qpos`, `infos/qvel`, `infos/action_log_probs`, and
   `metadata/policy/*`.  The [official task documentation](https://github.com/Farama-Foundation/D4RL/wiki/Tasks)
   states that v2 medium/expert files contain behavior-policy metadata.
2. **Policy reconstruction (less than one hour).** Reconstruct the two-hidden-
   layer SAC policy from `metadata/policy/fc0`, `fc1`, `last_fc`, and
   `last_fc_log_std`.  Before collection, compare computed log probabilities
   against `infos/action_log_probs` and evaluate unperturbed returns.  Do not
   proceed if shape/transposition or tanh correction is unresolved.
3. **Mechanism audit first (roughly minutes on GPU).** Run the existing
   cross-fitted score audit on complete trajectories.  Freeze folds before
   looking at policy returns.  Compare raw-return rank, constant lambda,
   sigmoid, rank, posterior-mean rank, and LCB rank.  Primary mechanism metrics
   are clean-return ranking/Kendall correlation and high-return selection
   precision; clean labels are oracle diagnostics only.
4. **Small policy test (development).** One Walker2d dataset, TD3+BC or IQL,
   100k updates, one seed, equal batch/data/evaluation budgets.  Include base,
   original rank HUBL, constant HUBL, and uncertainty-calibrated HUBL.  If the
   proposed score cannot beat constant HUBL under stochasticity, the stated
   mechanism has not survived its most direct control.
5. **Frozen confirmation.** Fix code, alpha/kappa selection, folds, checkpoint
   rule, and evaluation rule; then run seeds `{0,1,10}` with one million updates
   and all noise levels.  Never reuse the seed/noise instances used to choose
   the scoring rule as confirmation.

The measured environment rate implies a one-million-step simulator-only lower
bound of about 4.6 minutes for Walker2d.  Neural policy inference, HDF5 writing,
frequent resets, and stochastic bookkeeping will increase this; time one 10k
collection before extrapolating.  No trustworthy full-training wall-time has
yet been measured on this 2080 Ti, so it must not be reported as known.

## Remaining protocol risk

The strongest unresolved reproducibility issue is version alignment.  The
paper used Gym/MuJoCo `walker2d-v2`; the working environment is Gymnasium
`Walker2d-v4`.  D4RL's [dataset reproducibility guide](https://github.com/Farama-Foundation/D4RL/wiki/Dataset-Reproducibility-Guide)
still lists the downloadable v2 RLKit snapshot as `TODO`, but the v2 HDF5
policy metadata provides a plausible recovery route.  A successful metadata
log-probability audit and matched unperturbed return are required before
calling that recovery faithful.  Until then, experiments are a clearly labeled
modern reimplementation, not reproduction of Table 13.
