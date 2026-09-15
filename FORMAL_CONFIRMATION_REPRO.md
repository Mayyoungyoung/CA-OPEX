# Formal CA-OPEX confirmation: immutable reproduction record

This file is the post-run entry point for the completed 2026-09-15
Walker2d-v4 confirmation.  It does not modify the pre-run manifest.

## Scope

- Canonical frozen manifest SHA-256:
  `57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807`
- Strict aggregate JSON SHA-256:
  `bbe5d2908df1ec6499a5afd2b4cc7db64b9fc9a83b175b0860a912dc63ba7795`
- Aggregate status: `complete`, validator `strict_fail_closed_v2`
- Inventory: 10 adapter runs, 14 external evaluations, 56 comparisons,
  117 referenced records / 97 unique files.

The manifest's `evidence_scope` text is an immutable statement written before
the run.  Its phrase saying that the template was unusable "until" placeholders
were resolved describes a prerequisite at freeze time, not the current status.
The canonical manifest fulfilled that prerequisite and has no placeholder.

## Environment

The executed server environment was:

```bash
source /root/hubl_backup_env/bin/activate
cd /root/hubl_research_20260914/code
python -c "import torch, gymnasium, mujoco, numpy, scipy, h5py; print(torch.__version__, torch.cuda.is_available())"
```

Pinned package versions are in `code/requirements-research.txt`.  The GPU was
an NVIDIA GeForce RTX 2080 Ti with 22,528 MiB reported memory.

## Actual freeze and execution chronology

The canonical freeze entry point was:

```bash
python freeze_inverse_residual_manifest.py \
  --results-root /root/hubl_research_20260914/results \
  --dataset /root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5 \
  --calibration /root/hubl_research_20260914/results/channel_calibration/beta125_n512_seed27001_calibration.json \
  --base-seed1 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt \
  --base-seed10 /root/hubl_research_20260914/results/inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt
```

The first freeze artifact is retained as
`inverse_residual_frozen_confirmation_manifest.invalid_preflight_placeholder_keys.json`.
It left placeholder tokens in provenance-map keys.  The first actual OPEX
runner rejected it during preflight before any formal raw output was created.
The corrected freezer then produced the canonical manifest above.  Runner
`--dry-run` only renders commands and was not the semantic validation step.

The successful one-shot runners were:

```bash
bash run_frozen_channel_opex_confirmation.sh
bash run_frozen_inverse_residual_confirmation.sh
```

Both runners refuse an existing output target.  Do not rerun them over this
archive.  `latest.pt` stores adapter, optimizer, minibatch/channel/global RNG
states for same-runtime resume; bitwise interrupted/uninterrupted equality was
tested on CPU, not universally guaranteed on CUDA.

## Rebuild the strict aggregate

Extract the reproduction bundle so that it recreates
`/root/hubl_research_20260914/{code,results}`.  Then use a new output directory;
the archived canonical aggregate remains immutable:

```bash
source /root/hubl_backup_env/bin/activate
cd /root/hubl_research_20260914/code
python aggregate_inverse_residual_results.py \
  --manifest inverse_residual_frozen_confirmation_manifest.json \
  --output-dir /root/hubl_research_20260914/results/inverse_residual_confirm/reaggregate_audit \
  --stem frozen_confirmation_reaggregate
```

The original aggregate files are:

- `results/inverse_residual_confirm/aggregate/frozen_confirmation.json`
  (`bbe5d2908df1ec6499a5afd2b4cc7db64b9fc9a83b175b0860a912dc63ba7795`)
- `results/inverse_residual_confirm/aggregate/frozen_confirmation.csv`
  (`9c3d896c4a1c9738afeb2be6a7a6ec1607b957d76d2848f30672cbdb70066753`)
- `results/inverse_residual_confirm/aggregate/frozen_confirmation.md`
  (`c04ee00a1f148c7b70086df84c763b699ffb69b695d17cd0adfce9e68e513d58`)

The archive includes every one of the aggregate's 97 unique inputs, including
all raw returns, configurations, checkpoints, sibling training console logs,
calibration files, executable source, tests, and the dependency record.  The
large source HDF5 is not duplicated in the archive; it is needed to retrain but
not to verify or rebuild the completed aggregate.  Its SHA-256 is
`159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a`.
