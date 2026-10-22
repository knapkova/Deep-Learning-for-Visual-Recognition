# Camelyon17 full-data experiment code

This package runs the main hospital generalization experiment: **180 training runs**
(3 fixed patient partitions × 5 source hospitals × 4 fine-tuning depths × 3 model seeds).
It evaluates each source-trained ResNet-50 on all five hospitals and computes fixed
ImageNet representation distances. The custom hospital-wise patient split is
approximately 60/20/20; it is **not the official WILDS split**. Source validation
loss selects checkpoints; target tests do not select a checkpoint or depth.

## Files

| File | Purpose |
| --- | --- |
| `configs/full.json` | Unchanged full-data plan: 455,954 patches, partition/model seeds 42/43/44, four depths, 20-epoch maximum, patience 5. |
| `full_run.py` | Prepare inputs, run sequentially on one GPU, build reports and audit results. |
| `camelyon/data.py`, `models.py`, `train.py` | Pinned data acquisition, patient grouping, ResNet-50 training, resume and integrity checks. |
| `full_analysis.py` | Fixed ImageNet avgpool features and exact mean marginal one-dimensional Wasserstein-1 distances on training members. |
| `full_results.py` | Combine all partitions/runs into metrics, stability summaries and figures. |
| `audit_results.py` | Check prediction coverage, metrics and provenance for one split. |
| `full_cache.py` | Optional private PNG-cache export/restore used by preparation. |
| `ece_run.py` | Optional POSIX two-GPU orchestration with separate one-GPU workers and finalization locks. |
| `camelyon/analysis.py` | Retained shared analysis implementation; also contains historical CORAL helpers, which the main full-data pipeline does not invoke. |
| `tests/` | Original focused regression tests for data, training, distances, cache, reports, audit and orchestration. |

## Prerequisites

Use Python 3.10+ with a matching **CUDA-enabled PyTorch/torchvision pair**, then
install `requirements.txt` into that environment. The minimum versions are not a
reproduction lockfile; software/GPU differences can change numerical results.
The full plan requires an NVIDIA CUDA GPU. Optional two-worker execution requires
two distinct GPUs and a POSIX system (`fcntl` file locks).

Preparation needs internet access to the pinned
[`wltjr1007/Camelyon17-WILDS` export](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/tree/d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f).
Training/feature extraction also needs torchvision ImageNet V2 weights, downloaded
on first use if absent from the normal torch cache. Check the upstream data terms
and access requirements before downloading. Raw images, downloaded shards,
pretrained weights, trained checkpoints and feature/image caches are **not included**.
Provision substantial free disk space for all 455,954 PNGs, source shards, features
and 180 run folders; this is a long GPU workload, not a quick example.

## Fresh run

Run these commands from this `code` directory, in your prepared Python environment.
Keep generated data and outputs in a **new writable directory outside the checkout**.
Choose paths for your machine; do not point at archived result folders.

```sh
python -m pip install -r requirements.txt
RUN_ROOT="$HOME/camelyon17-fresh-run"
OUTPUT="$RUN_ROOT/outputs/full180"
mkdir -p "$RUN_ROOT"
python full_run.py prepare --root "$RUN_ROOT" --config configs/full.json --output "$OUTPUT"
CUDA_VISIBLE_DEVICES=0 python full_run.py run --root "$RUN_ROOT" --config configs/full.json --output "$OUTPUT"
python full_run.py verify --root "$RUN_ROOT" --config configs/full.json --output "$OUTPUT"
```

`prepare` obtains the full pinned export, builds the three prespecified manifests,
checks patient/slide isolation and class coverage, and hashes prepared images.
`run` writes checkpoints, histories, predictions, distances and reports.
`verify` audits all three splits and the combined coverage/stability report.
`all` performs all three stages. Repeating a command with matching inputs can
reuse verified results or resume checkpoints; mismatching signatures are rejected.
The reference feature cache is also checked before reuse. An optional
`--image-cache-dir /path/to/private/cache` on `prepare` can restore/export PNGs.
Never add that directory to Git.

To regenerate reports without training:

```sh
python full_results.py --results "$OUTPUT"
```

For a direct audit of one prepared split, pass its generated configuration:

```sh
python audit_results.py --manifest "$RUN_ROOT/data/full/split_42/manifest.csv" --results "$OUTPUT/split_42" --config "$OUTPUT/split_42/config.json" --main-only
```

For optional two-GPU execution, prepare first, then use `ece_run.py supervise`
instead of `full_run.py run`. It requires `--gpus GPU_A GPU_B` and an explicit
`--expected-environment` JSON containing exact `python`, `torch`, `torchvision`,
`numpy`, `pandas`, `scipy`, `sklearn` version strings for the environment you intend
to use. Inspect `python ece_run.py --help`; no server account or SSH setup is bundled.
Finalization checks all checkpoint artifacts before generating reports.

## Checks and archival boundaries

```sh
python full_run.py --help
python full_results.py --help
python audit_results.py --help
python ece_run.py --help
python -m unittest discover -s tests
```

The tests use temporary fixtures, including small synthetic CPU training; they do
not constitute a new full-data run. Public-package preparation only checked
syntax/imports/CLI help and selected non-training fixtures; no experiment or data
download was launched for this packaging step.

Any original-run signatures and checkpoint hashes in accompanying archived
results describe that historical run. This source package supports a **fresh run**;
it does not include the checkpoint bytes needed to independently verify original
checkpoint hashes. `audit_results.py` checks saved reload records and predictions,
not checkpoint bytes. Resume/checkpoint checks concern the artifacts actually
available in the chosen fresh output directory. Do not overwrite archival results.

The public copy preserves the core algorithms and `configs/full.json` byte for
byte. Packaging changes are limited to requiring an explicit split config for the
standalone audit CLI, a reduced dependency list, this README, and selecting the
relevant source/tests. Remote login scripts, receipts, archives, private Drive
notebooks and the separate adaptation experiment are excluded.
