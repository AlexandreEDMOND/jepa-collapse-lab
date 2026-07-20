# Roadmap

Status legend: `[ ]` todo · `[x]` done

## Phase 0 — Project scaffolding ✅

- [x] `uv init`, dependencies: `torch`, `torchvision`, `matplotlib`, `scikit-learn`,
      `umap-learn`, `einops`, `pyyaml`, `tqdm`, `pytest`, `ruff`
- [x] repo layout: `src/`, `scripts/`, `configs/`, `tests/`, `results/`
- [x] lint config (ruff) + minimal CI (optional)
- [x] smoke test: forward pass of a tiny model on a random batch

**Done when:** `uv run pytest` passes and the repo layout exists. → 8/8 tests green.

## Phase 1 — Data pipeline ✅

- [x] STL-10 loaders: `unlabeled` split for SSL pretraining, labeled `train`/`test` kept aside for the probe
- [x] augmentation pipeline producing two views: RandomResizedCrop, hflip, color jitter, grayscale, (blur)
- [x] CIFAR-10 debug config (faster iteration)
- [x] visualization script: show augmented pairs side by side
- [x] tests: batch shapes, both views come from the same image

**Done when:** a script displays trustworthy augmented pairs and the labeled data is untouched by pretraining. → verified on both datasets, see `results/figures/augmented_pairs_*.png`.

## Phase 2 — Model ✅

- [x] ResNet-18 backbone adapted for small images (3×3 stem, no maxpool) — STL-10 at 96×96
- [x] MLP projector with BatchNorm in hidden layers (as in BT / VICReg)
- [x] checkpoint save/load, reproducible init

**Done when:** backbone + projector produce the expected embedding shapes. → 16/16 tests green; `forward` returns `h` (512-d, for the probe) and `z` (128-d, for the losses); 12.9M params total.

## Phase 3 — Losses & training loop ✅

- [x] naive invariance loss: `||z_A - z_B||²`
- [x] Barlow Twins loss: batch cross-correlation, on-diagonal + λ · off-diagonal terms
- [x] VICReg loss: invariance + variance hinge (target γ) + covariance penalty, weights (λ, μ, ν)
- [x] single config-driven trainer: per-term logging, checkpointing, fixed seeds

**Done when:** the three experiments train through the same trainer, differing only by config.
→ `uv run scripts/train.py --config configs/cifar10_debug.yaml --experiment {naive,barlow_twins,vicreg}`

## Phase 4 — Collapse diagnostics

- [ ] embedding collection on a fixed eval set
- [ ] per-dimension variance curve over training
- [ ] covariance heatmap
- [ ] singular value spectrum of the embedding matrix
- [ ] effective rank over training
- [ ] UMAP / PCA projection colored by true label

**Done when:** each diagnostic is one function + one figure, runnable on any checkpoint.

## Phase 5 — Linear evaluation

- [ ] freeze encoder, train logistic regression on the labeled train split
- [ ] report test accuracy (+ per-class)

**Done when:** a single number per run lands in the results table.

## Phase 6 — Experiments & figures

- [ ] run A (naive), B (Barlow Twins), C (VICReg) with identical budget
- [ ] generate every README figure: variance curve, heatmap, spectrum, effective rank, UMAP
- [ ] results table (loss, std, rank, linear probe)

**Done when:** the collapse story is visible in the figures without any explanation.

## Phase 7 — Polish & release

- [ ] fill the README results section
- [ ] short write-up of findings and surprises
- [ ] tag `v1.0`

---

## Where to start

**Phase 4 — Collapse diagnostics.** Losses and the trainer are in place. Next: collect
embeddings from a fixed eval set and plot variance / covariance / spectrum / effective rank
so the naive collapse is visible next to BT and VICReg.
