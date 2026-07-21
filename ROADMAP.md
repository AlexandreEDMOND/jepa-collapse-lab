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

## Phase 4 — Collapse diagnostics ✅

- [x] embedding collection on a fixed eval set
- [x] per-dimension variance curve over training (`z_std` from history + per-dim bar)
- [x] covariance heatmap
- [x] singular value spectrum of the embedding matrix
- [x] effective rank (scalar + spectrum figure)
- [x] UMAP / PCA projection colored by true label

**Done when:** each diagnostic is one function + one figure, runnable on any checkpoint.
→ `uv run scripts/diagnose.py --checkpoint results/checkpoints/<run>/last.pt`

## Phase 5 — Linear evaluation ✅

- [x] freeze encoder, train logistic regression on the labeled train split
- [x] report test accuracy (+ per-class)

**Done when:** a single number per run lands in the results table.
→ `uv run scripts/probe.py --checkpoint results/checkpoints/<run>/last.pt`

## Phase 6 — Experiments & figures ✅

- [x] run A (naive), B (Barlow Twins), C (VICReg) with identical budget
- [x] generate every README figure: variance curve, heatmap, spectrum, effective rank, UMAP
- [x] results table (loss, std, rank, linear probe)

**Done when:** the collapse story is visible in the figures without any explanation.
→ STL-10, 10 epochs, bs=256, seed 0. Table: `results/results_table.md`; figures:
`results/figures/stl10_*/`. Naive: z_std 0,004, probe(h) 42,9 % (random features),
UMAP dégénéré. Barlow Twins: z_std 0,86, erank 61,6, probe(h) 71,8 %. VICReg:
z_std 1,00 (= γ), erank 87,6, probe(h) 73,5 %.

## Phase 7 — Polish & release ✅

- [x] fill the README results section
- [x] short write-up of findings and surprises
- [x] tag `v1.0`

---

## Where to start

**Projet terminé — v1.0.** Phases 0–7 complètes. Résultats dans le README (section
*Results* / *Findings and surprises*), table dans `results/results_table.md`, figures
dans `results/figures/`. Les notes de handoff ci-dessous retracent les décisions
(module `data/` reconstruit, gate erank, budget 10 epochs / bs=256 sur RTX 3090).

### Handoff notes (2026-07-20) — read before Phase 6

Moving to the Linux + RTX 3090 machine for the full-budget runs (MPS on the MacBook is
too slow, see pitfall below). State: Phases 0–5 done, 40/40 tests green, CIFAR-10 debug
runs (5 epochs) done for all three experiments.

CIFAR-10 debug results (5 epochs, `results/checkpoints/cifar10_*/` on the Mac — not
committed, numbers copied here):

| run            | z_std (first→last) | loss (last) | mean_std | eff. rank | probe acc (h) |
| -------------- | ------------------ | ----------- | -------- | --------- | ------------- |
| naive          | 0.072 → 0.0077     | 1.1e-4      | 0.0079   | **109.6** | **40.8 %**    |
| barlow_twins   | 0.79 → 1.05        | 19.4        | 1.15     | 20.7      | 49.4 %        |
| vicreg         | 0.56 → 0.74        | 26.7        | 0.75     | 49.9      | 52.2 %        |

Two open questions to settle BEFORE the 3 STL-10 runs:

1. **Naive probe is 40.8 %, not ~chance (10 %).** The probe reads `h` (backbone), the
   collapse is measured on `z` (projector). Hypothesis: the projector collapses to a
   constant map while the backbone stays near its random init (random-feature probe on
   CIFAR-10 is ~30–40 %). Test: `uv run scripts/train.py --config configs/cifar10_debug.yaml
   --experiment naive --epochs 20 --output-dir results/checkpoints_long`, then
   `diagnose.py` + `probe.py` on that checkpoint. If probe(h) stays ~40 % while z_std → 0,
   that's a finding for the write-up (collapse in z ≠ collapse in h); consider also
   reporting probe(z) in the results table.
2. **Effective rank is misleading on collapsed runs** (naive shows 109.6 because the SVD
   of a near-constant matrix sees only numerical noise, whose spectrum is almost flat).
   Options: threshold tiny singular values, report erank only when mean_std is above a
   noise floor, or keep the metric and explain the artefact. Decide after seeing the
   naive-long numbers.

On the RTX machine, in order:

```bash
uv sync && uv run pytest            # sanity
# 1. STL-10 smoke (validates pipeline + measures s/step; ~390 batches/epoch at bs=256):
uv run scripts/train.py --config configs/stl10.yaml --experiment naive --epochs 1 \
    --max-batches 20 --output-dir /tmp/stl10_smoke
# 2. Naive-long CIFAR-10 test (question 1 above), then diagnose + probe on it
# 3. The three full-budget runs (identical budget!):
uv run scripts/train.py --config configs/stl10.yaml --experiment naive
uv run scripts/train.py --config configs/stl10.yaml --experiment barlow_twins
uv run scripts/train.py --config configs/stl10.yaml --experiment vicreg
```

With 24 GB VRAM, `batch_size: 512` (or 1024) fits easily and Barlow Twins likes large
batches — but keep it identical across the three runs.

**macOS pitfall (why the move):** STL-10 `unlabeled` is a 2.9 GB in-memory numpy array;
on macOS DataLoader workers use *spawn*, so the whole dataset is pickled to each worker
(4 workers ≈ 12 GB copied) → training appears to hang before batch 1. On Linux the
default *fork* start method shares memory copy-on-write, so `num_workers: 4` is fine.
If back on macOS: set `num_workers: 0` in `configs/stl10.yaml`.

### Handoff notes (2026-07-20, soir) — RTX 3090, après premiers essais

**Le module `data/` a dû être reconstruit** : `data/` dans `.gitignore` matchait aussi
`src/jepa_collapse_lab/data/`, le package n'avait jamais été commité. Recréé
(`augmentations.py`, `loaders.py`), `.gitignore` corrigé (`data/` → `/data/`), 41/41 tests.

**Réponses aux deux questions ouvertes** (naive-long CIFAR-10, 20 epochs,
`results/checkpoints_long/cifar10_naive/`) :

1. probe(h) = **41,9 %** alors que z est totalement collapssé (mean_std = 0,0009) → le
   collapse a lieu dans le projecteur ; le backbone reste proche de son init (niveau
   random-features). probe(z) = 32,9 % ≠ 10 % parce que le `StandardScaler` du probe
   ré-amplifie les variations résiduelles ~1e-4 — à mentionner dans le write-up.
2. erank(collapsé) = 37,1 (bruit numérique, spectre quasi-plat) → **erank gaté** dans
   `summarize_embeddings` : `None` + `"collapsed": true` quand mean_std < 1e-2
   (`COLLAPSE_STD_FLOOR` dans `diagnostics/metrics.py`). La métrique elle-même est
   inchangée ; le titre du spectre affiche « erank=n/a (collapsed) ».

**Budget des runs STL-10 (décision du soir) :** bs=512/1024 ne rentre **pas** en 24 Go
(fp32 eager : ResNet-18 sans maxpool garde des activations 64×96×96 ×2 vues ≈ 20,5 Go
à bs=256, mesuré). Mesure réelle : 1,22 it/s → ~5,3 min/epoch à bs=256. 50 epochs
(4,4 h/run) jugé trop long → **budget réduit à 10 epochs (~53 min/run, ~2 h 40 total)**,
toujours identique pour les trois runs. Le naive collappe dès l'epoch ~5, 10 epochs
suffisent au contraste BT/VICReg.

**Commandes à lancer (demain matin, ~2 h 40 au total) :**

```bash
cd ~/CodePuant/jepa-collapse-lab
uv run scripts/train.py --config configs/stl10.yaml --experiment naive        --epochs 10
uv run scripts/train.py --config configs/stl10.yaml --experiment barlow_twins --epochs 10
uv run scripts/train.py --config configs/stl10.yaml --experiment vicreg       --epochs 10
```

Puis diagnostics + probes (~15 min) :

```bash
uv run scripts/diagnose.py --all --checkpoints-root results/checkpoints
uv run scripts/probe.py    --all --checkpoints-root results/checkpoints            # probe(h)
uv run scripts/probe.py    --all --checkpoints-root results/checkpoints --space z  # probe(z)
```

### Résultats STL-10 (2026-07-21) — Phase 6 terminée

Runs séquentiels ~53 min chacun (1,22 it/s). Table complète : `results/results_table.md`.

| run | loss | z_std | mean_std | erank | probe(h) | probe(z) |
| --- | --- | --- | --- | --- | --- | --- |
| naive | 2,6e-5 | 0,0039 | 0,0056 | n/a (collapsed) | 42,9 % | 33,1 % |
| barlow_twins | 2,81 | 0,86 | 0,91 | 61,6 | 71,8 % | 66,1 % |
| vicreg | 9,08 | 1,00 | 1,03 | 87,6 | 73,5 % | 67,8 % |

Confirmé sur STL-10 : le probe(h) du naive reste au niveau random-features (~43 %) —
le collapse est dans le projecteur, pas le backbone. probe(z) du naive = 33 % ≠ 10 % :
artefact du `StandardScaler` qui ré-amplifie les variations résiduelles (à expliquer
dans le write-up, Phase 7). Figures dans `results/figures/stl10_*/` (le spectre du
naive affiche « erank=n/a (collapsed) », UMAP naive dégénéré vs clusters VICReg).
