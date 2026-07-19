# JEPA Collapse Lab

> Why do joint-embedding models collapse — and how do Barlow Twins and VICReg prevent it?

![python](https://img.shields.io/badge/python-3.11+-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.x-orange)
![status](https://img.shields.io/badge/status-work%20in%20progress-yellow)

A small, self-contained experimental lab that **reproduces representation collapse** in a
joint-embedding architecture, **diagnoses it quantitatively**, and shows how two classic
non-contrastive objectives — **Barlow Twins** and **VICReg** — prevent it *without* negative
pairs, *without* stop-gradient, and *without* a predictor.

This is not a full JEPA (there is no predictor). It is the experiment that demonstrates you
understand the fundamental problem that motivated JEPA in the first place.

---

## The question

Given an image $x$, sample two augmented views $x_A = t_A(x)$ and $x_B = t_B(x)$, encode them
with the **same** encoder, and pull the embeddings together:

```
                 ┌──────────────────┐
 x ── t_A(x) ──► │                  ├──► h_A ──► Projector ──► z_A ──┐
                 │   Encoder E      │                                ├──► Loss
 x ── t_B(x) ──► │   (shared)       ├──► h_B ──► Projector ──► z_B ──┘
                 └──────────────────┘
```

There is a trivial shortcut: **output the same vector for every input**. The invariance loss
goes to zero, and the representation carries no information whatsoever. This is
**representation collapse**.

The interesting part is not that collapse happens — it is *how we measure it*, and *which
mechanisms provably prevent it*.

## The three experiments

All three runs share the **same backbone, same projector, same augmentations, same budget**.
Only the loss changes.

### Experiment A — Naive invariance (the collapse)

$$\mathcal{L}_{\text{inv}} = \|z_A - z_B\|_2^2$$

The network progressively discovers it can emit a constant vector. Expected signature:

- per-dimension variance of embeddings $\to 0$,
- all embeddings nearly identical,
- effective rank of the embedding matrix $\to 1$,
- linear-probe accuracy $\to$ chance level.

**Key lesson: a low loss does not mean a good representation.**

### Experiment B — Barlow Twins

Compute the cross-correlation matrix of the two embedding batches:

$$\mathcal{C}_{ij} = \frac{\sum_b z^A_{b,i}\, z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\,\sqrt{\sum_b (z^B_{b,j})^2}}$$

and push it toward the identity:

$$\mathcal{L}_{BT} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2$$

- **Diagonal terms** enforce invariance between the two views.
- **Off-diagonal terms** reduce redundancy between embedding dimensions.

A constant output cannot satisfy the unit diagonal without exploding the off-diagonal penalty,
so collapse is ruled out by construction.

### Experiment C — VICReg

Three explicit terms:

$$\mathcal{L} = \lambda\, \underbrace{s(Z_A, Z_B)}_{\text{invariance}} + \mu\, \underbrace{\big[v(Z_A) + v(Z_B)\big]}_{\text{variance}} + \nu\, \underbrace{\big[c(Z_A) + c(Z_B)\big]}_{\text{covariance}}$$

- **Invariance** $s$: MSE between the two views (pulls them together).
- **Variance** $v$: hinge loss that keeps the per-dimension standard deviation above a target $\gamma$ — each dimension is forbidden to become constant.
- **Covariance** $c$: penalizes off-diagonal covariance — dimensions are forbidden to copy the same information.

VICReg makes the fight against collapse fully explicit: invariance, variance, decorrelation.

## Setup

| Component      | Choice                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| Dataset        | **STL-10**: 100k unlabeled images for pretraining; labeled split used **only** for linear evaluation. CIFAR-10 as a cheap debug config. |
| Backbone       | ResNet-18 (small-image stem: 3×3 conv, no maxpool)                      |
| Projector      | MLP (e.g. 512→512→128; Barlow Twins benefits from a wider projector)   |
| Augmentations  | RandomResizedCrop, horizontal flip, color jitter, grayscale, blur      |
| Protocol       | Same backbone / augmentations / epochs / optimizer for A, B, C; fixed seeds |

## Diagnostics — how we *prove* collapse (or its absence)

| Diagnostic                | What it shows                                                        |
| ------------------------- | -------------------------------------------------------------------- |
| Per-dimension variance    | The collapse curve: std of each embedding dimension over training    |
| Covariance heatmap        | Redundant vs. decorrelated dimensions                                |
| Singular value spectrum   | How many directions the representation actually spans                |
| Effective rank            | $\mathrm{erank}(Z) = \exp(-\sum_i p_i \log p_i)$, $p_i = \sigma_i / \sum_j \sigma_j$ |
| UMAP / PCA projection     | Visual cluster structure (colored by true label)                     |
| Linear probe accuracy     | Frozen encoder + logistic regression on the labeled split            |

## Expected results

| Method          | Invariance loss ↓ | Mean per-dim std | Effective rank | Linear probe acc |
| --------------- | ----------------- | ---------------- | -------------- | ---------------- |
| A — Naive       | → 0 *(degenerate)*| → 0              | → 1            | ~ chance (≈10 %) |
| B — Barlow Twins| low, healthy      | ≈ 1              | high           | good             |
| C — VICReg      | low, healthy      | ≈ γ (hinged)     | high           | good             |

The visual punchline: **the naive model collapses, the other two don't** — visible in one
variance curve, one heatmap, one spectrum.

## Repository structure (planned)

```
├── configs/            # one config per experiment (naive / barlow_twins / vicreg)
├── src/
│   ├── data/           # STL-10 / CIFAR-10 loaders, augmentation pipeline
│   ├── models/         # ResNet-18 backbone, MLP projector
│   ├── losses/         # naive, barlow_twins, vicreg
│   ├── diagnostics/    # variance, covariance, spectrum, effective rank, UMAP
│   └── eval/           # linear probe
├── scripts/            # train.py, diagnose.py, make_figures.py
├── tests/
└── results/            # figures & metrics
```

## Quickstart

```bash
uv sync

# train the three variants (same budget, same seed)
uv run scripts/train.py experiment=naive
uv run scripts/train.py experiment=barlow_twins
uv run scripts/train.py experiment=vicreg

# diagnostics + linear probe + figures
uv run scripts/diagnose.py --all
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full phased plan and the current status.

## What this project demonstrates

- what representation collapse is, and how to **diagnose** it (not just observe it);
- why a low SSL loss is **not** evidence of a good representation;
- the role of **variance** and **decorrelation** in non-contrastive learning;
- the conceptual bridge toward JEPA: predicting in representation space only works if the
  representation space hasn't collapsed.

## References

- Zbontar et al., [*Barlow Twins: Self-Supervised Learning via Redundancy Reduction*](https://arxiv.org/abs/2103.03230), ICML 2021.
- Bardes et al., [*VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*](https://arxiv.org/abs/2105.04906), ICLR 2022.
- LeCun, [*A Path Towards Autonomous Machine Intelligence*](https://openreview.net/forum?id=BZ5a1r-kVsf), 2022.
- Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)*](https://arxiv.org/abs/2301.08243), CVPR 2023.
