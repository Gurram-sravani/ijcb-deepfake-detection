# Identity-Disjoint Audio-Visual Deepfake Detection

**An Empirical Study of Biometric Embeddings, Recursive Fusion, and Task Decomposition**

> IJCB 2026 Submission


## Overview

This repository contains the full Colab notebook and supporting code for our empirical study of audio-visual deepfake detection under a strict identity-disjoint protocol. We compare 8 model families — from raw-pixel MLPs to recursive cross-modal attention models — all trained and evaluated on the same canonical split of FakeAVCeleb v1.2.

**Key finding:** Task decomposition (Two-Stage MLP Pipeline) achieves the best 4-class Macro F1. Recursive GRU fusion without attention (Simplified TRM) is the best single-stage model. Full TRM-VFD leads on binary consistency ranking (AUC). Simple architectures outperform deeper recursive attention on the primary 4-class metric.


## Results

### Table 1 — Main 4-Class Results (Test set, n=2,000)

| Model | Accuracy | Balanced Acc. | Macro F1 | Binary AUC |
|---|---|---|---|---|
| Pixel MLP (lower bound) | 0.34 | 0.30 | 0.19 | 0.61 |
| Improved Full TRM | 0.90 | 0.55 | 0.54 | 0.75 |
| Full TRM-VFD | 0.91 | 0.61 | 0.59 | **0.82** |
| One-Shot Transformer | 0.91 | 0.61 | 0.60 | 0.79 |
| Embedding MLP | 0.92 | 0.62 | 0.63 | 0.77 |
| Simplified TRM | 0.94 | 0.63 | 0.65 | 0.79 |
| **Two-Stage MLP Pipeline** | **0.94** | **0.64** | **0.66** | 0.77 |

### Table 2 — Face-Only Transfer to Celeb-DF v2

| Model | AUC |
|---|---|
| Embedding MLP | 0.54 |
| Two-Stage MLP Pipeline | 0.60 |
| Full TRM-VFD | 0.62 |
| **Simplified TRM** | **0.65** |

*Voice embeddings set to zeros — face-only evaluation. AUC only.*


## Dataset

**FakeAVCeleb v1.2** — 21,544 videos, 500 identities, 4 classes:

| Label | Class | Description |
|---|---|---|
| 0 | real | Real face + real voice |
| 1 | fake_audio | Real face + fake voice |
| 2 | fake_video | Fake face + real voice |
| 3 | fake_both | Fake face + fake voice |

Download: [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)



## Split Protocol

```
Total identities: 500
Train: 400 identities → 17,017 samples
Val:    50 identities →  2,100 samples  (model selection only)
Test:   50 identities →  2,000 samples  (touched once per model)

Zero identity overlap between splits.
Class distribution (train): [398, 397, 7705, 8517]
Class distribution (test):  [50, 50, 901, 999]
```



## Embeddings

Each video is represented by a **704-dimensional biometric descriptor**:

```
Face:  ArcFace buffalo_l    → 512-dim  (InsightFace)
Voice: ECAPA-TDNN VoxCeleb  → 192-dim  (SpeechBrain)
Combined: concat([face, voice]) → 704-dim
```

Note: `normed_embedding` is used (L2-normalized), not raw `.embedding`.



## Models

| Model | Type | Params | Notes |
|---|---|---|---|
| Pixel MLP | Raw pixel | 6.4M | 64×64 RGB → 12,288-dim lower bound |
| Embedding MLP | Flat MLP | ~200k | 704→512→256→4 |
| VFD Cosine | Binary | — | Voice→face projection, binary-only |
| MVF | Binary | — | Published reference, binary-only |
| One-Shot Transformer | Attention | ~200k | Single cross-modal attention pass |
| Simplified TRM | GRU | ~789k | Recursion without attention, K=4 |
| Full TRM-VFD | Attention+GRU | ~856k | Cross-modal attention + GRU, K=5 |
| Two-Stage MLP | Hierarchical | ~400k | Stage1: real/fake, Stage2: subtype |
| Improved Full TRM | Tokenized | ~318k | 8×64 + 6×32 tokens → 128-dim |



## Quick Start

### 1. Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `TRM_DETECTION_public_final.ipynb` to Colab.

### 2. Upload dataset to Drive

```
Google Drive path: MyDrive/FakeAVCeleb_v1.2.zip
```

### 3. Run cells in order

```
Phase 0  → Imports
Phase 1  → Build identity-disjoint split (~60 min)
Phase 2  → EDA figures (optional)
Phase 3  → Pixel MLP
Phase 4  → MVF baseline (skip if checkpoint exists)
Phase 5  → Extract embeddings (~60 min, once only)
Phase 6  → Embedding MLP
Phase 7  → VFD Cosine
Phase 8  → One-Shot Transformer
Phase 9  → Simplified TRM
Phase 10 → Full TRM-VFD, Two-Stage MLP, Improved Full TRM
Phase 11 → Tables, figures, ablation, transfer probe
```

Total runtime: ~5 hours on T4 GPU.



## File Structure

```
TRM_DETECTION_public_final.ipynb   Main notebook (44 cells)
requirements.txt                    Python dependencies
README.md                           This file
```



## Requirements

See `requirements.txt`. Key dependencies:

```
torch >= 2.0
insightface
speechbrain
scikit-learn
```



## Training Details

```
Optimizer:    Adam (lr=5e-4, weight_decay=1e-4)
Loss:         FocalLoss (gamma=2.0) for all except Improved TRM
Sampler:      WeightedRandomSampler (1.5x boost: real + fake_audio)
Val protocol: Model selection only — test touched once per model
Seeds:        42 (primary), 123, 456 (stability study)
```

**Why 1.5x minority boost?**
- 1.0x → real class F1 collapsed to 0.03
- 1.5x → stable real F1 ~0.05–0.08
- 2.0x → loss oscillation, unstable training



## Binary Reference Baselines

**VFD Cosine** and **MVF** are binary-only models — they predict face-voice consistency, not 4-class subtypes. They are excluded from Table 1 and reported separately as reference baselines. AUC is derived from their learned similarity scores.



## Citation

```bibtex
@inproceedings{trm2026,
  title     = {Identity-Disjoint Audio-Visual Deepfake Detection:
               An Empirical Study of Biometric Embeddings,
               Recursive Fusion, and Task Decomposition},
  booktitle = {IJCB 2026},
  year      = {2026}
}
```

---

## License

This code is released for research purposes only.
