
# Identity-Disjoint Audio-Visual Deepfake Detection

Simple Recursive Fusion Outperforms Attention Under Static Biometric Descriptors

IJCB 2026 Submission

---

## Overview

This repository contains the full implementation, results, and figures for our IJCB 2026 paper. We study which multimodal fusion architecture works best for audio-visual deepfake subtype detection when face and voice are encoded as static biometric descriptors, without any temporal modeling.

Seven supervised model families are compared under a strict identity-disjoint protocol on FakeAVCeleb v1.2. The main finding is that simple GRU-based recursive fusion (Simplified TRM, Macro F1 = 0.65) outperforms deeper cross-modal attention for 4-class subtype diagnosis. Full TRM-VFD achieves the strongest binary consistency ranking (AUC = 0.82). Neither architecture overcomes the real-class collapse caused by the mismatch between identity-optimised embeddings and authenticity detection.

---

## Results Summary

| Model | Macro F1 | Binary AUC | Note |
|---|---|---|---|
| Simplified TRM | 0.65 | 0.79 | Best 4-class model |
| Two-Stage MLP | 0.64 | 0.76 | Hierarchical baseline |
| Embedding MLP | 0.63 | 0.77 | Strongest simple baseline |
| One-Shot Transformer | 0.60 | 0.79 | Single attention pass |
| Full TRM-VFD | 0.59 | 0.82 | Best binary AUC |
| Improved Full TRM | 0.54 | 0.75 | Tokenised, underperforms |
| Pixel MLP | 0.18 | 0.61 | Lower bound |

Test set: n = 2,000 identity-disjoint samples. Primary metric: Macro F1.

---

## Dataset

FakeAVCeleb v1.2 is a synchronized audio-visual deepfake dataset with four manipulation classes: real, fake-audio, fake-video, and fake-both.

We use an identity-disjoint split with no identity overlap across partitions.

| Split | Samples | Identities |
|---|---|---|
| Train | 17,017 | 400 |
| Validation | 2,100 | 50 |
| Test | 2,000 | 50 |

Class counts per split: train [398, 397, 7705, 8517], val [50, 49, 934, 1067], test [50, 50, 901, 999].

Real and fake-audio classes together make up only about 4.7% of training data. The main results use the natural class distribution without resampling. Imbalance ablations are in supplementary cells 45-52 and are not used to define the core ranking.

Dataset access: https://github.com/DASH-Lab/FakeAVCeleb

For the cross-dataset transfer probe, Celeb-DF v2 is used with voice embeddings set to zeros (face-only evaluation). Access: https://github.com/yuezunli/celeb-deepfakeforensics

---

## Architecture

Each video is represented by a single static 704-dimensional descriptor:

- ArcFace (IR-SE50) produces a 512-dimensional face embedding (L2-normalised)
- ECAPA-TDNN produces a 192-dimensional voice embedding (L2-normalised)
- Both embeddings are concatenated to form the 704-dimensional multimodal descriptor

The descriptor is passed to one of seven supervised model families:

1. Pixel MLP - raw 64x64 RGB crops through a fully connected MLP (lower bound)
2. Embedding MLP - 704-d descriptor through MLP (strongest simple baseline)
3. One-Shot Transformer - single cross-attention pass with face as query, voice as key/value
4. Simplified TRM - 4-step GRU recursion, no attention (best 4-class model)
5. Full TRM-VFD - cross-modal attention before each of 5 GRU steps (best binary AUC)
6. Improved Full TRM - tokenised embeddings with recursive attention (underperforms)
7. Two-Stage MLP - Stage 1 classifies real vs fake, Stage 2 classifies fake subtype

Binary-only reference baselines (not ranked by Macro F1): VFD Cosine, MVF.

---

## Installation

### Google Colab (recommended)

Open TRM_DETECTION_public_final.ipynb in Google Colab. Cells 1-3 handle all dependency installation automatically. You need a Google Drive account with at least 20 GB of free space for checkpoints and cached embeddings.

### Local setup

```bash
