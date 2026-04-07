# TRM-VFD: Temporal Recursive Model for Video-Face Deepfake Detection

> **Recursive Cross-Modal Biometric Identity Verification for Robust Audiovisual Deepfake Detection**
> Sravani Gurram — San José State University

##  Overview

TRM-VFD detects deepfake videos by verifying whether the **face identity** and **voice identity** within the same video belong to the same real person.

Unlike traditional binary classifiers, this framework models deepfake detection as a:

> **Cross-modal biometric identity consistency problem**

This makes the system robust against:

* Face-only deepfakes
* Audio-only deepfakes
* High-quality single-modality spoofing

##  Key Idea

A deepfake video often contains:

* Fake face + real voice
* Real face + fake voice

Unimodal systems fail here.

TRM-VFD solves this by:

* Extracting **ArcFace embeddings (face)**
* Extracting **ECAPA-TDNN embeddings (voice)**
* Verifying identity consistency across modalities

## Repository Structure

```
TRM-VFD/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── TRM_DETECTION.ipynb
│   ├── EDA.ipynb
│   └── MVF_baseline.ipynb
├── src/
│   ├── data/
│   │   ├── load_fakeavceleb.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── mlp_baseline.py
│   │   ├── mvf_baseline.py
│   │   └── trm.py
│   └── evaluate.py
├── configs/
│   └── FakeAVCeleb/
│       ├── finetune_full.yaml
│       └── test_full.yaml
└── results/
    └── baseline_results.json

## Dataset

**FakeAVCeleb v1.2 (DASH Lab, KAIST)**

| Property     | Value                                  |
| ------------ | -------------------------------------- |
| Total videos | 21,544                                 |
| Identities   | 500                                    |
| Dataset size | ~19 GB                                 |
| Labels       | 0 = Real, 1 = FakeAudio, 2 = FakeVideo |

## Key Design Choice

* Identity-disjoint split**

* Train: 400 identities
* Validation: 100 identities

# Prevents identity leakage (critical for biometrics)

## Installation

bash
git clone https://github.com/<your-username>/TRM-VFD.git](https://github.com/Gurram-sravani/ijcb-deepfake-detection.git
cd TRM-VFD

pip install -r requirements.txt

git clone https://github.com/xaCheng1996/MVF.git


## Running Experiments

## 1. Data Preparation (Colab)

python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/FakeAVCeleb_v1.2.zip') as z:
    z.extractall('/content/dataset')

## 2. MLP Baseline

Run:

notebooks/TRM_DETECTION.ipynb


Includes:

* 3 MLP variants
* imbalance experiments
* confusion matrix + metrics


## 3. MVF Baseline

bash
cd MVF

python finetune_deepfake.py \
  --config configs/FakeAVCeleb/finetune_full.yaml \
  --log_time fullrun1


## Baseline Results

| Model         | Accuracy | Balanced Acc | Macro F1 | AUC   | Insight           |
| ------------- | -------- | ------------ | -------- | ----- | ----------------- |
| MLP (plain)   | 0.908    | 0.333        | 0.317    | —     | Majority collapse |
| MLP + weights | 0.908    | 0.333        | 0.317    | —     | No improvement    |
| MLP + sampler | 0.046    | 0.333        | 0.029    | —     | Minority collapse |
| MVF           | 0.531    | —            | —        | 0.568 | Undertrained      |

## Key Insight

> Accuracy is misleading under imbalance
> Balanced Accuracy + Macro F1 are critical


##  Known Issues

* Severe class imbalance (~20:1)
* MLP fails → weak feature representation
* MVF undertrained (10 epochs only)

##  Research Gap

> Existing models either:
>
> * Perform **one-shot scoring without reasoning**, OR
> * Use **recursive reasoning without cross-modal identity modeling**

**TRM-VFD integrates both**

## Roadmap

* [x] Dataset + preprocessing
* [x] MLP baseline
* [x] MVF baseline
* [ ] Tuned MVF
* [ ] AV-HuBERT baseline
* [ ] TRM model (main contribution)


## References

* FakeAVCeleb Dataset (NeurIPS 2021)
* Voice–Face Homogeneity (ACM MM 2023)
* AV-HuBERT (NeurIPS 2022)
* Tiny Recursive Models (SAIT)
* Deepfake Detection Survey (IEEE)



## 🔗 Code Repository

https://github.com/Gurram-sravani/ijcb-deepfake-detection/blob/main/notebooks/TRM_DETECTION.ipynb

##  License

Academic use only.
FakeAVCeleb dataset subject to its own license.
