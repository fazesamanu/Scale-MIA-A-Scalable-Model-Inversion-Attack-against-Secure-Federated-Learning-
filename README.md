# Scale-MIA Reproducibility & Extension Project

## Project Title

**Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction**

---

# Overview

This project reproduces the NDSS 2025 research paper:

> **“Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction”**

The goal of the project is to:

* reproduce the original attack pipeline,
* validate the published results,
* analyze scalability,
* visualize reconstruction quality,
* and extend the repository with reproducibility and visualization improvements.

---

# Project Objectives

## Primary Objective — Exact Reproducibility

The following components from the paper were reproduced:

* Secure Federated Learning setup
* Scale-MIA attack pipeline
* Latent space reconstruction
* Reconstruction quality evaluation
* Scalability experiments
* Multiple reconstruction batch sizes
* Reconstruction visualization

---

## Secondary Objective — Novel Engineering Extensions

Additional work was implemented beyond the original repository:

### 1. Dynamic Reconstruction Saving

Implemented automatic batch-wise reconstruction tensor saving.

### 2. Generalized Visualization Pipeline

Implemented scalable visualization support for arbitrary batch sizes.

### 3. Reproducibility Sensitivity Analysis

Investigated the effect of federated local epochs on reconstruction quality.

---

# What is Scale-MIA?

Scale-MIA is a **Model Inversion Attack** against Federated Learning.

Even when:

* clients do not share raw data,
* and secure aggregation is enabled,

the malicious server can still reconstruct private client images from model updates.

---

# Attack Pipeline

```text
Original Client Image
        ↓
Encoder Network
        ↓
Latent Representation
        ↓
Gradient Leakage
        ↓
Latent Reconstruction
        ↓
Decoder Network
        ↓
Recovered Image
```

---

# Datasets Used

| Dataset      | Purpose                          |
| ------------ | -------------------------------- |
| FashionMNIST | Primary reproducibility          |
| CIFAR-10     | Complex color reconstruction     |
| HMNIST       | Medical/privacy-sensitive images |

---

# System Requirements

## Hardware

* NVIDIA RTX 4060 Laptop GPU
* 8GB VRAM recommended
* WSL2 Ubuntu

---

## Software

| Package     | Version     |
| ----------- | ----------- |
| Python      | 3.10        |
| PyTorch     | 2.5.1+cu121 |
| CUDA        | 12.1        |
| torchvision | 0.20.1      |

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/unknown123489/Scale-MIA.git
cd Scale-MIA
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scipy datasets opacus einops
```

---

# Important Fixes Applied

## CUDA Multi-GPU Checkpoint Fix

The original repository was trained on multi-GPU systems (`cuda:1`).

The following fix was applied to all `torch.load()` calls:

```python
torch.load(PATH, map_location=device)
```

This resolved:

* CUDA deserialization errors
* GPU mismatch crashes

---

# Running Experiments

# FashionMNIST Experiments

Move into folder:

```bash
cd fedavg-fmnist
```

---

## Run Reconstruction Attack

### Batch Size 64

```bash
python fedavg-recover-attack.py \
--batch_size=64 \
--local_epoch=1 \
--test_rounds=3
```

---

### Batch Size 128

```bash
python fedavg-recover-attack.py \
--batch_size=128 \
--local_epoch=1 \
--test_rounds=3
```

---

### Batch Size 256

```bash
python fedavg-recover-attack.py \
--batch_size=256 \
--local_epoch=1 \
--test_rounds=3
```

---

### Batch Size 512

```bash
python fedavg-recover-attack.py \
--batch_size=512 \
--local_epoch=1 \
--test_rounds=3
```

---

### Batch Size 1024

```bash
python fedavg-recover-attack.py \
--batch_size=1024 \
--local_epoch=1 \
--test_rounds=3
```

---

# Reproduced FashionMNIST Results

## FedSGD (iter=1)

| Batch Size | Reconstruction Rate | PSNR (dB) | Attack Time |
| ---------- | ------------------- | --------- | ----------- |
| 64         | 0.9479              | 39.80     | 0.418 s     |
| 128        | 0.8333              | 35.90     | 0.448 s     |
| 256        | 0.7813              | 34.96     | 0.356 s     |
| 512        | 0.6777              | 32.44     | 0.410 s     |
| 1024       | 0.4300              | 29.65     | 0.597 s     |

---

# Visualization

The visualization system was upgraded to support:

* arbitrary batch sizes,
* organized experiment storage,
* high-resolution reconstruction figures.

---

## Generate Reconstruction Figures

### Batch 64

```bash
python visual.py --batch_size=64 --iter=1
```

### Batch 512

```bash
python visual.py --batch_size=512 --iter=1
```

### Batch 1024

```bash
python visual.py --batch_size=1024 --iter=1
```

---

# Generated Results

Figures are automatically saved inside:

```text
results/fmnist/iter1/
```

Example:

```text
bs64_original.png
bs64_reconstructed.png
bs512_original.png
bs512_reconstructed.png
bs1024_original.png
bs1024_reconstructed.png
```

---

# Novel Work Implemented

## 1. Dynamic Tensor Saving

### Problem

The original repository overwrote reconstruction tensors for every run.

### Solution

Implemented dynamic tensor saving:

```python
torch.save(original_data, f"data/data_batch_{batch_size}.pt")
torch.save(recovered_data, f"data/recoved_{batch_size}.pt")
torch.save(sorted_list, f"data/list_{batch_size}.pt")
```

### Benefits

* scalable reconstruction storage
* reproducible experiment tracking
* no data overwrite
* easier comparative analysis

---

## 2. Generalized Visualization Framework

### Problem

Original visualization only supported batch size 64.

### Improvements

Implemented:

* arbitrary batch-size support
* automatic result organization
* publication-quality figure saving
* scalable reconstruction plotting

### Benefits

* easier report generation
* scalability visualization
* cleaner reproducibility workflow

---

## 3. Reproducibility Sensitivity Analysis

### Observation

Pretrained artifacts reproduce:

```text
FedSGD(iter=1)
```

very accurately.

However:

* iter=3
* iter=5

show noticeable reconstruction degradation.

### Interpretation

This suggests:

* latent reconstruction quality depends strongly on federated optimization dynamics,
* and attack parameter transferability is limited across FL settings.

This became an important reproducibility finding.

---

# Folder Structure

```text
Scale-MIA/
│
├── fedavg-fmnist/
├── fedavg-cifar/
├── fedavg-hmnist/
│
├── results/
│   ├── fmnist/
│   │   ├── iter1/
│   │   ├── iter3/
│   │   └── iter5/
│   │
│   ├── cifar/
│   └── hmnist/
│
├── data/
├── models/
└── figs/
```

---

# Key Findings

1. Scale-MIA successfully reconstructs private images from FL updates.

2. Reconstruction remains effective even for extremely large batches.

3. Attack runtime remains very small even at high scales.

4. Reconstruction quality degrades gradually as batch size increases.

5. Reconstruction performance is sensitive to local federated optimization settings.

---

# Future Work

Potential future extensions include:

* adaptive latent noise defenses,
* differential privacy integration,
* transformer-based latent decoders,
* diffusion prior reconstruction,
* cross-client aggregation attacks.

---

# References

* NDSS 2025 Scale-MIA Paper
* Official Scale-MIA GitHub Repository
* PyTorch Documentation
* Federated Learning Literature
