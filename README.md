# PhySATFormer

**Physics-Guided Spatio-Temporal Transformer for Satellite Telemetry Anomaly Localization**

PhySATFormer is a physics-informed deep learning architecture for fine-grained anomaly localization in satellite telemetry. Unlike conventional time-series anomaly detectors that produce a single anomaly score for an entire sequence, PhySATFormer identifies **which telemetry channels are anomalous at each timestamp** by combining spacecraft subsystem knowledge with Transformer-based temporal modeling.

---

## Motivation

Satellite telemetry consists of hundreds of interdependent sensor channels. Faults rarely occur in isolation—they propagate across physically connected subsystems.

Traditional Transformers learn these relationships solely from data.

PhySATFormer incorporates **explicit physics priors** into the attention mechanism, allowing the model to reason about known subsystem relationships while simultaneously learning temporal fault evolution.

---

## Key Features

- Physics-guided channel attention
- Spatio-temporal Transformer architecture
- Channel-level anomaly localization
- Physics relationship matrix integration
- Modular preprocessing pipeline
- Production-quality PyTorch implementation
- Explainability-ready architecture
- IEEE conference research implementation

---

# Architecture

```
Telemetry Window
(B × T × C)
        │
        ▼
Telemetry Channel Encoder
        │
        ▼
Physics-Guided Channel Attention
        │
        ▼
Channel Attention Block(s)
        │
        ▼
Channel Pool
        │
        ▼
Positional Encoding
        │
        ▼
Temporal Transformer Encoder
        │
        ▼
Linear Prediction Head
        │
        ▼
Channel-wise Anomaly Logits
(B × T × C)
```

---

# Research Contribution

The primary contribution of PhySATFormer is the introduction of **Physics-Guided Channel Attention**, where spacecraft subsystem knowledge is injected directly into the attention mechanism using a fixed physics relationship matrix.

Instead of learning channel relationships entirely from data, attention scores are biased according to known physical subsystem connectivity.

For each attention head:

\[
S_h=\frac{Q_hK_h^{T}}{\sqrt{d_k}}+\lambda_hP
\]

where

- \(Q,K\) are query and key matrices
- \(P\) is the fixed physics relationship matrix
- \(\lambda_h\) is a learnable scalar for attention head \(h\)

---

# Prediction Objective

Given a telemetry window

\[
X \in \mathbb{R}^{T\times C}
\]

the model predicts

\[
\hat{Y}\in\mathbb{R}^{T\times C}
\]

where

\[
\hat{Y}_{t,c}
\]

represents the anomaly logit for telemetry channel \(c\) at timestamp \(t\).

Training uses **BCEWithLogitsLoss** for multi-label channel-wise anomaly localization.

---

# Repository Structure

```
PhySATFormer/

├── configs/
├── data/
├── logs/
├── notebooks/
├── outputs/
├── paper/
├── scripts/
│
├── src/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── utils/
│
├── tests/
│
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# Project Status

| Stage | Status |
|---------|--------|
| Data Loading | ✅ |
| Preprocessing Pipeline | ✅ |
| Baseline Transformer | ✅ |
| Physics-Guided Architecture | ✅ |
| Training Pipeline | 🚧 |
| Explainability | 🚧 |
| Experiments | 🚧 |
| IEEE Paper | 🚧 |

---

# Current Architecture

## Input

```
(B, T, C)
```

where

- **B** = batch size
- **T** = sequence length
- **C** = telemetry channels

---

## Output

```
(B, T, C)
```

Each output value represents the anomaly logit of one telemetry channel at one timestamp.

---

# Dataset

The implementation is designed around the ESA Mission 1 satellite telemetry dataset.

The preprocessing pipeline performs

- telemetry synchronization
- normalization
- interval-to-dense label generation
- sliding-window generation
- PyTorch dataset creation

---

# Technology Stack

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- SHAP
- Ruff

---

# Roadmap

- Physics-Guided Channel Attention
- Temporal Transformer
- Spatio-Temporal Anomaly Localization
- Training Framework
- Explainability (Attention + SHAP)
- Ablation Studies
- IEEE Conference Paper

---

# License

This repository is intended for academic and research purposes.