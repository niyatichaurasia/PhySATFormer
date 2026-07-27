# PhySATFormer

> **Physics-Guided Spatio-Temporal Transformer for Channel-Level Telemetry Anomaly Localization**

PhySATFormer is a research framework for **spacecraft telemetry anomaly localization** using a **Physics-Guided Transformer**.

Unlike conventional anomaly detectors that only determine whether a telemetry window is anomalous, PhySATFormer performs **fine-grained spatio-temporal localization**, identifying **which telemetry channels become anomalous and when**.

The core contribution is the incorporation of **physics-based subsystem relationships** into the Transformer attention mechanism, allowing the model to exploit prior engineering knowledge during representation learning.

---

## Research Objective

Given multivariate spacecraft telemetry,

```
Telemetry
(batch, sequence_length, channels)
```

predict

```
Anomaly Labels
(batch, sequence_length, channels)
```

where each prediction answers:

> **"Is this telemetry channel anomalous at this timestep?"**

This transforms anomaly detection into a **channel-level spatio-temporal localization** problem rather than a simple binary classification task.

---

# Architecture

```
Telemetry
(B,T,C)

        │
        ▼

Telemetry Channel Encoder

        │
        ▼

Physics-Guided Channel Attention

        │
        ▼

Channel Attention Blocks

        │
        ▼

Channel Pooling

        │
        ▼

Temporal Transformer Encoder

        │
        ▼

Prediction Head

        │
        ▼

Channel-Level Anomaly Logits
(B,T,C)
```

---

# Key Features

- Physics-guided channel attention
- Transformer-based temporal modeling
- Channel-level anomaly localization
- Spatio-temporal predictions
- Modular preprocessing pipeline
- Reproducible experiment configuration
- Training framework with checkpointing and early stopping
- Explainability-ready architecture

---

# Repository Structure

```
physatformer/

├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── src/
│   ├── core/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── explainability/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
├── checkpoints/
├── logs/
├── outputs/
├── paper/
│
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# Core Components

## Preprocessing

- Mission abstraction
- Metadata parsing
- Telemetry assembly
- Normalization
- Window generation
- Interval label generation
- Dataset construction

---

## Model

The model consists of two stages.

### Physics-Guided Channel Modeling

Learns relationships between telemetry channels while incorporating prior subsystem knowledge through a Physics Relationship Matrix.

### Temporal Modeling

Models long-range temporal dependencies using Transformer encoder blocks after channel interactions have been aggregated.

---

# Physics Prior

A Physics Relationship Matrix is constructed from spacecraft subsystem metadata.

Instead of allowing every telemetry channel to attend equally, the model biases attention toward physically related channels.

This embeds engineering knowledge directly into the attention mechanism while preserving end-to-end learning.

---

# Model Output

PhySATFormer predicts

```
(batch_size,
 sequence_length,
 num_channels)
```

Each element is an independent anomaly prediction for one telemetry channel at one timestep.

The model outputs **raw logits**, trained using **BCEWithLogitsLoss**.

---

# Training Pipeline

The training framework includes

- AdamW optimizer
- Cosine Annealing learning rate scheduler
- Gradient clipping
- Early stopping
- Model checkpointing
- Automatic metric tracking

---

# Evaluation

Performance is evaluated using

- Precision
- Recall
- F1-score
- Channel-wise Precision
- Channel-wise Recall
- Channel-wise F1

The objective is not only to detect anomalies but also to accurately localize them across telemetry channels and time.

---

# Explainability

The framework is designed to support post-hoc explainability through

- Attention visualization
- Channel importance analysis
- Physics-guided attention inspection
- SHAP-based explanations
- Temporal attribution analysis

---

# Configuration

Experiments are configured using YAML files.

```
configs/

dataset.yaml

model.yaml

train.yaml
```

This enables reproducible experiments and straightforward hyperparameter tuning.

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/physatformer.git
cd physatformer
```

Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

---

# Running Training

```bash
python train.py
```

---

# Project Status

Current implementation includes

- Physics-guided Transformer architecture
- Complete telemetry preprocessing pipeline
- Modular training framework
- Configuration-based experiment management

Upcoming work includes

- End-to-end training
- Explainability module
- Benchmark experiments
- Ablation studies
- IEEE paper figures and evaluation

---

# Research Contribution

PhySATFormer introduces a **physics-guided attention mechanism** for multivariate spacecraft telemetry, combining prior subsystem knowledge with Transformer-based sequence modeling to achieve **channel-level spatio-temporal anomaly localization**.

---

# License

This project is released under the MIT License.

---

# Citation

If you use this repository in your research, please cite the accompanying paper (to be released).