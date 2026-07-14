# PhySATFormer

<p align="center">
  <b>Physics-Guided Spatio-Temporal Transformer for Satellite Telemetry Anomaly Detection</b>
</p>

<p align="center">

Research project implementing a novel Transformer architecture that incorporates spacecraft engineering knowledge into the attention mechanism for interpretable and physics-informed anomaly detection.

</p>

---

## Overview

Satellite telemetry consists of hundreds of interdependent sensor measurements collected over time. Conventional Transformer architectures learn relationships between telemetry channels entirely from data, ignoring prior engineering knowledge already available from spacecraft design.

**PhySATFormer** introduces a **Physics-Guided Channel Attention** mechanism that incorporates subsystem relationships directly into the attention computation before temporal sequence modeling.

The proposed architecture separates learning into two complementary stages:

- **Channel-level reasoning** using a physics-guided attention mechanism
- **Temporal reasoning** using a Transformer encoder

This design allows the model to leverage both **domain knowledge** and **long-range temporal dependencies** for anomaly detection.

---

# Overall Architecture

<p align="center">

*(Architecture figure will be added here)*

</p>

```text
Raw Telemetry
      │
      ▼
Telemetry Channel Encoder
      │
      ▼
Physics-Guided Channel Attention
      │
      ▼
Channel Attention Block
      │
      ▼
Channel Pool
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Encoder Stack
      │
      ▼
Mean Pooling
      │
      ▼
Classification Head
```

---

# Core Contributions

- Physics-guided channel attention using subsystem relationship priors
- Factorized spatio-temporal Transformer architecture
- Modular Transformer implementation built from scratch in PyTorch
- Memory-efficient preprocessing pipeline for large-scale satellite telemetry
- Fully reproducible training and evaluation workflow
- Explainability through attention visualization and SHAP analysis *(in progress)*

---

# Mathematical Formulation

The mathematical derivation of the proposed Physics-Guided Attention mechanism is summarized below.

<p align="center">

**(Equation images will be inserted here)**

</p>

Recommended equations:

- Channel Embedding
- Query, Key and Value Projection
- Standard Scaled Dot-Product Attention
- Physics Relationship Matrix
- Physics-Guided Attention
- Attention Weight Computation
- Context Vector
- Mean Pooling
- Classification Layer

The complete derivation will also be included in the accompanying IEEE paper.

---

# Repository Structure

```text
physatformer/
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── checkpoints/
├── logs/
├── outputs/
├── notebooks/
├── paper/
│
├── scripts/
│   ├── run_all_tests.py
│   └── test_day*.py
│
├── src/
│   ├── core/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
├── README.md
└── pyproject.toml
```

---

# Features

### Dataset Pipeline

- ESA Mission abstraction
- Lazy metadata loading
- Multi-channel telemetry assembly
- Telemetry inspection and analysis

### Preprocessing

- Sliding window generation
- Chronological train / validation / test split
- Memory-efficient normalization
- PyTorch dataset generation

### Models

- Baseline Transformer
- Physics-Guided Channel Attention
- Channel Attention Block
- Physics Relationship Matrix
- PhySATFormer

### Engineering

- Modular architecture
- End-to-end integration tests
- YAML-based configuration
- Reproducible preprocessing pipeline

---

# Dataset

Current experiments use the **ESA Mission 1 Telemetry Dataset**.

Expected directory structure:

```text
Mission1/
│
├── channels.csv
├── labels.csv
├── anomaly_types.csv
├── telecommands.csv
└── telemetry/
```

Dataset locations are configured through the YAML configuration files.

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/physatformer.git

cd physatformer
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```powershell
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running Integration Tests

Run the complete integration suite

```bash
python -m scripts.run_all_tests
```

---

# Project Status

| Component | Status |
|------------|:------:|
| Dataset Pipeline | ✅ |
| Preprocessing Pipeline | ✅ |
| Baseline Transformer | ✅ |
| Physics-Guided Architecture | ✅ |
| Training Pipeline | 🚧 |
| Evaluation | 🚧 |
| Explainability | 🚧 |
| IEEE Paper | 🚧 |

---

# Roadmap

- Train Baseline Transformer
- Train PhySATFormer
- Quantitative evaluation
- Attention visualization
- SHAP explainability
- Ablation studies
- IEEE conference submission

---

# Citation

If you use this work in your research, please cite the forthcoming publication.

```bibtex
@article{physatformer2026,
  title   = {PhySATFormer: Physics-Guided Spatio-Temporal Transformer for Satellite Telemetry Anomaly Detection},
  author  = {Niyati Chaurasia},
  journal = {Under Preparation},
  year    = {2026}
}
```

---

# License

This project is released under the MIT License.

---

# Author

**Niyati Chaurasia**

B.Tech Computer Science (Artificial Intelligence & Machine Learning)

Manipal University Jaipur

Research Interests

- Physics-Informed AI
- Deep Learning
- Transformer Architectures
- Explainable AI
- Satellite Telemetry
- Time-Series Modeling