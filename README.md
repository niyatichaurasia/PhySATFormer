# PhySATFormer

> **Physics-Guided Transformer for Channel-Level Spacecraft Telemetry Anomaly Localization**

PhySATFormer is a modular PyTorch-based research framework for **channel-level anomaly localization** in multivariate spacecraft telemetry.

The project combines a configurable preprocessing pipeline, Transformer-based sequence modeling, and physics-guided attention mechanisms to investigate anomaly localization across telemetry channels and time.

---

# Features

- Modular preprocessing pipeline
- Transformer-based sequence modeling
- Physics-guided attention mechanisms
- YAML-based configuration system
- Training and evaluation pipelines
- Model checkpointing
- Explainability modules
- Research-oriented repository structure

---

# Repository Structure

```text
physatformer/

├── configs/
├── data/
├── docs/
├── outputs/
├── paper/
├── scripts/
├── src/
│   ├── core/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── visualization/
│
├── pyproject.toml
├── uv.lock
├── README.md
└── requirements.txt
```

---

# Prerequisites

- Python **3.12**
- Git
- **uv** package manager

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd physatformer
```

---

## 2. Install uv

If you do not already have **uv** installed:

```bash
pip install uv
```

Alternatively, install it using the official Astral installation instructions:

https://docs.astral.sh/uv/

---

## 3. Create a virtual environment

```bash
uv venv
```

---

## 4. Activate the virtual environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 5. Install project dependencies

```bash
uv sync
```

This installs all project dependencies specified in `pyproject.toml` and locked in `uv.lock`.

---

## 6. Verify the installation

Check your Python version:

```bash
python --version
```

Expected output:

```text
Python 3.12.x
```

Verify PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

---

# Configuration

Project configuration files are located in:

```text
configs/
```

These configuration files define:

- Dataset settings
- Model hyperparameters
- Training parameters

Modify the appropriate configuration file before running experiments.

---

# Repository Components

## `src/preprocessing`

Contains the preprocessing pipeline responsible for preparing raw telemetry for model training.

---

## `src/models`

Contains the model implementations, including the Transformer architecture and physics-guided components.

---

## `src/training`

Contains the training pipeline, optimization logic, checkpoint handling, and training utilities.

---

## `src/evaluation`

Contains evaluation utilities and performance metrics.

---

## `src/explainability`

Contains explainability and post-hoc analysis utilities.

---

## `scripts`

Contains executable scripts used throughout the project.

---

# Development

Install development dependencies:

```bash
uv sync --dev
```

After making changes, run the project's configured quality checks before committing.

---

# Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run all project checks.
5. Submit a pull request.

---

# License

Add the appropriate project license here.

---

# Citation

If you use this repository in academic research, please cite the associated publication once it becomes available.