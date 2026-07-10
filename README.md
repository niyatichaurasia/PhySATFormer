# PhySATFormer

**Physics-Guided Transformer for Satellite Telemetry Anomaly Detection**

PhySATFormer is a research-oriented framework for multivariate satellite telemetry anomaly detection using Transformer architectures enhanced with physics-guided attention mechanisms.

The project is designed to provide a modular, reproducible pipeline for loading, preprocessing, modeling, and explaining satellite telemetry data.

---

## Features

- Lazy loading of ESA Mission1 telemetry channels
- Modular data ingestion pipeline
- Multivariate telemetry synchronization
- Sliding-window dataset generation
- Channel-wise normalization
- PyTorch dataset generation
- Baseline Transformer (coming soon)
- Physics-Guided Transformer (coming soon)
- Explainable AI using SHAP and Attention Visualization (planned)
- Ablation study support (planned)

---

## Repository Structure

```
PhySATFormer/
│
├── configs/
├── docs/
├── scripts/
├── src/
│   ├── core/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   ├── visualization/
│   └── utils/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Current Progress

### Day 1 – Data Layer ✅

- Mission
- Channel
- MetadataParser
- ChannelLoader
- ChannelInspector

### Day 2 – Preprocessing ✅

- TelemetryAnalyzer
- TelemetryAssembler
- WindowGenerator
- MissionDataset
- TelemetryNormalizer
- TelemetryPipeline

### Upcoming

- Baseline Transformer
- Physics-Guided Transformer
- Training and Evaluation
- Explainability
- Ablation Study

---

## Dataset

This project uses the ESA Mission1 telemetry dataset.

The dataset is **not included** in this repository because of its size.

Configure the dataset location using a `.env` file.

Example:

```env
PHYSATFORMER_DATASET=C:\Users\YourName\Datasets\Mission1
```

The default fallback location is:

```
data/raw/Mission1
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/niyatichaurasia/PhySATFormer.git
cd PhySATFormer
```

Create a virtual environment and install dependencies:

```bash
uv sync
```

or

```bash
pip install -r requirements.txt
```

---

## Running Tests

Day 1:

```bash
python -m scripts.test_day1
```

Day 2:

```bash
python -m scripts.test_day2
```

Run all:

```bash
python -m scripts.run_all_tests
```

---

## Development Philosophy

- Single Responsibility Principle
- Modular architecture
- Research-quality implementation
- Reproducible experiments
- Temporal leakage prevention
- No data leakage during preprocessing

---

## Research Roadmap

- Baseline Transformer
- Physics-Guided Transformer
- Training Pipeline
- Explainability (SHAP + Attention)
- Ablation Study
- IEEE Paper

---

## License

This project is licensed under the MIT License.