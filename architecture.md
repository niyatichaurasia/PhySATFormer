

# PhySATFormer Final Training Architecture

```
                           train.py
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Load YAML Configurations│
                  └────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ Set Random Seed        │
                  │ Configure Device       │
                  │ Configure AMP          │
                  └────────────────────────┘
                               │
                               ▼
                    Load Mission Dataset
                               │
                               ▼
                  ┌────────────────────────┐
                  │ TelemetryPipeline      │
                  └────────────────────────┘
                               │
                               ▼
         Train / Validation / Test MissionDatasets
                               │
                               ▼
             Train / Val / Test DataLoaders
                               │
                               ▼
              PhysicsMatrixBuilder
                               │
                               ▼
                    Physics Matrix (76×76)
                               │
                               ▼
                   Build PhySATFormer
                               │
                               ▼
              OptimizerFactory (AdamW)
                               │
                               ▼
            SchedulerFactory (CosineAnnealing)
                               │
                               ▼
             BCEWithLogitsLoss()
                               │
                               ▼
             Metrics (Precision/Recall/F1)
                               │
                               ▼
          CheckpointManager + EarlyStopping
                               │
                               ▼
                    Trainer.fit(...)
                               │
                               ▼
                      Training Loop
```

---

# Responsibilities

## train.py

**Only orchestration.**

It should never contain ML logic.

Responsibilities:

* parse configs
* initialize logging
* initialize seed
* create pipeline
* build datasets
* create dataloaders
* construct physics matrix
* instantiate model
* instantiate optimizer
* instantiate scheduler
* instantiate trainer
* call

```python
trainer.fit()
```

Nothing else.



### Section 1 — Imports & CLI

* imports
* argparse
* logging
* type hints
* constants

### Section 2 — Config loading

* load `dataset.yaml`
* load `model.yaml`
* load `train.yaml`
* validate configs

### Section 3 — Seed & device

* random seeds
* deterministic mode
* CUDA detection
* AMP setup

### Section 4 — Mission loading

* load Mission
* build `TelemetryPipeline`
* create datasets

### Section 5 — DataLoaders

* train
* validation
* test

### Section 6 — Physics matrix

* build adjacency matrix
* move to device

### Section 7 — Model

* instantiate `PhySATFormer`
* send to device

### Section 8 — Optimizer / Scheduler / Loss

* AdamW
* CosineAnnealing
* BCEWithLogitsLoss

### Section 9 — Metrics

* metric objects

### Section 10 — Checkpoint manager

* instantiate
* optional resume

### Section 11 — Trainer

* construct trainer

### Section 12 — Training

* call `trainer.fit()`
* save final checkpoint
* optional test evaluation



---

# TelemetryPipeline

Responsible for

```
Mission

↓

TelemetryAnalyzer

↓

TelemetryAssembler

↓

Normalizer

↓

WindowGenerator

↓

MissionDataset
```

Returns

```
train_dataset

val_dataset

test_dataset
```

No DataLoaders.

---

# PhysicsMatrixBuilder

New dedicated utility.

Input

```
Mission
```

Output

```
physics_matrix

shape

(C,C)
```

Current implementation

```
same subsystem

↓

1

else

↓

0
```

Diagonal

```
1
```

Keeping this separate allows future experiments with weighted or graph-based priors.

---

# DataLoaders

Constructed in train.py

```
train_loader
shuffle=True

validation_loader
shuffle=False

test_loader
shuffle=False
```

Configurable:

```
batch_size

num_workers

pin_memory
```

---

# Model

```
PhySATFormer
```

Outputs

```
(B,T,C)
```

Raw logits.

Never applies sigmoid internally.

---

# Loss

```
BCEWithLogitsLoss
```

Input

```
prediction

(B,T,C)

ground_truth

(B,T,C)
```

---

# Metrics

Computed after sigmoid.

Logged every validation.

```
Precision

Recall

F1
```

Primary metric

```
Validation F1
```

---

# OptimizerFactory

Returns

```
AdamW
```

Configured only from YAML.

---

# SchedulerFactory

Returns

```
CosineAnnealingLR
```

Configured only from YAML.

---

# Trainer

The Trainer owns the entire training lifecycle.

Responsibilities:

```
forward

loss

backward

AMP

gradient clipping

optimizer step

scheduler step

validation

metrics

checkpointing

early stopping

resume training

logging
```

No preprocessing.

No dataset logic.

No model construction.

---

# CheckpointManager

Owns saving and loading.

Checkpoint contents:

```
model_state_dict

optimizer_state_dict

scheduler_state_dict

scaler_state_dict

epoch

best_metric

early_stopping_state

config
```

Files:

```
best_model.pt

last_checkpoint.pt

(optional)

epoch_010.pt
```

---

# EarlyStopping

Monitors

```
val_f1
```

Configurable:

```
monitor

mode

patience

min_delta
```

---

# AMP

Enabled only when

```
mixed_precision == True

AND

device == CUDA
```

Uses

```
torch.autocast()

GradScaler()
```

Automatically disabled on CPU.

---

# Resume Training

```
python train.py --resume checkpoint.pt
```

Restores:

* model
* optimizer
* scheduler
* scaler
* epoch
* best metric
* early stopping

Training resumes seamlessly.

---

# Logging

Training

Every

```
log_every_n_batches
```

Logs:

* epoch
* batch
* loss
* learning rate

Validation

Every

```
validate_every_n_epochs
```

Logs:

* loss
* precision
* recall
* F1

---

# Configuration Layout

## model.yaml

```yaml
input_dim
num_channels
channel_embedding_dim
d_model
num_heads
num_channel_layers
num_temporal_layers
ff_dim
dropout
```

---

## dataset.yaml

```yaml
window_size
stride
normalization_method
train_ratio
validation_ratio
random_seed
direction
max_rows
label_strategy:
  type: interval
```

---

## train.yaml

```yaml
experiment:
  name: physatformer_baseline
  seed: 42
  deterministic: true
  output_dir: outputs

batch_size: 32
epochs: 50

device: auto

mixed_precision: true

gradient_clip: 1.0

gradient_accumulation_steps: 1

num_workers: 4

pin_memory: true

log_every_n_batches: 50

validate_every_n_epochs: 1

checkpoint:
  directory: checkpoints
  monitor: val_f1
  mode: max
  save_last: true
  save_best: true
  save_every_n_epochs: null

optimizer:
  type: adamw
  lr: 1e-4
  weight_decay: 1e-2
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: cosine
  t_max: 50
  eta_min: 1e-6

early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001
  monitor: val_f1
  mode: max

resume: null
```

---

# Directory Structure

```
physatformer/
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── src/
│   ├── core/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   │   ├── trainer.py
│   │   ├── checkpoint_manager.py
│   │   ├── early_stopping.py
│   │   ├── optimizer_factory.py
│   │   ├── scheduler_factory.py
│   │   ├── loss.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│       ├── constants.py
│       ├── physics_matrix_builder.py
│       ├── logger.py
│       └── seed.py
│
├── checkpoints/
├── logs/
├── outputs/
├── scripts/
└── train.py
```

## One architectural recommendation before implementation

I would add a dedicated `physics_matrix_builder.py` under `src/utils/`. Right now the physics matrix is a conceptual part of your model, but its construction is a preprocessing concern rather than a model concern. Isolating it into a builder keeps the model agnostic to how the prior is generated, making it easy to experiment later with binary adjacency, weighted subsystem graphs, or learned graph priors without modifying `PhySATFormer` itself.

With this architecture frozen, `train.py` becomes a thin composition root that wires together well-defined components rather than containing training logic itself. That's a hallmark of a maintainable research framework and fits the modular design you've established throughout the repository.
