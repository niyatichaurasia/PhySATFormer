"""
scripts/evaluate_model.py

One-command evaluation entry point:

    python scripts/evaluate_model.py

Mirrors train.py's own composition-root style (architecture.md,
"train.py becomes a thin composition root"): this script ONLY wires
together already-existing components (config loading, TelemetryPipeline,
PhysicsMatrixBuilder, model construction, CheckpointManager, Evaluator)
-- it contains no preprocessing, model, or metric logic of its own.

============================================================================
INTEGRATION NOTE -- please read before running
============================================================================
This script was written against the documented contracts in
architecture.md and the uploaded source files. Three pieces of your
actual codebase were NOT included in what was shared with me, so the
calls below are my best-faith reconstruction and are marked inline with
"# ASSUMPTION:" -- please diff them against your real implementations
before running:

  1. Config loading: your `train.py` loads `dataset.yaml`, `model.yaml`,
     `train.yaml` via some existing loader (Section 2 of train.py per
     architecture.md). I've assumed a plain `yaml.safe_load(...)` here;
     swap in your project's actual config-loading utility if it differs
     (e.g. if it does env-var interpolation, validation, or merges CLI
     overrides).
  2. `Mission` construction: architecture.md shows `TelemetryPipeline
     .build(mission, channel_ids, max_rows)` taking a `Mission` object,
     but I don't have `src/core/mission.py`. I've assumed
     `Mission.from_config(...)` / a `mission_path` config key -- adjust
     to however your `train.py` actually constructs its `Mission`.
  3. `PhysicsMatrixBuilder` / `PhysicsRelationshipMatrix`: the class name
     differs between architecture.md (`PhysicsMatrixBuilder`) and your
     prompt (`PhysicsRelationshipMatrix`). I've assumed the latter,
     constructed from `mission` and producing a `(C, C)` tensor -- adjust
     the import/constructor to match whichever one your `train.py`
     actually calls.

Everything else (Evaluator, checkpoint loading, figure/CSV generation)
uses only the modules this repository's evaluation package defines, and
should work unmodified.
============================================================================
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.core.mission import Mission  # ASSUMPTION: see note above
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import save_all_figures
from src.evaluation.prediction_export import export_predictions_csv
from src.models.baseline_transformer import BaselineTransformer
from src.models.physatformer import PhySATFormer
from src.models.physics_matrix import (  # ASSUMPTION: see note above
    PhysicsRelationshipMatrix,
)
from src.preprocessing.pipeline import TelemetryPipeline
from src.utils.results_dirs import ensure_results_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_DIR = pathlib.Path("configs")
RESULTS_DIR = pathlib.Path("results")


def _load_configs() -> dict:
    """Load dataset.yaml, model.yaml, train.yaml into one merged dict."""
    configs = {}
    for name in ("dataset", "model", "train"):
        config_path = CONFIG_DIR / f"{name}.yaml"
        with config_path.open("r", encoding="utf-8") as config_file:
            configs[name] = yaml.safe_load(config_file)
    return configs


def _resolve_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _build_model(model_config: dict, physics_matrix: torch.Tensor) -> torch.nn.Module:
    """Instantiate PhySATFormer or BaselineTransformer from model.yaml."""
    model_type = model_config.get("model_type", "physatformer")

    shared_kwargs = dict(
        input_dim=model_config["input_dim"],
        channel_embedding_dim=model_config["channel_embedding_dim"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_channel_layers=model_config["num_channel_layers"],
        num_temporal_layers=model_config["num_temporal_layers"],
        ff_dim=model_config["ff_dim"],
        num_channels=model_config["num_channels"],
        dropout=model_config.get("dropout", 0.1),
    )

    if model_type == "physatformer":
        return PhySATFormer(physics_matrix=physics_matrix, **shared_kwargs)
    if model_type == "baseline":
        return BaselineTransformer(**shared_kwargs)
    raise ValueError(f"Unknown model_type '{model_type}' in model.yaml.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PhySATFormer checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the checkpoint to evaluate. Defaults to the best "
            "checkpoint under train.yaml's checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid decision threshold for binarizing predictions.",
    )
    args = parser.parse_args()

    ensure_results_directories(RESULTS_DIR)

    configs = _load_configs()
    dataset_config = configs["dataset"]
    model_config = configs["model"]
    train_config = configs["train"]

    device = _resolve_device(train_config.get("device", "auto"))
    logger.info("Using device: %s", device)

    # ASSUMPTION: adjust to however your train.py actually builds a
    # Mission (e.g. a fixed mission id, a path in dataset.yaml, a CLI
    # flag). Kept as a config key here for symmetry with train.py.
    mission = Mission.from_config(dataset_config)

    pipeline = TelemetryPipeline(
        window_size=dataset_config["window_size"],
        stride=dataset_config["stride"],
        normalization_method=dataset_config["normalization_method"],
        train_ratio=dataset_config["train_ratio"],
        validation_ratio=dataset_config["validation_ratio"],
        random_seed=dataset_config["random_seed"],
        direction=dataset_config["direction"],
    )
    _, _, test_dataset = pipeline.build(
        mission=mission,
        channel_ids=dataset_config.get("channel_ids", []),
        max_rows=dataset_config.get("max_rows"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=train_config.get("pin_memory", False),
    )

    physics_matrix = PhysicsRelationshipMatrix(mission).build().to(device)

    model = _build_model(model_config, physics_matrix)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_dir = pathlib.Path(train_config["checkpoint_dir"])
        candidates = sorted(checkpoint_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found under '{checkpoint_dir}'. Pass "
                "--checkpoint explicitly."
            )
        # Best checkpoint = highest saved metric encoded by CheckpointManager
        # is not derivable from the filename alone, so default to the most
        # recently modified checkpoint unless told otherwise.
        checkpoint_path = str(max(candidates, key=lambda path: path.stat().st_mtime))
        logger.info("No --checkpoint given; defaulting to '%s'.", checkpoint_path)

    evaluator = Evaluator(model=model, device=device, threshold=args.threshold)
    evaluator.load_checkpoint(checkpoint_path)

    result = evaluator.evaluate(test_loader)

    export_predictions_csv(result, RESULTS_DIR / "predictions" / "predictions.csv")
    save_all_figures(
        ground_truth=result.ground_truth,
        predictions=result.predictions,
        probabilities=result.probabilities,
        figures_dir=RESULTS_DIR / "figures",
    )

    try:
        from src.evaluation.plots import generate_history_plots

        generate_history_plots(
            history_path=RESULTS_DIR / "history" / "history.json",
            figures_dir=RESULTS_DIR / "figures",
        )
    except FileNotFoundError:
        logger.warning(
            "No results/history/history.json found; skipping training-"
            "history curves. Run training with history saving enabled first."
        )

    logger.info("Evaluation complete. Results written under '%s'.", RESULTS_DIR)


if __name__ == "__main__":
    main()
