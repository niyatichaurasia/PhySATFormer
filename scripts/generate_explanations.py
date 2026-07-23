"""
scripts/generate_explanations.py

One-command explainability entry point:

    python scripts/generate_explanations.py

Generates SHAP attributions, attention heatmaps, and (for PhySATFormer
only) physics-prior-vs-learned-attention comparisons, reusing the same
config-loading / model-construction / checkpoint-loading path as
`scripts/evaluate_model.py`.

See the INTEGRATION NOTE at the top of `scripts/evaluate_model.py` for
the three assumptions (config loading, Mission construction,
PhysicsRelationshipMatrix construction) that apply here too -- this
script reuses that same setup code.

If your model_type is "baseline" (BaselineTransformer), physics
comparison is automatically skipped, since BaselineTransformer has no
physics prior.
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

from src.core.mission import Mission  # ASSUMPTION: see evaluate_model.py note
from src.evaluation.evaluator import Evaluator
from src.explainability.attention_visualizer import (
    capture_attention,
    save_attention_heatmaps,
)
from src.explainability.physics_visualizer import generate_physics_comparison
from src.explainability.shap_explainer import compute_and_save_shap
from src.models.baseline_transformer import BaselineTransformer
from src.models.physatformer import PhySATFormer
from src.models.physics_matrix import (  # ASSUMPTION: see evaluate_model.py note
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


def _build_model(model_config: dict, physics_matrix) -> torch.nn.Module:
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


def _find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> str:
    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under '{checkpoint_dir}'. Pass "
            "--checkpoint explicitly."
        )
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP, attention, and physics explanations.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-shap-background", type=int, default=50)
    parser.add_argument("--num-shap-samples", type=int, default=20)
    args = parser.parse_args()

    ensure_results_directories(RESULTS_DIR)

    configs = _load_configs()
    dataset_config, model_config, train_config = (
        configs["dataset"],
        configs["model"],
        configs["train"],
    )

    device = _resolve_device(train_config.get("device", "auto"))
    logger.info("Using device: %s", device)

    mission = Mission.from_config(dataset_config)  # ASSUMPTION

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
    )

    model_type = model_config.get("model_type", "physatformer")
    physics_matrix = None
    if model_type == "physatformer":
        physics_matrix = PhysicsRelationshipMatrix(mission).build().to(device)  # ASSUMPTION

    model = _build_model(model_config, physics_matrix)

    checkpoint_path = args.checkpoint or _find_latest_checkpoint(
        pathlib.Path(train_config["checkpoint_dir"])
    )
    evaluator = Evaluator(model=model, device=device)
    evaluator.load_checkpoint(checkpoint_path)
    model = evaluator.model
    model.eval()

    # ---------------- SHAP ----------------
    logger.info("Generating SHAP explanations...")
    compute_and_save_shap(
        model=model,
        test_loader=test_loader,
        output_dir=RESULTS_DIR / "shap",
        num_background=args.num_shap_background,
        num_test_windows=args.num_shap_samples,
        device=device,
    )

    # ---------------- Attention maps ----------------
    logger.info("Generating attention heatmaps...")
    sample_inputs, _ = next(iter(test_loader))
    sample_inputs = sample_inputs.to(device)
    attention_weights = capture_attention(model, sample_inputs)
    save_attention_heatmaps(
        attention_weights,
        output_dir=RESULTS_DIR / "attention_maps",
        average_heads=True,
    )

    # ---------------- Physics comparison (PhySATFormer only) ----------------
    if physics_matrix is not None and attention_weights:
        logger.info("Generating physics-vs-learned-attention comparison...")
        # Average across every captured channel-attention layer/head/batch
        # entry into a single (C, C) learned-attention summary.
        all_layer_matrices = []
        for weights in attention_weights.values():
            batch_avg = weights[0]  # first sample in batch
            if batch_avg.dim() == 3:  # (num_heads, C, C)
                batch_avg = batch_avg.mean(dim=0)
            all_layer_matrices.append(batch_avg)
        learned_attention = torch.stack(all_layer_matrices).mean(dim=0).numpy()

        generate_physics_comparison(
            physics_matrix=physics_matrix,
            learned_attention=learned_attention,
            output_dir=RESULTS_DIR / "attention_maps",
        )
    elif physics_matrix is not None:
        logger.warning(
            "model_type is 'physatformer' but no attention weights were "
            "captured; skipping physics comparison. See the integration "
            "note in attention_visualizer.py."
        )
    else:
        logger.info("model_type is 'baseline'; skipping physics comparison.")

    logger.info("Explanation generation complete. Results under '%s'.", RESULTS_DIR)


if __name__ == "__main__":
    main()
