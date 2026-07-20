"""
scripts/test_one_epoch.py
==========================

ONE-EPOCH smoke test for PhySATFormer.

This is NOT a training run. It is the final verification step before
real training begins, and it comes after:

    * scripts/test_integration.py       (proves every component *constructs*)
    * scripts/test_train_step.py        (proves one manual train step runs)

This script proves the last remaining thing: that the *unmodified*
``Trainer.fit()`` can execute a complete epoch -- training, scheduler
step, validation, checkpointing, and early-stopping update -- from
end to end, without raising an exception and without producing
non-finite values anywhere in the returned history.

It reuses the *existing* orchestration functions from ``train.py``
directly (``load_configs``, ``setup_logging``, ``set_random_seed``,
``select_device``, ``build_mission``, ``build_pipeline``,
``build_datasets``, ``build_dataloaders``, ``build_physics_matrix``,
``build_model``, ``build_optimizer``, ``build_scheduler``,
``build_criterion``, ``build_metrics``, ``build_trainer``) to build the
full pipeline, exactly as ``train.py`` and the previous smoke tests do.
No pipeline-construction logic is duplicated here.

``Trainer.fit()`` itself is called exactly as ``train.py`` calls it --
with ``num_epochs=1`` -- and is never subclassed, wrapped, or
monkey-patched. Every piece of ``Trainer.fit()`` functionality (train
epoch, scheduler.step(), validate epoch, checkpoint save-on-improve,
early_stopping.update(), history collection, epoch-summary print) runs
exactly as it does during real training.

To keep this a *true* smoke test on large datasets, the train and
validation ``MissionDataset`` objects returned by ``build_datasets()``
are wrapped in ``torch.utils.data.Subset`` (first ``TRAIN_SMOKE_SAMPLES``
/ ``VALIDATION_SMOKE_SAMPLES`` indices, or the full dataset if smaller)
before being handed to the unmodified ``build_dataloaders()``. ``Subset``
implements the ``Dataset`` interface, so nothing downstream -- dataloader
construction, ``Trainer.fit()`` -- can tell the difference except that
there is far less data to iterate over. The test dataset is left
untouched.

What this script deliberately does NOT do:
    * modify ``Trainer``, ``train.py``, or any other project source file
    * monkey-patch any method
    * skip or stub out any stage of ``Trainer.fit()``
    * train for more than one epoch
    * change the *composition* of the pipeline -- only the amount of
      data it sees

Usage:
    python scripts/test_one_epoch.py
    python scripts/test_one_epoch.py --device cpu
    python scripts/test_one_epoch.py --train-config configs/train.yaml \\
        --model-config configs/model.yaml \\
        --dataset-config configs/dataset.yaml

Exit status:
    0  -> the one-epoch fit() call completed and every verification passed
    1  -> a stage raised an exception, or a verification failed
          (full traceback printed to stderr)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

import math

from torch.utils.data import Subset

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the current working
# directory this script is invoked from (e.g. `python scripts/test_one_epoch.py`
# run from the repo root, or from inside `scripts/`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Reuse the real orchestration functions from train.py. Nothing about
# pipeline *construction* is reimplemented -- every step below just calls
# into the existing code, exactly as train.py and the previous smoke
# tests do.
# ---------------------------------------------------------------------------
from train import (  # noqa: E402
    build_criterion,
    build_dataloaders,
    build_datasets,
    build_metrics,
    build_mission,
    build_model,
    build_optimizer,
    build_physics_matrix,
    build_pipeline,
    build_scheduler,
    build_trainer,
    load_configs,
    select_device,
    set_random_seed,
    setup_logging,
)
from src.training.trainer import Trainer  # noqa: E402

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Smoke-test subset sizes.
#
# Trainer.fit() is unmodified and will iterate over whatever dataset it is
# handed. To make this a *true* smoke test (a few minutes, not several
# hours, on CPU), we hand it tiny torch.utils.data.Subset views of the real
# train/validation MissionDatasets instead of the full datasets. Subset
# implements the Dataset interface, so build_dataloaders() (also
# unmodified) works with it exactly as it would with the full dataset.
# ---------------------------------------------------------------------------
TRAIN_SMOKE_SAMPLES = 512
VALIDATION_SMOKE_SAMPLES = 128

# The exact set of history keys Trainer.fit() is documented to return.
_EXPECTED_HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_f1",
    "val_f1",
    "train_precision",
    "val_precision",
    "train_recall",
    "val_recall",
)


class OneEpochTestFailure(Exception):
    """Raised internally when a stage fails, to unwind cleanly to main()."""


# ---------------------------------------------------------------------------
# Uniform stage runner (same reporting style as the previous smoke tests)
# ---------------------------------------------------------------------------


def run_stage(stage_name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a single stage with uniform PASS/FAIL reporting.

    Args:
        stage_name: Human-readable name of the stage, used in PASS/FAIL
            output (e.g. ``"Mission created"``).
        fn: The callable implementing the stage.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns, unchanged.

    Raises:
        OneEpochTestFailure: If ``fn`` raises any exception. The
            original traceback is printed (to stderr) before this is
            raised, so the caller can exit immediately.
    """
    try:
        result = fn(*args, **kwargs)
    except Exception:
        print(f"\n[FAIL] {stage_name}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        traceback.print_exc()
        print("-" * 60, file=sys.stderr)
        raise OneEpochTestFailure(stage_name)

    print(f"[PASS] {stage_name}")
    return result


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _verify_history_is_dict(history: Any) -> None:
    """Verify ``trainer.fit()`` returned a dict.

    Raises:
        AssertionError: If ``history`` is not a ``dict``.
    """
    if not isinstance(history, dict):
        raise AssertionError(
            f"Expected trainer.fit() to return a dict, got {type(history).__name__}."
        )


def _verify_history_keys(history: Dict[str, List[float]]) -> None:
    """Verify every expected history key is present.

    Raises:
        AssertionError: If any key in ``_EXPECTED_HISTORY_KEYS`` is
            missing from ``history``.
    """
    missing = [key for key in _EXPECTED_HISTORY_KEYS if key not in history]
    if missing:
        raise AssertionError(f"History is missing expected key(s): {missing}.")


def _verify_history_lengths(history: Dict[str, List[float]], num_epochs: int) -> None:
    """Verify every history list has exactly ``num_epochs`` entries.

    Args:
        history: The history dict returned by ``trainer.fit()``.
        num_epochs: The number of epochs ``fit()`` was asked to run
            (early stopping cannot trigger within a single epoch, so
            this should equal 1 for this script).

    Raises:
        AssertionError: If any history list's length differs from
            ``num_epochs``.
    """
    bad = {
        key: len(values)
        for key, values in history.items()
        if key in _EXPECTED_HISTORY_KEYS and len(values) != num_epochs
    }
    if bad:
        raise AssertionError(
            f"Expected every history list to have length {num_epochs}, "
            f"got: {bad}."
        )


def _verify_all_finite(history: Dict[str, List[float]]) -> None:
    """Verify every value in every history list is finite (no NaN/inf).

    Raises:
        AssertionError: If any non-finite value is found, naming the
            offending key, index, and value.
    """
    for key in _EXPECTED_HISTORY_KEYS:
        for index, value in enumerate(history[key]):
            if not math.isfinite(value):
                raise AssertionError(
                    f"history['{key}'][{index}] is not finite: {value!r}."
                )


def _verify_loss_values_finite(history: Dict[str, List[float]]) -> None:
    """Verify train/val loss values specifically are finite.

    This is a narrower, explicitly-named re-check of the loss entries
    (a subset already covered by :func:`_verify_all_finite`) so the
    PASS/FAIL output clearly reports loss finiteness as its own stage,
    per the verification checklist.

    Raises:
        AssertionError: If any loss value is non-finite.
    """
    for key in ("train_loss", "val_loss"):
        for index, value in enumerate(history[key]):
            if not math.isfinite(value):
                raise AssertionError(
                    f"history['{key}'][{index}] is not finite: {value!r}."
                )


def _verify_metric_values_finite(history: Dict[str, List[float]]) -> None:
    """Verify precision/recall/F1 values specifically are finite.

    Narrower, explicitly-named re-check of the metric entries (already
    covered by :func:`_verify_all_finite`), reported as its own stage.

    Raises:
        AssertionError: If any precision/recall/F1 value is non-finite.
    """
    metric_keys = (
        "train_precision",
        "val_precision",
        "train_recall",
        "val_recall",
        "train_f1",
        "val_f1",
    )
    for key in metric_keys:
        for index, value in enumerate(history[key]):
            if not math.isfinite(value):
                raise AssertionError(
                    f"history['{key}'][{index}] is not finite: {value!r}."
                )


def _make_smoke_subset(dataset: Any, num_samples: int) -> Subset:
    """Build a small ``Subset`` of ``dataset`` for the smoke test.

    Takes the first ``min(num_samples, len(dataset))`` indices. Using a
    plain leading slice (rather than a random sample) keeps the smoke
    test deterministic without touching ``set_random_seed`` state.

    Args:
        dataset: The full ``MissionDataset`` (or any ``Dataset``) to
            take a subset of.
        num_samples: The desired number of samples in the subset.

    Returns:
        Subset: A ``torch.utils.data.Subset`` wrapping ``dataset``,
            implementing the same ``Dataset`` interface so it can be
            passed anywhere the full dataset would be.
    """
    size = min(num_samples, len(dataset))
    return Subset(dataset, list(range(size)))


def _verify_checkpoint_exists(checkpoint_dir: Path) -> Path:
    """Verify at least one checkpoint file exists in ``checkpoint_dir``.

    ``Trainer.fit()`` saves a checkpoint only when validation F1
    improves over the running best (``Trainer._maybe_save_checkpoint``).
    Since this is the very first epoch, the initial validation F1 is
    always an improvement over ``-inf``, so a checkpoint is expected to
    have been written unconditionally.

    Args:
        checkpoint_dir: The configured checkpoint directory
            (``train_cfg["checkpoint_dir"]``).

    Returns:
        Path: The path of the checkpoint file found.

    Raises:
        AssertionError: If ``checkpoint_dir`` does not exist or
            contains no files.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise AssertionError(
            f"Checkpoint directory does not exist: '{checkpoint_dir}'."
        )

    checkpoint_files = sorted(
        p for p in checkpoint_dir.glob("**/*") if p.is_file()
    )
    if not checkpoint_files:
        raise AssertionError(
            f"No checkpoint files found in '{checkpoint_dir}' after "
            "the first epoch's fit() call."
        )

    return checkpoint_files[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the one-epoch smoke test.

    Returns:
        argparse.Namespace: Parsed arguments containing paths to the
            three configuration files and an optional device override.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-epoch PhySATFormer smoke test: a complete, "
            "unmodified Trainer.fit(num_epochs=1) call, verified end "
            "to end."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "train.yaml",
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "model.yaml",
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "dataset.yaml",
        help="Path to the dataset configuration YAML file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help=(
            "Device override. If not provided, the device specified in "
            "train.yaml is used."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build the pipeline (reusing train.py) and run exactly one epoch
    via the unmodified ``Trainer.fit()``, verifying the result end to end.
    """
    args = parse_arguments()

    print("=" * 60)
    print("PhySATFormer -- One-Epoch Smoke Test")
    print("=" * 60)

    try:
        # ------------------------------------------------------------
        # 1. Load configs
        # ------------------------------------------------------------
        train_cfg, model_cfg, dataset_cfg = run_stage(
            "Configs loaded",
            load_configs,
            train_config=args.train_config,
            model_config=args.model_config,
            dataset_config=args.dataset_config,
        )

        # ------------------------------------------------------------
        # 2. Initialize logger
        # ------------------------------------------------------------
        logger: logging.Logger = run_stage("Logger initialized", setup_logging)

        # ------------------------------------------------------------
        # 3. Set random seed (determinism)
        # ------------------------------------------------------------
        experiment_cfg = train_cfg.get("experiment", {}) or {}
        seed = experiment_cfg.get("seed", train_cfg.get("random_seed", 42))
        deterministic = experiment_cfg.get("deterministic", True)
        run_stage(
            "Random seed set",
            set_random_seed,
            seed=seed,
            deterministic=deterministic,
        )

        # ------------------------------------------------------------
        # 4. Select device
        # ------------------------------------------------------------
        device = run_stage(
            "Device selected",
            select_device,
            configured_device=train_cfg.get("device", "auto"),
            override_device=args.device,
        )
        logger.info("Resolved device: %s", device)

        # ------------------------------------------------------------
        # 5. Construct Mission + TelemetryPipeline
        # ------------------------------------------------------------
        mission = run_stage("Mission created", build_mission, dataset_cfg, logger)
        pipeline = run_stage(
            "TelemetryPipeline created", build_pipeline, dataset_cfg, logger
        )

        # ------------------------------------------------------------
        # 6. Construct datasets + dataloaders
        # ------------------------------------------------------------
        train_dataset, validation_dataset, test_dataset = run_stage(
            "Datasets created (train/validation/test)",
            build_datasets,
            pipeline=pipeline,
            mission=mission,
            dataset_cfg=dataset_cfg,
            logger=logger,
        )
        # ------------------------------------------------------------
        # 6b. Shrink train/validation datasets to a tiny smoke-test
        #     subset. Trainer, train.py, the model, the datasets
        #     themselves, and build_dataloaders() are all untouched --
        #     Subset just presents a bounded view of the same
        #     underlying MissionDataset, so everything downstream
        #     (DataLoader construction, Trainer.fit()) runs exactly as
        #     it would against the full dataset, only faster.
        # ------------------------------------------------------------
        train_dataset = run_stage(
            f"Train dataset reduced to smoke subset "
            f"(<= {TRAIN_SMOKE_SAMPLES} samples)",
            _make_smoke_subset,
            train_dataset,
            TRAIN_SMOKE_SAMPLES,
        )
        validation_dataset = run_stage(
            f"Validation dataset reduced to smoke subset "
            f"(<= {VALIDATION_SMOKE_SAMPLES} samples)",
            _make_smoke_subset,
            validation_dataset,
            VALIDATION_SMOKE_SAMPLES,
        )
        logger.info(
            "Smoke subset sizes: train=%d validation=%d (test unchanged=%d)",
            len(train_dataset),
            len(validation_dataset),
            len(test_dataset),
        )

        train_loader, validation_loader, test_loader = run_stage(
            "DataLoaders created (train/validation/test)",
            build_dataloaders,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            train_cfg=train_cfg,
            logger=logger,
        )

        # ------------------------------------------------------------
        # 7. Construct physics matrix
        # ------------------------------------------------------------
        physics_matrix = run_stage(
            "PhysicsRelationshipMatrix created",
            build_physics_matrix,
            mission,
            logger,
        )
        physics_matrix = physics_matrix.to(device)

        # ------------------------------------------------------------
        # 8. Construct model
        # ------------------------------------------------------------
        model = run_stage(
            "PhySATFormer created",
            build_model,
            model_cfg=model_cfg,
            physics_matrix=physics_matrix,
            device=device,
            logger=logger,
        )

        # ------------------------------------------------------------
        # 9. Construct optimizer / scheduler / loss / metrics
        # ------------------------------------------------------------
        optimizer = run_stage(
            "Optimizer created", build_optimizer, model, train_cfg, logger
        )
        scheduler = run_stage(
            "Scheduler created", build_scheduler, optimizer, train_cfg, logger
        )
        criterion = run_stage("Loss created", build_criterion, logger)
        metrics = run_stage("Metrics created", build_metrics, logger)

        # ------------------------------------------------------------
        # 10. Construct Trainer (unmodified, production class)
        # ------------------------------------------------------------
        trainer: Trainer = run_stage(
            "Trainer created",
            build_trainer,
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            train_cfg=train_cfg,
            device=device,
            logger=logger,
        )

        if "checkpoint_dir" not in train_cfg or train_cfg["checkpoint_dir"] is None:
            raise OneEpochTestFailure(
                "Missing required training configuration key: 'checkpoint_dir'."
            )
        checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        logger.info(
            "gradient_clip_value=%s checkpoint_dir=%s",
            trainer.gradient_clip_value,
            checkpoint_dir,
        )

        # ------------------------------------------------------------
        # 11. Run exactly ONE epoch through the unmodified Trainer.fit()
        # ------------------------------------------------------------
        history = run_stage(
            "One epoch completed (Trainer.fit(num_epochs=1))",
            trainer.fit,
            train_loader=train_loader,
            validation_loader=validation_loader,
            num_epochs=1,
        )

        # Trainer.fit() runs validate_epoch() internally as part of the
        # single epoch above; this stage exists only to explicitly
        # surface "validation completed" as its own PASS/FAIL line, per
        # the verification checklist. No extra validation pass is run.
        run_stage(
            "Validation completed",
            lambda: None if "val_loss" in history and len(history["val_loss"]) == 1
            else (_ for _ in ()).throw(
                AssertionError("Validation did not run during fit().")
            ),
        )

        # ------------------------------------------------------------
        # 12. Verify history
        # ------------------------------------------------------------
        run_stage("History is a dict", _verify_history_is_dict, history)
        run_stage("History keys verified", _verify_history_keys, history)
        run_stage(
            "History list lengths == 1",
            _verify_history_lengths,
            history,
            1,
        )
        run_stage(
            "Loss values finite (train_loss, val_loss)",
            _verify_loss_values_finite,
            history,
        )
        run_stage(
            "Precision/Recall/F1 values finite",
            _verify_metric_values_finite,
            history,
        )
        run_stage(
            "No NaN or Inf values anywhere in history",
            _verify_all_finite,
            history,
        )
        logger.info(
            "Epoch 1 summary: train_loss=%.4f val_loss=%.4f "
            "train_f1=%.4f val_f1=%.4f train_precision=%.4f "
            "val_precision=%.4f train_recall=%.4f val_recall=%.4f",
            history["train_loss"][0],
            history["val_loss"][0],
            history["train_f1"][0],
            history["val_f1"][0],
            history["train_precision"][0],
            history["val_precision"][0],
            history["train_recall"][0],
            history["val_recall"][0],
        )

        # ------------------------------------------------------------
        # 13. Verify checkpoint was written
        # ------------------------------------------------------------
        checkpoint_path = run_stage(
            "Checkpoint verified",
            _verify_checkpoint_exists,
            checkpoint_dir,
        )
        logger.info("Checkpoint found: %s", checkpoint_path)

    except OneEpochTestFailure as failure:
        print(f"\nOne-epoch smoke test aborted at stage: {failure}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ONE-EPOCH SMOKE TEST PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()