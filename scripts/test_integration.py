"""
scripts/test_integration.py
============================

End-to-end integration smoke test for PhySATFormer.

This is NOT a unit test. It does not assert on numerical correctness
or exercise individual functions in isolation. Its only job is to
prove that every major component of the training stack -- config
loading, seeding, device/AMP setup, mission/pipeline construction,
datasets, dataloaders, the physics matrix, the model, the optimizer,
the scheduler, the loss, the metrics, and the trainer -- can be
constructed together, in the same order train.py uses, without
raising an exception.

It reuses the *existing* orchestration functions from ``train.py``
directly (``load_configs``, ``setup_logging``, ``set_random_seed``,
``select_device``, ``setup_amp``, ``build_mission``, ``build_pipeline``,
``build_datasets``, ``build_dataloaders``, ``build_physics_matrix``,
``build_model``, ``build_optimizer``, ``build_scheduler``,
``build_criterion``, ``build_metrics``, ``build_trainer``). No ML or
orchestration logic is duplicated here.

What this script deliberately does NOT do:
    * call ``trainer.fit()``
    * run a forward pass
    * compute a loss value
    * save or load checkpoints
    * write anything to disk
    * modify any existing project file

Usage:
    python scripts/test_integration.py
    python scripts/test_integration.py --device cpu
    python scripts/test_integration.py --train-config configs/train.yaml \\
        --model-config configs/model.yaml \\
        --dataset-config configs/dataset.yaml

Exit status:
    0  -> every stage constructed successfully
    1  -> a stage raised an exception (traceback is printed)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the current working
# directory this script is invoked from (e.g. `python scripts/test_integration.py`
# run from the repo root, or from inside `scripts/`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Reuse the real orchestration functions from train.py. Nothing below is
# reimplemented -- every step just calls into the existing code.
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
    setup_amp,
    setup_logging,
)

T = TypeVar("T")


class IntegrationTestFailure(Exception):
    """Raised internally when a stage fails, to unwind cleanly to main()."""


def run_stage(stage_name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a single integration stage with uniform pass/fail reporting.

    Args:
        stage_name: Human-readable name of the stage, used in PASS/FAIL
            output (e.g. ``"Mission created"``).
        fn: The callable implementing the stage. This should always be
            an existing function imported from ``train.py``.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns, unchanged.

    Raises:
        IntegrationTestFailure: If ``fn`` raises any exception. The
            original traceback is printed before this is raised.
    """
    try:
        result = fn(*args, **kwargs)
    except Exception:
        print(f"\n[FAIL] {stage_name}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        traceback.print_exc()
        print("-" * 60, file=sys.stderr)
        raise IntegrationTestFailure(stage_name)

    print(f"[PASS] {stage_name}")
    return result


def _fetch_one_batch(loader: torch.utils.data.DataLoader) -> Any:
    """Pull exactly one batch from a DataLoader to prove it is wired up.

    This only exercises dataset/collate/worker plumbing; it performs no
    model computation.

    Args:
        loader: The DataLoader to draw a single batch from.

    Returns:
        Any: Whatever the DataLoader's collate function produces for one
            batch.

    Raises:
        StopIteration: If the loader yields no batches at all.
    """
    return next(iter(loader))


def _devices_match(expected: torch.device, actual: torch.device) -> bool:
    """Compare two torch.device values leniently on device index.

    A ``torch.device`` constructed without an explicit index (e.g.
    ``torch.device("cuda")``) does not always compare equal to one with
    an explicit index (e.g. ``torch.device("cuda", 0)``) even though
    they refer to the same physical device. This helper treats such
    cases as a match by only comparing indices when both are specified.

    Args:
        expected: The device the object was supposed to be moved to.
        actual: The device the object is actually on.

    Returns:
        bool: True if the devices refer to the same type/index.
    """
    if expected.type != actual.type:
        return False
    if expected.index is None or actual.index is None:
        return True
    return expected.index == actual.index


def _verify_model_device(model: torch.nn.Module, expected_device: torch.device) -> None:
    """Verify the model's parameters live on the expected device.

    Args:
        model: The constructed PhySATFormer model.
        expected_device: The device selected earlier in the pipeline.

    Raises:
        AssertionError: If the model's parameters are on a different
            device than expected.
    """
    actual_device = next(model.parameters()).device
    if not _devices_match(expected_device, actual_device):
        raise AssertionError(
            f"Model device mismatch: expected {expected_device}, "
            f"got {actual_device}."
        )


def _verify_tensor_device(
    tensor: torch.Tensor,
    expected_device: torch.device,
    label: str,
) -> None:
    """Verify a tensor lives on the expected device.

    Args:
        tensor: The tensor to check (e.g. the physics matrix).
        expected_device: The device the tensor was moved to.
        label: Human-readable name used in the error message.

    Raises:
        AssertionError: If the tensor is on a different device than
            expected.
    """
    if not _devices_match(expected_device, tensor.device):
        raise AssertionError(
            f"{label} device mismatch: expected {expected_device}, "
            f"got {tensor.device}."
        )


def _verify_optimizer_has_parameters(optimizer: torch.optim.Optimizer) -> None:
    """Verify the optimizer was wired to at least one trainable parameter.

    Args:
        optimizer: The constructed optimizer.

    Raises:
        AssertionError: If the optimizer has zero parameters across all
            of its parameter groups.
    """
    num_params = sum(len(group["params"]) for group in optimizer.param_groups)
    if num_params == 0:
        raise AssertionError(
            "Optimizer was constructed with zero trainable parameters."
        )


def _verify_scheduler_attached(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Verify the scheduler references the same optimizer instance.

    Only performs the check if the scheduler exposes an ``optimizer``
    attribute; not every scheduler implementation is guaranteed to.

    Args:
        scheduler: The constructed scheduler.
        optimizer: The optimizer the scheduler should be attached to.

    Raises:
        AssertionError: If the scheduler's ``optimizer`` attribute
            exists but does not reference the same optimizer instance.
    """
    if not hasattr(scheduler, "optimizer"):
        return
    if scheduler.optimizer is not optimizer:
        raise AssertionError(
            "Scheduler.optimizer does not reference the constructed "
            "optimizer instance."
        )


def _verify_trainer_wiring(trainer: Any, expected: dict[str, Any]) -> None:
    """Verify the trainer's stored references match the constructed objects.

    Only checks attributes that actually exist on the trainer instance,
    since not every Trainer implementation is guaranteed to expose every
    collaborator under the same name.

    Args:
        trainer: The constructed Trainer.
        expected: Mapping of attribute name -> the object that attribute
            is expected to be (via identity, ``is``) if present on the
            trainer.

    Raises:
        AssertionError: If an attribute exists on the trainer but does
            not refer to the expected object.
    """
    for attr_name, expected_obj in expected.items():
        if not hasattr(trainer, attr_name):
            continue
        actual_obj = getattr(trainer, attr_name)
        if actual_obj is not expected_obj:
            raise AssertionError(
                f"Trainer.{attr_name} does not match the object passed "
                "to build_trainer()."
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the integration test.

    Returns:
        argparse.Namespace: Parsed arguments containing paths to the
            three configuration files and an optional device override.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run an end-to-end PhySATFormer integration smoke test: "
            "construct every training component without running training."
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


def main() -> None:
    """Run every integration stage in order and report the final result."""
    args = parse_arguments()

    print("=" * 60)
    print("PhySATFormer -- End-to-End Integration Test")
    print("=" * 60)

    try:
        # ------------------------------------------------------------
        # 1. Load configs
        # ------------------------------------------------------------
        train_cfg, model_cfg, dataset_cfg = run_stage(
            "Loaded configs",
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
        # 3. Set random seed
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
        # 5. Initialize AMP (setup only -- no autocast context is entered)
        # ------------------------------------------------------------
        amp_enabled, scaler = run_stage(
            "AMP initialized",
            setup_amp,
            mixed_precision=train_cfg.get("mixed_precision", False),
            device=device,
        )
        logger.info("AMP enabled: %s (scaler=%s)", amp_enabled, scaler is not None)

        # ------------------------------------------------------------
        # 6. Construct Mission
        # ------------------------------------------------------------
        mission = run_stage("Mission created", build_mission, dataset_cfg, logger)

        # ------------------------------------------------------------
        # 7. Construct TelemetryPipeline
        # ------------------------------------------------------------
        pipeline = run_stage(
            "TelemetryPipeline created", build_pipeline, dataset_cfg, logger
        )

        # ------------------------------------------------------------
        # 8. Construct train/validation/test datasets
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
        # 9. Construct train/validation/test dataloaders
        # ------------------------------------------------------------
        train_loader, validation_loader, test_loader = run_stage(
            "DataLoaders created (train/validation/test)",
            build_dataloaders,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            train_cfg=train_cfg,
            logger=logger,
        )

        run_stage(
            "Train DataLoader yielded one batch", _fetch_one_batch, train_loader
        )
        run_stage(
            "Validation DataLoader yielded one batch",
            _fetch_one_batch,
            validation_loader,
        )
        run_stage(
            "Test DataLoader yielded one batch", _fetch_one_batch, test_loader
        )

        # ------------------------------------------------------------
        # 10. Construct PhysicsRelationshipMatrix
        # ------------------------------------------------------------
        physics_matrix = run_stage(
            "PhysicsRelationshipMatrix created",
            build_physics_matrix,
            mission,
            logger,
        )
        physics_matrix = physics_matrix.to(device)
        run_stage(
            "Physics matrix moved to expected device",
            _verify_tensor_device,
            physics_matrix,
            device,
            "Physics matrix",
        )

        # ------------------------------------------------------------
        # 11. Construct PhySATFormer
        # ------------------------------------------------------------
        model = run_stage(
            "PhySATFormer created",
            build_model,
            model_cfg=model_cfg,
            physics_matrix=physics_matrix,
            device=device,
            logger=logger,
        )
        run_stage(
            "Model moved to expected device", _verify_model_device, model, device
        )

        # ------------------------------------------------------------
        # 12. Construct Optimizer
        # ------------------------------------------------------------
        optimizer = run_stage(
            "Optimizer created", build_optimizer, model, train_cfg, logger
        )
        run_stage(
            "Optimizer contains trainable parameters",
            _verify_optimizer_has_parameters,
            optimizer,
        )

        # ------------------------------------------------------------
        # 13. Construct Scheduler
        # ------------------------------------------------------------
        scheduler = run_stage(
            "Scheduler created", build_scheduler, optimizer, train_cfg, logger
        )
        run_stage(
            "Scheduler attached to optimizer",
            _verify_scheduler_attached,
            scheduler,
            optimizer,
        )

        # ------------------------------------------------------------
        # 14. Construct Loss
        # ------------------------------------------------------------
        criterion = run_stage("Loss created", build_criterion, logger)

        # ------------------------------------------------------------
        # 15. Construct Metrics
        # ------------------------------------------------------------
        metrics = run_stage("Metrics created", build_metrics, logger)

        # ------------------------------------------------------------
        # 16. Construct Trainer
        # ------------------------------------------------------------
        trainer = run_stage(
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
        run_stage(
            "Trainer wiring verified",
            _verify_trainer_wiring,
            trainer,
            {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "criterion": criterion,
                "loss_fn": criterion,
                "metrics": metrics,
            },
        )

    except IntegrationTestFailure as failure:
        print(f"\nIntegration test aborted at stage: {failure}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()