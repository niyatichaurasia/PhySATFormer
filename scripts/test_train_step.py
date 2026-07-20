"""
scripts/test_train_step.py
===========================

Single-batch train-step smoke test for PhySATFormer.

This is NOT a unit test and NOT the integration test
(``scripts/test_integration.py``). ``test_integration.py`` proves that
every component *constructs* correctly; this script proves that the
constructed pipeline can actually execute one complete, real
forward -> loss -> backward -> gradient-clip -> optimizer-step
iteration without raising an exception and without producing
non-finite values.

It reuses the *existing* orchestration functions from ``train.py``
directly (``load_configs``, ``setup_logging``, ``set_random_seed``,
``select_device``, ``build_mission``, ``build_pipeline``,
``build_datasets``, ``build_dataloaders``, ``build_physics_matrix``,
``build_model``, ``build_optimizer``, ``build_scheduler``,
``build_criterion``, ``build_metrics``, ``build_trainer``) to build the
full pipeline, exactly as ``train.py`` and ``test_integration.py`` do.
No pipeline-construction logic is duplicated here.

For the training step itself, this script also reuses ``Trainer``'s own
batch-handling and gradient-clipping helpers (``Trainer._unpack_batch``,
``Trainer._move_to_device``, ``Trainer._clip_gradients``) rather than
reimplementing them, so that gradient clipping is performed *exactly*
as ``Trainer._training_step`` performs it -- same
``gradient_clip_value``, same ``torch.nn.utils.clip_grad_norm_`` call.
The forward/loss/backward/optimizer-step/zero-grad calls themselves are
plain, explicit statements in this script (rather than a single call to
the private ``Trainer._training_step``) so that this script can insert
its own finiteness checks *between* each stage, as required.

What this script deliberately does NOT do:
    * iterate through the dataset (only one batch is ever fetched)
    * train for an epoch, or call ``trainer.fit()``
    * run validation
    * save or load checkpoints
    * step the LR scheduler (the normal training loop, per
      ``Trainer.fit()``, steps the scheduler once per epoch, after
      validation -- never per batch -- so a single-batch smoke test
      correctly never calls ``scheduler.step()``)
    * write anything to disk
    * modify any existing project file

Usage:
    python scripts/test_train_step.py
    python scripts/test_train_step.py --device cpu
    python scripts/test_train_step.py --train-config configs/train.yaml \\
        --model-config configs/model.yaml \\
        --dataset-config configs/dataset.yaml

Exit status:
    0  -> the single train step completed successfully
    1  -> a stage raised an exception, or a verification failed
          (full traceback printed to stderr)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, TypeVar

import torch

# ---------------------------------------------------------------------------
# Make the project root importable regardless of the current working
# directory this script is invoked from (e.g. `python scripts/test_train_step.py`
# run from the repo root, or from inside `scripts/`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Reuse the real orchestration functions from train.py. Nothing about
# pipeline *construction* is reimplemented -- every step below just calls
# into the existing code.
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


class TrainStepTestFailure(Exception):
    """Raised internally when a stage fails, to unwind cleanly to main()."""


# ---------------------------------------------------------------------------
# Uniform stage runner (same reporting style as scripts/test_integration.py)
# ---------------------------------------------------------------------------


def run_stage(stage_name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a single stage with uniform PASS/FAIL reporting.

    Args:
        stage_name: Human-readable name of the stage, used in PASS/FAIL
            output (e.g. ``"Forward pass executed"``).
        fn: The callable implementing the stage.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns, unchanged.

    Raises:
        TrainStepTestFailure: If ``fn`` raises any exception. The
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
        raise TrainStepTestFailure(stage_name)

    print(f"[PASS] {stage_name}")
    return result


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _fetch_one_batch(loader: torch.utils.data.DataLoader) -> Any:
    """Pull exactly one batch from the training DataLoader.

    Args:
        loader: The DataLoader to draw a single batch from.

    Returns:
        Any: Whatever the DataLoader's collate function produces for one
            batch (an ``(index, window, label)`` triple).

    Raises:
        StopIteration: If the loader yields no batches at all.
    """
    return next(iter(loader))


def _unpack_and_move_batch(
    trainer: Trainer,
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split and move a batch using Trainer's own (reused) helpers.

    Args:
        trainer: The constructed Trainer, whose ``_unpack_batch`` and
            ``_move_to_device`` helpers are reused verbatim so batch
            handling matches ``Trainer._training_step`` exactly.
        batch: The raw ``(index, window, label)`` batch from the
            DataLoader.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(inputs, targets)``, both
        moved to ``trainer.device``.
    """
    inputs, targets = trainer._unpack_batch(batch)
    inputs = trainer._move_to_device(inputs)
    targets = trainer._move_to_device(targets)
    return inputs, targets


def _verify_input_target_shapes(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Verify the fetched batch has the expected 3-D, matching shape.

    Args:
        inputs: The telemetry window batch, expected shape
            ``(batch_size, sequence_length, num_channels)``.
        targets: The label batch, expected to match ``inputs.shape``.

    Raises:
        AssertionError: If either tensor is not 3-D, or if their shapes
            do not match.
    """
    if inputs.dim() != 3:
        raise AssertionError(
            "Expected input batch of shape (batch_size, sequence_length, "
            f"num_channels), got tensor with shape {tuple(inputs.shape)}."
        )
    if targets.shape != inputs.shape:
        raise AssertionError(
            "Input and target batch shapes must match, got "
            f"{tuple(inputs.shape)} and {tuple(targets.shape)}."
        )


def _snapshot_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clone every trainable parameter's current values.

    Args:
        model: The model whose parameters will be snapshotted.

    Returns:
        Dict[str, torch.Tensor]: Mapping of parameter name to a detached
        CPU clone of its current values.

    Raises:
        AssertionError: If the model has no trainable parameters.
    """
    snapshot = {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not snapshot:
        raise AssertionError("Model has no trainable parameters to snapshot.")
    return snapshot


def _forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run the forward pass, producing raw (pre-sigmoid) logits."""
    return model(inputs)


def _verify_output_shape(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Verify the model's output shape matches the target shape.

    Args:
        logits: Raw model output, expected shape
            ``(batch_size, sequence_length, num_channels)``.
        targets: Ground-truth labels of the same expected shape.

    Raises:
        AssertionError: If ``logits.shape != targets.shape``.
    """
    if logits.shape != targets.shape:
        raise AssertionError(
            f"Model output shape {tuple(logits.shape)} does not match "
            f"target shape {tuple(targets.shape)}."
        )


def _compute_loss(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the scalar training loss via the injected criterion."""
    return criterion(logits, targets)


def _verify_loss(loss: torch.Tensor) -> None:
    """Verify the loss is a finite, 0-D (scalar) tensor.

    Args:
        loss: The loss tensor returned by ``criterion(logits, targets)``.

    Raises:
        AssertionError: If ``loss`` is not 0-D, or is not finite
            (NaN or +/-inf).
    """
    if loss.dim() != 0:
        raise AssertionError(
            f"Expected a scalar (0-D) loss tensor, got shape {tuple(loss.shape)}."
        )
    if not torch.isfinite(loss):
        raise AssertionError(f"Loss is not finite: {loss.item()!r}.")


def _backward(loss: torch.Tensor) -> None:
    """Run backpropagation on the scalar loss."""
    loss.backward()


def _verify_gradients_finite(model: torch.nn.Module) -> None:
    """Verify every trainable parameter received a finite gradient.

    Args:
        model: The model that was just backpropagated through.

    Raises:
        AssertionError: If any trainable parameter has no gradient at
            all, or has a gradient containing NaN/inf values.
    """
    found_any_gradient = False
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            raise AssertionError(
                f"Trainable parameter '{name}' has no gradient after "
                "backward()."
            )
        found_any_gradient = True
        if not torch.isfinite(param.grad).all():
            raise AssertionError(
                f"Gradient of parameter '{name}' contains non-finite "
                "values (NaN or inf)."
            )

    if not found_any_gradient:
        raise AssertionError("No trainable parameters received gradients.")


def _clip_gradients(trainer: Trainer) -> None:
    """Clip gradients using Trainer's own (reused) clipping helper.

    Reusing ``Trainer._clip_gradients`` guarantees this script performs
    gradient clipping with exactly the same ``gradient_clip_value`` and
    the same ``torch.nn.utils.clip_grad_norm_`` call that
    ``Trainer._training_step`` uses during real training.

    Args:
        trainer: The constructed Trainer.
    """
    trainer._clip_gradients()


def _optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    """Apply one optimizer update using the currently accumulated gradients."""
    optimizer.step()


def _zero_grad(optimizer: torch.optim.Optimizer) -> None:
    """Clear accumulated gradients."""
    optimizer.zero_grad()


def _verify_parameters_changed(
    before: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    """Verify at least one trainable parameter changed after the step.

    Args:
        before: Snapshot captured via :func:`_snapshot_parameters` prior
            to ``optimizer.step()``.
        model: The model after ``optimizer.step()`` has been called.

    Raises:
        AssertionError: If no trainable parameter's values changed.
    """
    for name, param in model.named_parameters():
        if name not in before:
            continue
        if not torch.equal(before[name], param.detach().cpu()):
            return

    raise AssertionError(
        "No trainable parameter changed after optimizer.step(); the "
        "optimizer update appears to have had no effect."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the train-step smoke test.

    Returns:
        argparse.Namespace: Parsed arguments containing paths to the
            three configuration files and an optional device override.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-batch PhySATFormer train-step smoke test: "
            "one forward/backward/optimizer-step iteration, verified "
            "end to end."
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
    """Build the pipeline and run exactly one train step, end to end."""
    args = parse_arguments()

    print("=" * 60)
    print("PhySATFormer -- Single-Batch Train-Step Smoke Test")
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
        train_loader, _validation_loader, _test_loader = run_stage(
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
        # 10. Construct Trainer
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
        logger.info(
            "gradient_clip_value=%s (used for the clipping stage below)",
            trainer.gradient_clip_value,
        )

        # ------------------------------------------------------------
        # 11. Fetch exactly ONE batch from the training DataLoader
        # ------------------------------------------------------------
        raw_batch = run_stage(
            "Fetched one training batch", _fetch_one_batch, train_loader
        )
        inputs, targets = run_stage(
            "Batch unpacked and moved to device",
            _unpack_and_move_batch,
            trainer,
            raw_batch,
        )
        run_stage(
            "Input/target batch shapes verified",
            _verify_input_target_shapes,
            inputs,
            targets,
        )
        logger.info(
            "Batch shape: inputs=%s targets=%s device=%s",
            tuple(inputs.shape),
            tuple(targets.shape),
            inputs.device,
        )

        # ------------------------------------------------------------
        # 12. One complete training iteration
        # ------------------------------------------------------------
        run_stage("Model set to train mode", trainer.model.train)

        params_before = run_stage(
            "Captured pre-step parameter snapshot",
            _snapshot_parameters,
            trainer.model,
        )

        logits = run_stage(
            "Forward pass executed", _forward, trainer.model, inputs
        )
        run_stage(
            "Output tensor shape verified",
            _verify_output_shape,
            logits,
            targets,
        )

        loss = run_stage(
            "Loss computed", _compute_loss, trainer.loss_fn, logits, targets
        )
        run_stage("Loss verified finite and scalar", _verify_loss, loss)
        logger.info("Training loss (pre-step, single batch): %.6f", loss.item())

        run_stage("Backward pass executed", _backward, loss)
        run_stage(
            "Gradients verified finite", _verify_gradients_finite, trainer.model
        )

        run_stage(
            "Gradients clipped (Trainer._clip_gradients)",
            _clip_gradients,
            trainer,
        )

        run_stage("Optimizer step executed", _optimizer_step, trainer.optimizer)

        # NOTE: The normal training loop (Trainer.fit) only calls
        # `scheduler.step()` once per *epoch*, after validation -- never
        # once per batch. Since this script executes a single batch and
        # never completes an epoch, stepping the scheduler here would
        # not match real training behavior, so it is intentionally
        # skipped.
        logger.info(
            "Scheduler.step() skipped: Trainer.fit() only steps the "
            "scheduler once per epoch (after validation), never per batch."
        )

        run_stage("Gradients zeroed", _zero_grad, trainer.optimizer)

        run_stage(
            "At least one trainable parameter changed",
            _verify_parameters_changed,
            params_before,
            trainer.model,
        )

        # ------------------------------------------------------------
        # 13. Batch-level metrics sanity check (reuses PhySATMetrics;
        #     purely informational, not part of the optimization step)
        # ------------------------------------------------------------
        with torch.no_grad():
            batch_metrics = run_stage(
                "Metrics computed for the batch",
                trainer.metrics,
                logits.detach(),
                targets,
            )
        logger.info(
            "Batch metrics: precision=%.4f recall=%.4f f1=%.4f",
            batch_metrics["overall_precision"].item(),
            batch_metrics["overall_recall"].item(),
            batch_metrics["overall_f1"].item(),
        )

    except TrainStepTestFailure as failure:
        print(f"\nTrain-step smoke test aborted at stage: {failure}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("TRAIN STEP SMOKE TEST PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()