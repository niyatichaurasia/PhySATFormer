"""
train.py
========

Orchestration entry point for PhySATFormer training.

This script is intentionally limited to orchestration logic. It is
responsible for parsing command-line arguments, loading configuration
files, constructing the preprocessing pipeline, datasets, dataloaders,
physics matrix, model, optimizer, scheduler, and trainer, and finally
invoking the training loop. No machine learning logic (model
definitions, loss computation, training/validation steps, or dataset
construction internals) is implemented in this file; those concerns
live in their respective modules under ``src/``.

Usage:
    python train.py
    python train.py --resume checkpoints/last_checkpoint.pt
    python train.py --device cuda
"""

# Standard library
import argparse
import logging
import random
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import torch
import yaml

# Project
from src.utils.constants import LOGGER_NAME
# (none required for this module)

# =============================================================================
# Additional imports required for Modules 4-6
# =============================================================================

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.mission import Mission
from src.preprocessing.pipeline import TelemetryPipeline
from src.preprocessing.dataset import MissionDataset
from src.models.physics_matrix import PhysicsRelationshipMatrix
from src.utils.constants import CHANNEL_ID_COLUMN

# =============================================================================
# Additional imports required for Modules 7-12
# =============================================================================

from src.models.physatformer import PhySATFormer
from src.models.baseline_transformer import BaselineTransformer
from src.training.optimizer import OptimizerFactory
from src.training.scheduler import SchedulerFactory
from src.training.losses import PhySATLoss
from src.training.metrics import PhySATMetrics
from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the training entry point.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing
            paths to configuration files, an optional checkpoint to
            resume from, and an optional device override.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the PhySATFormer model for channel-level spacecraft "
            "telemetry anomaly localization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help=(
            "Device to use for training. If not provided, the device "
            "specified in train.yaml is used."
        ),
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to the dataset configuration YAML file.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a single YAML configuration file.

    The file is checked for existence, parsed with ``yaml.safe_load``,
    and validated to ensure it contains a non-empty top-level mapping.
    This function performs no semantic validation of the configuration
    contents; it only guarantees that a well-formed dictionary is
    returned.

    Args:
        path: Path to the YAML configuration file to load.

    Returns:
        dict[str, Any]: The parsed configuration as a dictionary.

    Raises:
        FileNotFoundError:
            If ``path`` does not exist or is not a regular file.

        ValueError:
            If the configuration file contains invalid YAML syntax,
            is empty, or does not contain a top-level dictionary.
    """
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: '{path}'. "
            "Please verify the path is correct."
        )

    with path.open("r", encoding="utf-8") as config_file:
        try:
            contents: Any = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML syntax in configuration file '{path}'."
            ) from error

    if contents is None or not contents:
        raise ValueError(
            f"Configuration file '{path}' is empty. "
            "A non-empty mapping of configuration keys is required."
        )

    if not isinstance(contents, dict):
        raise ValueError(
            f"Configuration file '{path}' must contain a top-level "
            f"mapping (dictionary), but got {type(contents).__name__}."
        )

    return contents


def load_configs(
    train_config: Path,
    model_config: Path,
    dataset_config: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load the training, model, and dataset configuration files.

    Each configuration file is loaded independently via :func:`load_yaml`.
    No merging, cross-validation, or semantic checking of hyperparameters
    is performed; the raw parsed dictionaries are returned as-is.

    Args:
        train_config: Path to the training configuration YAML file
            (e.g. ``configs/train.yaml``).
        model_config: Path to the model configuration YAML file
            (e.g. ``configs/model.yaml``).
        dataset_config: Path to the dataset configuration YAML file
            (e.g. ``configs/dataset.yaml``).

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: A
        3-tuple ``(train_cfg, model_cfg, dataset_cfg)`` containing the
        parsed configuration dictionaries in that order.

    Raises:
        FileNotFoundError:
            If any configuration file does not exist.

        ValueError:
            If any configuration file contains invalid YAML,
            is empty, or does not contain a top-level dictionary.
    """
    train_cfg = load_yaml(train_config)
    model_cfg = load_yaml(model_config)
    dataset_cfg = load_yaml(dataset_config)

    return train_cfg, model_cfg, dataset_cfg

# =============================================================================
# Module 3: Runtime Setup
# =============================================================================


def setup_logging() -> logging.Logger:
    """Configure and return the project-wide logger.

    The logger is configured with a single ``StreamHandler`` and a
    formatter that includes the timestamp, log level, logger name, and
    message. Configuration is performed only once; subsequent calls
    return the already-configured logger without adding duplicate
    handlers.

    Returns:
        logging.Logger: The configured project logger.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger


def set_random_seed(seed: int, deterministic: bool) -> None:
    """Seed all relevant random number generators for reproducibility.

    Seeds the ``random``, ``numpy``, and ``torch`` random number
    generators. If a CUDA device is available, all CUDA devices are
    also seeded. When ``deterministic`` is ``True``, PyTorch is
    configured to use deterministic algorithms where possible.

    Args:
        seed: The random seed to apply.
        deterministic: Whether to configure PyTorch for deterministic
            (reproducible) algorithm behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def select_device(
    configured_device: str,
    override_device: str | None,
) -> torch.device:
    """Select the compute device for training.

    A CLI-provided ``override_device`` takes precedence over the
    device specified in the configuration file. Supported values are
    ``"cpu"``, ``"cuda"``, and ``"auto"``. When ``"auto"`` is
    resolved, CUDA is used if available, otherwise CPU is used.

    Args:
        configured_device: The device value from the training
            configuration file (``"cpu"``, ``"cuda"``, or ``"auto"``).
        override_device: An optional device value from the CLI that
            overrides ``configured_device`` if provided.

    Returns:
        torch.device: The resolved compute device.

    Raises:
        RuntimeError: If ``"cuda"`` is explicitly requested but no
            CUDA device is available.
    """
    requested_device = (
        override_device
        if override_device is not None
        else configured_device
    )

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was explicitly requested but no CUDA "
                "device is available on this machine."
            )
        return torch.device("cuda")

    return torch.device("cpu")


def setup_amp(
    mixed_precision: bool,
    device: torch.device,
) -> tuple[bool, torch.amp.GradScaler | None]:
    """Configure automatic mixed precision (AMP) state.

    AMP is enabled only when ``mixed_precision`` is ``True`` and the
    resolved device is CUDA. This function only prepares the AMP
    enablement flag and gradient scaler; it does not create autocast
    contexts, which are the responsibility of the Trainer.

    Args:
        mixed_precision: Whether mixed precision was requested in the
            training configuration.
        device: The resolved compute device.

    Returns:
        tuple[bool, torch.amp.GradScaler | None]: A 2-tuple of
        ``(amp_enabled, scaler)``. ``scaler`` is a ``GradScaler``
        instance when AMP is enabled, otherwise ``None``.
    """
    amp_enabled = mixed_precision and device.type == "cuda"

    if amp_enabled:
        return True, torch.amp.GradScaler()

    return False, None




# =============================================================================
# Module 4: Mission & Preprocessing Pipeline
# =============================================================================


def build_mission(
    dataset_cfg: dict[str, Any],
    logger: logging.Logger,
) -> Mission:
    """Construct the Mission object from dataset configuration.

    Reads the mission root directory from the dataset configuration and
    constructs a :class:`Mission` instance. This function performs no
    telemetry inspection or preprocessing; it only validates configuration
    values and delegates construction to the existing ``Mission`` API.

    Args:
        dataset_cfg: Parsed contents of ``dataset.yaml``.
        logger: Project logger used to report progress.

    Returns:
        Mission: The constructed Mission object.

    Raises:
        KeyError: If the required ``root`` key is missing or ``None`` in
            ``dataset_cfg``.
    """
    if "root" not in dataset_cfg or dataset_cfg["root"] is None:
        raise KeyError(
            "Missing required dataset configuration key: 'root'. "
            "Please specify the mission root directory in dataset.yaml."
        )

    mission_root = Path(dataset_cfg["root"])

    logger.info("Constructing Mission from root=%s", mission_root)
    mission = Mission(root=mission_root)
    logger.info("Mission constructed: %s", mission)

    return mission


def build_pipeline(
    dataset_cfg: dict[str, Any],
    logger: logging.Logger,
) -> TelemetryPipeline:
    """Construct the TelemetryPipeline from dataset configuration.

    Reads preprocessing parameters from ``dataset.yaml`` and instantiates
    a :class:`TelemetryPipeline` using its existing constructor API. This
    function does not build datasets or dataloaders; it only prepares the
    pipeline object for later use.

    Args:
        dataset_cfg: Parsed contents of ``dataset.yaml``.
        logger: Project logger used to report progress.

    Returns:
        TelemetryPipeline: The constructed preprocessing pipeline.

    Raises:
        KeyError: If any required preprocessing configuration key is
            missing or ``None`` in ``dataset_cfg``.
    """
    required_keys = (
        "window_size",
        "stride",
        "normalization_method",
        "train_ratio",
        "validation_ratio",
        "random_seed",
        "direction",
    )
    missing_keys = [
        key for key in required_keys
        if key not in dataset_cfg or dataset_cfg[key] is None
    ]
    if missing_keys:
        raise KeyError(
            f"Missing required dataset configuration key(s): {missing_keys}. "
            "Please specify them in dataset.yaml."
        )

    logger.info(
        "Initializing TelemetryPipeline (window_size=%s, stride=%s, "
        "normalization_method=%s, train_ratio=%s, validation_ratio=%s, "
        "random_seed=%s, direction=%s)",
        dataset_cfg["window_size"],
        dataset_cfg["stride"],
        dataset_cfg["normalization_method"],
        dataset_cfg["train_ratio"],
        dataset_cfg["validation_ratio"],
        dataset_cfg["random_seed"],
        dataset_cfg["direction"],
    )

    pipeline = TelemetryPipeline(
        window_size=dataset_cfg["window_size"],
        stride=dataset_cfg["stride"],
        normalization_method=dataset_cfg["normalization_method"],
        train_ratio=dataset_cfg["train_ratio"],
        validation_ratio=dataset_cfg["validation_ratio"],
        random_seed=dataset_cfg["random_seed"],
        direction=dataset_cfg["direction"],
    )

    logger.info("TelemetryPipeline initialized successfully.")

    return pipeline


# =============================================================================
# Module 5: Datasets & DataLoaders
# =============================================================================


def build_datasets(
    pipeline: TelemetryPipeline,
    mission: Mission,
    dataset_cfg: dict[str, Any],
    logger: logging.Logger,
) -> tuple[MissionDataset, MissionDataset, MissionDataset]:
    """Build train/validation/test MissionDatasets using the pipeline.

    Note:
        ``TelemetryPipeline.build`` requires both a ``Mission`` and a list
        of ``channel_ids`` as input. Since ``mission`` is required by the
        existing pipeline API but was not part of the originally proposed
        signature, it has been added here as an explicit parameter so
        that this function can call the real ``pipeline.build`` interface
        without inventing a wrapper or hidden global state. Channel IDs
        are derived directly from ``mission.channels`` (the existing
        Mission metadata API), covering all channels present in the
        mission's metadata.

    Args:
        pipeline: The constructed TelemetryPipeline.
        mission: The Mission object to preprocess.
        dataset_cfg: Parsed contents of ``dataset.yaml`` (used only for
            the optional ``max_rows`` development setting).
        logger: Project logger used to report progress.

    Returns:
        tuple[MissionDataset, MissionDataset, MissionDataset]: The
        ``(train_dataset, validation_dataset, test_dataset)`` triple.

    Raises:
        KeyError: If mission channel metadata does not expose the
            expected channel identifier column.
    """
    channels = mission.channels

    if CHANNEL_ID_COLUMN not in channels.columns:
        raise KeyError(
            f"Mission.channels DataFrame does not contain the expected "
            f"'{CHANNEL_ID_COLUMN}' column; cannot determine channel_ids "
            "for pipeline construction."
        )

    channel_ids = channels[CHANNEL_ID_COLUMN].tolist()
    max_rows = dataset_cfg.get("max_rows")

    logger.info(
        "Building datasets for %d channels (max_rows=%s)",
        len(channel_ids),
        max_rows,
    )

    train_dataset, validation_dataset, test_dataset = pipeline.build(
        mission=mission,
        channel_ids=channel_ids,
        max_rows=max_rows,
    )

    logger.info(
        "Datasets built: train=%d validation=%d test=%d",
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
    )

    return train_dataset, validation_dataset, test_dataset


def configure_training_augmentation(
    train_dataset: MissionDataset,
    train_cfg: dict[str, Any],
    seed: int,
    logger: logging.Logger,
) -> None:
    """Enable anomaly-only augmentation on the training split, if configured.

    Reads the ``augmentation`` section of ``train.yaml``::

        augmentation:
          enabled: true
          probability: 0.6

    and, only when enabled, switches ``train_dataset`` into training
    mode with augmentation active via
    ``MissionDataset.enable_anomaly_augmentation``. This function must
    only ever be called with the *training* ``MissionDataset`` --
    ``validation_dataset`` and ``test_dataset`` are intentionally never
    passed to it anywhere in this file, so they keep their default
    ``training=False`` state and are never augmented, regardless of
    this configuration.

    Args:
        train_dataset: The training ``MissionDataset``, and only the
            training ``MissionDataset``.
        train_cfg: Parsed contents of ``train.yaml``.
        seed: The resolved experiment seed (see ``set_random_seed`` in
            ``main``), reused here so per-window augmentation draws are
            deterministic under the same seed that governs the rest of
            the run.
        logger: Project logger used to report progress.
    """
    augmentation_cfg = train_cfg.get("augmentation", {}) or {}
    augmentation_enabled = bool(augmentation_cfg.get("enabled", False))

    if not augmentation_enabled:
        logger.info(
            "Anomaly-only augmentation disabled (augmentation.enabled=false)."
        )
        return

    probability = float(augmentation_cfg.get("probability", 0.6))

    train_dataset.enable_anomaly_augmentation(probability=probability, seed=seed)

    # Explicit startup banner, in addition to the structured log line
    # below, so augmentation state is impossible to miss when scanning
    # console output at the start of a run.
    print("Augmentation enabled")
    print(f"Probability: {probability}")
    print("Augmenting anomaly windows only")

    logger.info(
        "Anomaly-only augmentation enabled on training dataset "
        "(probability=%.3f, seed=%s).",
        probability,
        seed,
    )


def compute_training_sample_weights(
    train_dataset: MissionDataset,
    logger: logging.Logger,
) -> torch.Tensor:
    """Compute per-window sampling weights for the training dataset.

    Each training window is classified using the existing, index-aligned
    ``MissionDataset`` contract:

        * "anomaly window" -- at least one channel-timestep in the
          window's label tensor is anomalous (label > 0).
        * "normal window"  -- every channel-timestep in the window is
          non-anomalous.

    Per-class weights are then derived automatically from the observed
    class counts via inverse-frequency weighting

        weight_c = num_windows / (num_classes * count_c)

    so that, in expectation, a ``WeightedRandomSampler`` built from the
    returned per-window weights draws normal and anomaly windows in
    roughly equal proportion each epoch -- no magic numbers, purely a
    function of the dataset's own statistics.

    This function only reads ``train_dataset`` through its existing
    public ``Dataset`` contract (``__len__`` / ``__getitem__``). It does
    not construct, mutate, or reorder the dataset in any way; window
    contents and their on-disk/back-end ordering are left untouched.
    Only the *sampling* of indices during training is later affected by
    the weights computed here.

    Args:
        train_dataset: The training MissionDataset.
        logger: Project logger used to report progress.

    Returns:
        torch.Tensor: A 1D float64 tensor of shape
        ``(len(train_dataset),)`` holding one sampling weight per
        training window, index-aligned with ``train_dataset``.

    Raises:
        ValueError: If the training dataset is empty, or if every
            window falls into a single class (balancing is impossible).
    """
    num_windows = len(train_dataset)
    if num_windows == 0:
        raise ValueError(
            "Cannot compute sample weights: train_dataset is empty."
        )

    is_anomaly = torch.zeros(num_windows, dtype=torch.bool)
    for idx in range(num_windows):
        # Only the label half of the (telemetry, label) pair is used;
        # this deliberately goes through MissionDataset's normal
        # __getitem__ so no assumptions are made about its internal
        # storage (LazyWindowSet vs. dense array) beyond the documented
        # Dataset contract.
        _, label_window = train_dataset[idx]
        is_anomaly[idx] = bool(torch.any(label_window > 0))

    num_anomaly = int(is_anomaly.sum().item())
    num_normal = num_windows - num_anomaly

    if num_anomaly == 0 or num_normal == 0:
        raise ValueError(
            "Cannot build a WeightedRandomSampler: the training dataset "
            f"contains only one class (normal={num_normal}, "
            f"anomaly={num_anomaly}). Set 'sampler.enabled: false' in "
            "train.yaml to fall back to standard shuffling."
        )

    # Inverse-frequency class weights, computed purely from observed
    # counts. Rarer windows (anomalies) receive proportionally larger
    # weights; no values are hardcoded.
    normal_weight = num_windows / (2.0 * num_normal)
    anomaly_weight = num_windows / (2.0 * num_anomaly)

    sample_weights = torch.where(
        is_anomaly,
        torch.full((num_windows,), anomaly_weight, dtype=torch.float64),
        torch.full((num_windows,), normal_weight, dtype=torch.float64),
    )

    logger.info("Training windows: %d", num_windows)
    logger.info("Normal windows: %d", num_normal)
    logger.info("Anomaly windows: %d", num_anomaly)
    logger.info("Normal weight: %.6f", normal_weight)
    logger.info("Anomaly weight: %.6f", anomaly_weight)

    return sample_weights


def build_train_sampler(
    train_dataset: MissionDataset,
    train_cfg: dict[str, Any],
    logger: logging.Logger,
) -> "WeightedRandomSampler | None":
    """Optionally construct a WeightedRandomSampler for the train loader.

    Controlled entirely by the ``sampler`` section of ``train.yaml``:

        sampler:
          enabled: true

    When ``sampler.enabled`` is missing or ``false``, this returns
    ``None`` and training behaves exactly as before -- ``build_dataloaders``
    falls back to plain ``shuffle=True`` for the training loader.
    Validation and test loaders are never affected by this setting; they
    are constructed with ``shuffle=False`` unconditionally.

    Args:
        train_dataset: The training MissionDataset.
        train_cfg: Parsed contents of ``train.yaml``.
        logger: Project logger used to report progress.

    Returns:
        Optional[WeightedRandomSampler]: The constructed sampler, or
        ``None`` if disabled.
    """
    sampler_cfg = train_cfg.get("sampler", {}) or {}
    sampler_enabled = bool(sampler_cfg.get("enabled", False))

    if not sampler_enabled:
        logger.info(
            "WeightedRandomSampler disabled (sampler.enabled=false); "
            "using standard shuffling for the training loader."
        )
        return None

    sample_weights = compute_training_sample_weights(train_dataset, logger)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    logger.info("Using WeightedRandomSampler")

    return sampler


def build_dataloaders(
    train_dataset: MissionDataset,
    validation_dataset: MissionDataset,
    test_dataset: MissionDataset,
    train_cfg: dict[str, Any],
    logger: logging.Logger,
    train_sampler: "WeightedRandomSampler | None" = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/validation/test DataLoaders.

    Reads ``batch_size``, ``num_workers``, and ``pin_memory`` from the
    training configuration.

    Training loader:
        * If ``train_sampler`` is provided (i.e. ``sampler.enabled: true``
          in train.yaml), it is used via the ``sampler=`` argument and
          ``shuffle`` is omitted (PyTorch's ``DataLoader`` disallows
          passing both ``shuffle=True`` and ``sampler`` together).
        * Otherwise the loader falls back to ``shuffle=True``, i.e. the
          original behavior.

    Validation and test loaders are always constructed with
    ``shuffle=False`` and are completely unaffected by ``train_sampler``.

    Args:
        train_dataset: The training MissionDataset.
        validation_dataset: The validation MissionDataset.
        test_dataset: The test MissionDataset.
        train_cfg: Parsed contents of ``train.yaml``.
        logger: Project logger used to report progress.
        train_sampler: An optional ``WeightedRandomSampler`` to use for
            the training loader in place of ``shuffle=True``. Pass
            ``None`` (the default) to preserve the original behavior.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The
        ``(train_loader, validation_loader, test_loader)`` triple.

    Raises:
        KeyError: If the required ``batch_size`` key is missing or
            ``None`` in ``train_cfg``.
    """
    if "batch_size" not in train_cfg or train_cfg["batch_size"] is None:
        raise KeyError(
            "Missing required training configuration key: 'batch_size'."
        )

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = train_cfg.get("pin_memory", False)

    # persistent_workers / prefetch_factor are only valid (and only make
    # sense) when num_workers > 0; DataLoader raises if passed otherwise.
    persistent_workers = train_cfg.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", None) if num_workers > 0 else None

    logger.info(
        "Building DataLoaders (batch_size=%s, num_workers=%s, "
        "pin_memory=%s, persistent_workers=%s, prefetch_factor=%s)",
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )

    loader_kwargs: dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if train_sampler is not None:
        # sampler= and shuffle=True are mutually exclusive in
        # torch.utils.data.DataLoader; the sampler already governs draw
        # order/frequency, so shuffle is simply omitted here.
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(
        "DataLoaders built: train_batches=%d validation_batches=%d "
        "test_batches=%d",
        len(train_loader),
        len(validation_loader),
        len(test_loader),
    )

    return train_loader, validation_loader, test_loader


# =============================================================================
# Module 6: Physics Matrix
# =============================================================================


def build_physics_matrix(
    mission: Mission,
    logger: logging.Logger,
) -> torch.FloatTensor:
    """Construct the physics relationship matrix for a mission.

    Delegates matrix construction entirely to the existing
    :class:`PhysicsRelationshipMatrix` implementation. No matrix
    generation logic is implemented in this function.

    Args:
        mission: The Mission object providing channel/subsystem metadata.
        logger: Project logger used to report progress.

    Returns:
        torch.FloatTensor: The physics relationship matrix of shape
        ``(num_channels, num_channels)``.
    """
    logger.info("Building physics relationship matrix...")

    builder = PhysicsRelationshipMatrix(mission)
    physics_matrix = builder.build()

    logger.info(
        "Physics relationship matrix constructed: shape=%s",
        tuple(physics_matrix.shape),
    )

    return physics_matrix


# =============================================================================
# Module 7: Model
# =============================================================================


def build_model(
    model_cfg: dict[str, Any],
    physics_matrix: torch.FloatTensor,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Instantiate the selected model from model.yaml and move it to `device`.

    Supports:
        - PhySATFormer
        - BaselineTransformer

    The model is selected using the `model_type` field in model.yaml.
    """

    required_keys = (
        "input_dim",
        "channel_embedding_dim",
        "d_model",
        "num_heads",
        "num_channel_layers",
        "num_temporal_layers",
        "ff_dim",
        "num_channels",
    )

    missing_keys = [
        key for key in required_keys
        if key not in model_cfg or model_cfg[key] is None
    ]

    if missing_keys:
        raise KeyError(
            f"Missing required model configuration key(s): {missing_keys}. "
            "Please specify them in model.yaml."
        )

    model_type = model_cfg.get("model_type", "physatformer").lower()

    logger.info("Building model: %s", model_type)

    if model_type == "physatformer":

        model = PhySATFormer(
            input_dim=model_cfg["input_dim"],
            channel_embedding_dim=model_cfg["channel_embedding_dim"],
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_channel_layers=model_cfg["num_channel_layers"],
            num_temporal_layers=model_cfg["num_temporal_layers"],
            ff_dim=model_cfg["ff_dim"],
            num_channels=model_cfg["num_channels"],
            physics_matrix=physics_matrix,
            dropout=model_cfg.get("dropout", 0.1),
        )

    elif model_type == "baseline":

        model = BaselineTransformer(
            input_dim=model_cfg["input_dim"],
            channel_embedding_dim=model_cfg["channel_embedding_dim"],
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["num_heads"],
            num_channel_layers=model_cfg["num_channel_layers"],
            num_temporal_layers=model_cfg["num_temporal_layers"],
            ff_dim=model_cfg["ff_dim"],
            num_channels=model_cfg["num_channels"],
            dropout=model_cfg.get("dropout", 0.1),
        )

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Supported values are: 'physatformer', 'baseline'."
        )

    model = model.to(device)

    logger.info(
        "%s built successfully and moved to %s",
        model.__class__.__name__,
        device,
    )

    return model


# =============================================================================
# Module 8: Optimizer / Scheduler / Loss
# =============================================================================


def build_optimizer(
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    logger: logging.Logger,
) -> torch.optim.AdamW:
    """Construct the AdamW optimizer via the existing `OptimizerFactory`.

    Args:
        model: The model whose parameters will be optimized.
        train_cfg: Parsed contents of ``train.yaml``.
        logger: Project logger used to report progress.

    Returns:
        torch.optim.AdamW: The constructed optimizer.

    Raises:
        KeyError: If ``learning_rate`` is missing or ``None`` in
            ``train_cfg``.
    """
    if "learning_rate" not in train_cfg or train_cfg["learning_rate"] is None:
        raise KeyError(
            "Missing required training configuration key: 'learning_rate'."
        )

    optimizer_cfg = train_cfg.get("optimizer", {}) or {}
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    eps = optimizer_cfg.get("eps", 1e-8)
    weight_decay = train_cfg.get("weight_decay", 0.0)

    logger.info(
        "Building AdamW optimizer (lr=%s, weight_decay=%s, betas=%s, eps=%s)",
        train_cfg["learning_rate"],
        weight_decay,
        betas,
        eps,
    )

    factory = OptimizerFactory(
        learning_rate=train_cfg["learning_rate"],
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    optimizer = factory.create_optimizer(model)

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    logger: logging.Logger,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Construct the CosineAnnealingLR scheduler via `SchedulerFactory`.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        train_cfg: Parsed contents of ``train.yaml``.
        logger: Project logger used to report progress.

    Returns:
        torch.optim.lr_scheduler.CosineAnnealingLR: The constructed
        scheduler.

    Raises:
        KeyError: If ``epochs`` is missing or ``None`` in `train_cfg`
            and no ``scheduler.t_max`` override is present.
    """
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}

    t_max = scheduler_cfg.get("t_max")
    if t_max is None:
        if "epochs" not in train_cfg or train_cfg["epochs"] is None:
            raise KeyError(
                "Missing required configuration: either 'scheduler.t_max' "
                "or 'epochs' must be specified in train.yaml."
            )
        t_max = train_cfg["epochs"]

    eta_min = scheduler_cfg.get("eta_min", 0.0)

    logger.info(
        "Building CosineAnnealingLR scheduler (T_max=%s, eta_min=%s)",
        t_max,
        eta_min,
    )

    factory = SchedulerFactory(T_max=t_max, eta_min=eta_min)
    scheduler = factory.create_scheduler(optimizer)

    return scheduler


def build_criterion(logger: logging.Logger) -> PhySATLoss:
    """Construct the training/validation loss.

    `Trainer` requires a `PhySATLoss` instance (it type-checks the
    injected `loss_fn` against that exact class), so `PhySATLoss` is
    reused here rather than substituting a bare
    ``torch.nn.BCEWithLogitsLoss``, which `Trainer` would reject.

    NOTE: `losses.py` was not among the attached files, so this calls
    `PhySATLoss()` with no arguments based only on how `trainer.py`
    invokes it (``self.loss_fn(predictions, targets)``, i.e. a plain
    callable). If `PhySATLoss` requires constructor arguments, this
    call must be updated to match its real signature.

    Args:
        logger: Project logger used to report progress.

    Returns:
        PhySATLoss: The constructed loss instance.
    """
    logger.info("Building PhySATLoss criterion.")  # ASSUMED API: PhySATLoss()
    return PhySATLoss()


# =============================================================================
# Module 9: Metrics
# =============================================================================


def build_metrics(logger: logging.Logger) -> PhySATMetrics:
    """Construct the metrics object used for precision/recall/F1.

    `Trainer` requires a `PhySATMetrics` instance (type-checked in its
    constructor), so the existing `PhySATMetrics` utility is reused
    rather than reimplementing metric computation.

    NOTE: `metrics.py` was not among the attached files. `PhySATMetrics`
    is instantiated here with no arguments, inferred only from
    `Trainer._MetricAccumulator`, which reads `self._metrics.eps`. If
    `PhySATMetrics` requires constructor arguments (e.g. a decision
    threshold), this call must be updated to match its real signature.

    Args:
        logger: Project logger used to report progress.

    Returns:
        PhySATMetrics: The constructed metrics instance.
    """
    logger.info("Building PhySATMetrics.")  # ASSUMED API: PhySATMetrics()
    return PhySATMetrics()


# =============================================================================
# Module 10: Trainer
# =============================================================================


def build_trainer(
    model: torch.nn.Module,
    criterion: PhySATLoss,
    metrics: PhySATMetrics,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_cfg: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    amp_enabled: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> Trainer:
    """Instantiate the existing `Trainer`, wiring in all collaborators.

    `Trainer`'s constructor accepts `model, loss_fn, metrics, optimizer,
    scheduler, checkpoint_manager, early_stopping, device,
    gradient_clip_value, amp_enabled, scaler`. `amp_enabled`/`scaler` are
    optional and default to off, so this call is backward-compatible
    with any caller that does not pass them.

    NOTE: `early_stopping.py` was not among the attached files.
    `EarlyStopping` is instantiated here as
    ``EarlyStopping(patience=..., min_delta=..., mode="max")``, inferred
    from `Trainer`'s docstring ("constructed by the caller with
    `mode="max"`", monitored on validation F1). If the real
    `EarlyStopping` constructor differs, this call must be updated.

    Args:
        model: The constructed PhySATFormer model.
        criterion: The constructed PhySATLoss.
        metrics: The constructed PhySATMetrics.
        optimizer: The constructed AdamW optimizer.
        scheduler: The constructed CosineAnnealingLR scheduler.
        train_cfg: Parsed contents of ``train.yaml``.
        device: The resolved compute device.
        logger: Project logger used to report progress.
        amp_enabled: Whether automatic mixed precision is enabled for
            this run (as resolved by `setup_amp`).
        scaler: The `torch.cuda.amp.GradScaler` instance to use when
            `amp_enabled` is True, otherwise `None`.

    Returns:
        Trainer: The constructed trainer, ready for `fit()`.

    Raises:
        KeyError: If ``checkpoint_dir`` is missing or ``None`` in
            `train_cfg`.
    """
    if "checkpoint_dir" not in train_cfg or train_cfg["checkpoint_dir"] is None:
        raise KeyError(
            "Missing required training configuration key: 'checkpoint_dir'."
        )

    checkpoint_manager = CheckpointManager(train_cfg["checkpoint_dir"])

    early_stopping_cfg = train_cfg.get("early_stopping", {}) or {}
    early_stopping = EarlyStopping(  # ASSUMED API
        patience=early_stopping_cfg.get("patience", 10),
        min_delta=early_stopping_cfg.get("min_delta", 0.0),
        mode="max",
    )

    gradient_clip_value = train_cfg.get("gradient_clip", 1.0)

    logger.info(
        "Building Trainer (checkpoint_dir=%s, gradient_clip=%s, "
        "early_stopping_patience=%s)",
        train_cfg["checkpoint_dir"],
        gradient_clip_value,
        early_stopping_cfg.get("patience", 10),
    )

    trainer = Trainer(
        model=model,
        loss_fn=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        device=device,
        gradient_clip_value=gradient_clip_value,
        amp_enabled=amp_enabled,
        scaler=scaler,
    )

    return trainer


# =============================================================================
# Module 11: Checkpoint Resume
# =============================================================================


def resume_checkpoint(
    resume_path: Path | None,
    trainer: Trainer,
    logger: logging.Logger,
) -> None:
    """Optionally restore model/optimizer/scheduler state from a checkpoint.

    Delegates entirely to `CheckpointManager.load_checkpoint`, which is
    the only checkpoint-restoration API that exists in the repository.

    LIMITATION: `CheckpointManager` only serializes/restores
    `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`,
    `epoch`, and `metric`. It has no support for an AMP grad-scaler or
    for early-stopping state, and `Trainer.fit()` has no `start_epoch`
    parameter — so resuming restores model/optimizer/scheduler weights,
    but training always restarts epoch numbering at 1, and early
    stopping's internal counters are not restored. Implementing either
    of those would require modifying `Trainer` or `CheckpointManager`,
    which is out of scope for this orchestration-only change.

    Args:
        resume_path: Path to a checkpoint file, or ``None`` to skip
            resuming.
        trainer: The constructed `Trainer`, whose `model`, `optimizer`,
            and `scheduler` will be restored in place.
        logger: Project logger used to report progress.
    """
    if resume_path is None:
        return 0

    logger.info("Resuming from checkpoint: %s", resume_path)

    metadata = trainer.checkpoint_manager.load_checkpoint(
        checkpoint_path=resume_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        device=trainer.device,
    )

    best_metric = metadata.get("metric")

    if best_metric is not None:
        trainer.set_best_validation_f1(best_metric)

    logger.info(
        "Checkpoint restored: epoch=%s metric=%s. Resuming training from epoch %s.",
        metadata.get("epoch"),
        metadata.get("metric"),
        metadata.get("epoch", 0) + 1,
    )

    return metadata.get("epoch", 0)

# =============================================================================
# Module 12: Main
# =============================================================================


def main() -> None:
    """Orchestrate the full PhySATFormer training run.

    Performs no ML logic itself; every step delegates to an existing,
    previously-implemented component.
    """
    args = parse_arguments()
    logger = setup_logging()

    train_cfg, model_cfg, dataset_cfg = load_configs(
        train_config=args.train_config,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
    )

    experiment_cfg = train_cfg.get("experiment", {}) or {}
    seed = experiment_cfg.get("seed", train_cfg.get("random_seed", 42))
    deterministic = experiment_cfg.get("deterministic", True)
    set_random_seed(seed=seed, deterministic=deterministic)

    device = select_device(
        configured_device=train_cfg.get("device", "auto"),
        override_device=args.device,
    )
    logger.info("Using device: %s", device)

    amp_enabled, scaler = setup_amp(
        mixed_precision=train_cfg.get("mixed_precision", False),
        device=device,
    )
    logger.info("Automatic mixed precision enabled: %s", amp_enabled)

    mission = build_mission(dataset_cfg, logger)
    pipeline = build_pipeline(dataset_cfg, logger)

    train_dataset, validation_dataset, test_dataset = build_datasets(
        pipeline=pipeline,
        mission=mission,
        dataset_cfg=dataset_cfg,
        logger=logger,
    )

    # NOTE: sampler weights must be computed BEFORE augmentation is
    # enabled. build_train_sampler() (via compute_training_sample_weights)
    # iterates train_dataset[idx] to inspect labels, which goes through
    # MissionDataset.__getitem__ -- if augmentation were already enabled,
    # that pass would needlessly consume augmentation RNG draws and
    # augment telemetry that's immediately discarded (only labels are
    # used for weighting). Running this first guarantees the sampler
    # always sees raw, unaugmented windows.
    train_sampler = build_train_sampler(
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        logger=logger,
    )

    # Augmentation is enabled only after sampler weights are computed.
    # DataLoader construction below is lazy (it does not call
    # __getitem__ until iterated during training), so training still
    # receives augmented anomaly windows exactly as before.
    configure_training_augmentation(
        train_dataset=train_dataset,
        train_cfg=train_cfg,
        seed=seed,
        logger=logger,
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_cfg=train_cfg,
        logger=logger,
        train_sampler=train_sampler,
    )

    physics_matrix = build_physics_matrix(mission, logger)
    physics_matrix = physics_matrix.to(device)

    model = build_model(
        model_cfg=model_cfg,
        physics_matrix=physics_matrix,
        device=device,
        logger=logger,
    )

    optimizer = build_optimizer(model, train_cfg, logger)
    scheduler = build_scheduler(optimizer, train_cfg, logger)
    criterion = build_criterion(logger)
    metrics = build_metrics(logger)

    trainer = build_trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=train_cfg,
        device=device,
        logger=logger,
        amp_enabled=amp_enabled,
        scaler=scaler,
    )

    last_epoch = resume_checkpoint(
        args.resume,
        trainer,
        logger,
    )

    start_epoch = last_epoch + 1

    if "epochs" not in train_cfg or train_cfg["epochs"] is None:
        raise KeyError("Missing required training configuration key: 'epochs'.")

    logger.info("Starting training for %s epochs.", train_cfg["epochs"])
    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=train_cfg["epochs"],
        start_epoch=start_epoch,
    )

    final_checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=len(history["val_f1"]),
        metric=history["val_f1"][-1] if history["val_f1"] else None,
        filename="last_checkpoint.pt",
    )
    logger.info("Final checkpoint saved to: %s", final_checkpoint_path)

    # Optional test evaluation, reusing Trainer's existing public
    # validate_epoch() method (generic over any loader) rather than
    # implementing new test-loop logic.
    test_metrics = trainer.validate_epoch(test_loader)
    logger.info(
        "Test evaluation: loss=%.4f precision=%.4f recall=%.4f f1=%.4f",
        test_metrics["loss"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()