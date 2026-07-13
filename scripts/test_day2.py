"""Day 2 integration test."""

from src.utils.constants import MISSION_ROOT
from src.core.mission import Mission
from src.preprocessing.pipeline import TelemetryPipeline


def main() -> None:

    print("=" * 60)
    print("PhySATFormer - Day 2 Integration Test")
    print("=" * 60)

    mission = Mission(MISSION_ROOT)

    pipeline = TelemetryPipeline(
        window_size=128,
        stride=1,
        normalization_method="zscore",
        train_ratio=0.70,
        validation_ratio=0.15,
        random_seed=42,
        direction="nearest",
    )

    train_dataset, validation_dataset, test_dataset = pipeline.build(
        mission=mission,
        channel_ids=[1, 2, 3],
        max_rows=50000,
    )

    print("\nDatasets")

    print(f"Train      : {len(train_dataset)}")
    print(f"Validation : {len(validation_dataset)}")
    print(f"Test       : {len(test_dataset)}")

    index, window = train_dataset[0]

    print("\nFirst Sample")

    print(f"Index : {index}")
    print(f"Shape : {window.shape}")
    print(f"Dtype : {window.dtype}")

    print("\nStatistics")

    print(f"Mean : {window.mean().item():.6f}")
    print(f"Std  : {window.std().item():.6f}")

    print("\n" + "=" * 60)
    print("DAY 2 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()