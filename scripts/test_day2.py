
from src.core.mission import Mission

from src.preprocessing.pipeline import TelemetryPipeline

from src.utils.constants import MISSION_ROOT

def main():

    print("=" * 60)
    print("PhySATFormer - Day 2 Integration Test")
    print("=" * 60)

    mission = Mission(MISSION_ROOT)

    pipeline = TelemetryPipeline(
        window_size=128,
        stride=32,
        normalization_method="zscore",
        train_ratio=0.70,
        validation_ratio=0.15,
        random_seed=42,
        direction="nearest",
    )

    train_ds, val_ds, test_ds = pipeline.build(
        mission,
        [
            "channel_1",
            "channel_2",
            "channel_3",
        ],
    )



    print("\nDatasets")

    print("Train:", len(train_ds))
    print("Validation:", len(val_ds))
    print("Test:", len(test_ds))

    print("\nChecking first sample...")

    sample = train_ds[0]

    if len(sample) == 2:

        idx, window = sample

        print("Index:", idx)
        print("Shape:", window.shape)
        print("dtype:", window.dtype)

    else:

        idx, window, label = sample

        print("Index:", idx)
        print("Shape:", window.shape)
        print("dtype:", window.dtype)
        print("Label:", label)

    print("\nStatistics")

    print("Mean:", float(window.mean()))
    print("Std :", float(window.std()))

    print("\n" + "=" * 60)
    print("DAY 2 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()