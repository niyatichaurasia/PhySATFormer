from pathlib import Path

from src.core.mission import Mission
from src.preprocessing.assembler import TelemetryAssembler
from src.preprocessing.analyzer import TelemetryAnalyzer



MISSION_ROOT = Path(
    r"C:\Users\Niyati Chaurasia\STORAGE\DATASETS\Mission1"
)

def main():

    print("=" * 60)
    print("PhySATFormer - Day 1 Integration Test")
    print("=" * 60)

    print("\n1. Loading Mission...")

    mission = Mission(MISSION_ROOT)

    print("PASS")
    print(mission)

    print("\n2. Loading Metadata...")

    channels = mission.channels

    print(f"PASS ({len(channels)} channels)")

    print("\n3. Loading Channel...")

    channel = mission.get_channel("channel_1")

    print(channel)

    print("\n4. Loading Telemetry...")

    df = channel.load()

    print(df.head())
    print(df.shape)

    print("\n5. Running Channel Inspector...")

    report = channel.inspect()

    print("Rows:", report["rows"])
    print("Columns:", report["columns"])
    print("Memory:", report["memory_usage_mb"], "MB")

    print("\n6. Running Telemetry Analyzer...")

    analyzer = TelemetryAnalyzer()

    analysis = analyzer.analyze(df)

    print("Duration:", analysis["duration"])
    print("Frequency:", analysis["inferred_sampling_frequency"])

    print("\n7. Running Telemetry Assembler...")

    assembler = TelemetryAssembler()

    assembled = assembler.assemble(
        mission,
        [
            "channel_1",
            "channel_2",
            "channel_3",
        ],
    )
    # ----------------------------------------
    # Development mode
    # Limit data size for fast testing
    # ----------------------------------------
    assembled = assembled.iloc[:50000]

    print(f"Using {len(assembled)} rows for testing.")


    print(assembled.head())

    print("Shape:", assembled.shape)

    print("\n" + "=" * 60)
    print("DAY 1 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()