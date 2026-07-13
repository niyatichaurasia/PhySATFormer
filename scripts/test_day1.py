"""Day 1 integration test."""

from src.utils.constants import MISSION_ROOT

from src.core.mission import Mission
from src.preprocessing.analyzer import TelemetryAnalyzer
from src.preprocessing.assembler import TelemetryAssembler


def main() -> None:
    print("=" * 60)
    print("PhySATFormer - Day 1 Integration Test")
    print("=" * 60)

    # Mission
    print("\n1. Loading Mission...")
    mission = Mission(MISSION_ROOT)
    print("PASS")
    print(mission)

    # Metadata
    print("\n2. Loading Metadata...")

    labels = mission.labels
    telecommands = mission.telecommands
    anomaly_types = mission.anomaly_types

    print("PASS")
    print(f"Channels      : {mission.num_channels}")
    print(f"Labels        : {len(labels)}")
    print(f"Telecommands  : {len(telecommands)}")
    print(f"Anomaly Types : {len(anomaly_types)}")

    # Channel
    print("\n3. Loading Channel...")
    channel = mission.get_channel(1)
    print(channel)

    # Telemetry
    print("\n4. Loading Telemetry...")
    telemetry = channel.load()

    print(telemetry.head())
    print(f"Shape: {telemetry.shape}")

    # Inspector
    print("\n5. Channel Inspection...")
    report = channel.inspect()

    print(f"Rows    : {report['rows']}")
    print(f"Columns : {report['columns']}")
    print(f"Memory  : {report["memory_usage_mb"]:.2f} MB")

    # Analyzer
    print("\n6. Telemetry Analysis...")
    analyzer = TelemetryAnalyzer()

    analysis = analyzer.analyze(telemetry)

    print(f"Duration                : {analysis['duration']}")
    print(f"Start Timestamp         : {analysis['start_timestamp']}")
    print(f"End Timestamp           : {analysis['end_timestamp']}")
    print(f"Inferred Frequency      : {analysis['inferred_sampling_frequency']}")
    print(f"Irregular Sampling      : {analysis['irregular_sampling_detected']}")
    print(f"Duplicate Timestamps    : {analysis['duplicate_timestamps']}")

    # Assembler
    print("\n7. Assembling Channels...")

    assembler = TelemetryAssembler()

    assembled = assembler.assemble(
        mission,
        channel_ids=[1, 2, 3],
    )

    assembled = assembled.iloc[:50000]

    print(f"Development rows: {len(assembled)}")
    print(assembled.head())
    print(f"Shape: {assembled.shape}")

    print("\n" + "=" * 60)
    print("DAY 1 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()