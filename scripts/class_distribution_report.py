"""
class_distribution_report.py — run this against your real Mission1 dataset.
Uses only existing, unmodified project components.
"""
import numpy as np
from src.core.mission import Mission
from src.preprocessing.assembler import TelemetryAssembler
from src.preprocessing.interval_label_generator import IntervalLabelGenerator
from src.utils.constants import CHANNEL_ID_COLUMN

ROOT = "C:/Users/Niyati Chaurasia/STORAGE/DATASETS/Mission1"
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # match dataset.yaml

mission = Mission(root=ROOT)
channel_ids = mission.channels[CHANNEL_ID_COLUMN].tolist()

synced = TelemetryAssembler(direction="forward").assemble(mission, channel_ids)

n = len(synced)
train_end = int(np.floor(TRAIN_RATIO * n))
val_end = train_end + int(np.floor(VAL_RATIO * n))

splits = {
    "train": synced.iloc[:train_end],
    "validation": synced.iloc[train_end:val_end],
    "test": synced.iloc[val_end:],
}

gen = IntervalLabelGenerator()
for name, df in splits.items():
    labels = gen.generate(mission, df)
    overall_rate = labels.mean()
    per_channel_rate = labels.mean(axis=0)
    print(f"\n=== {name} split ===")
    print(f"rows: {len(df)}")
    print(f"overall anomaly rate: {overall_rate:.6f}  ({int(labels.sum())} / {labels.size} entries)")
    print("top 5 most-anomalous channels:")
    top5 = np.argsort(per_channel_rate)[::-1][:5]
    for idx in top5:
        print(f"  {channel_ids[idx]}: {per_channel_rate[idx]:.6f}")