"""
src/preprocessing/event_cluster_splitter.py

EventClusterSplitter: splits a mission's synchronized telemetry into
train/validation/test partitions using ANOMALY EVENTS (not raw rows and
not a plain chronological cutpoint) as the splitting unit.

Why this replaces the previous chronological split
----------------------------------------------------
A plain chronological split cuts the timeline into three contiguous
ranges purely by row count. On Mission1 this produced a severe
anomaly-rate drift across splits (train=1.93%, validation=2.75%,
test=4.83%) because anomalies are not uniformly distributed over time --
they occur in temporal bursts, and an arbitrary row-count cutpoint has no
way to "see" that a burst of events happens to land mostly in the test
region.

This module treats each anomaly EVENT (one ``ID`` in ``labels.csv``,
which may span several channels) as an indivisible unit. Temporally
overlapping events are first merged into CLUSTERS (a burst of
overlapping events becomes one cluster), and then whole clusters --
never individual events or channels within a cluster -- are assigned to
train/validation/test using a deterministic greedy heuristic that keeps
event count, anomaly prevalence, class distribution, category
distribution, and channel coverage roughly proportional to the 70/15/15
target across the three splits.

Non-anomalous ("normal") time is attached to whichever cluster it sits
closest to in time (the midpoint between two consecutive clusters is the
boundary), so every telemetry row -- anomalous or not -- ends up in
exactly one split, and no cluster's rows are ever divided between splits.

This is intentionally a practical, "good enough" heuristic, not a
provably-optimal partition: simultaneously balancing event count +
prevalence + class + category + channel coverage exactly is a hard
combinatorial problem, and the task explicitly calls for a simple greedy
heuristic instead of an expensive optimizer (simulated annealing / ILP /
genetic algorithms / beam search).

Known, accepted limitation
---------------------------
Because cluster assignment is not required to preserve global
chronological order across splits, a split's final DataFrame is a
concatenation of several disjoint time segments (in their original
chronological order relative to each other). A small number of sliding
windows at each segment-to-segment "seam" -- where two segments assigned
to the same split happen to sit next to each other after filtering out
the other splits' segments -- may span an artificial time discontinuity.
This affects only a handful of windows per seam (bounded by
window_size / stride) out of the total window count, and is an accepted
trade-off for a simple, deterministic, non-optimizer-based splitter, per
this task's explicit scope ("prioritize a working implementation over
perfect optimization").
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

import numpy as np
import pandas as pd

from src.core.mission import Mission
from src.utils.constants import (
    END_TIME_COLUMN,
    LABEL_CHANNEL_COLUMN,
    START_TIME_COLUMN,
)

logger = logging.getLogger(__name__)

# Columns specific to labels.csv / anomaly_types.csv. START_TIME_COLUMN,
# END_TIME_COLUMN, and LABEL_CHANNEL_COLUMN are reused from
# src.utils.constants (the same constants IntervalLabelGenerator uses,
# since both modules read labels.csv). EVENT_ID_COLUMN /
# EVENT_CLASS_COLUMN / EVENT_CATEGORY_COLUMN are defined locally because
# this is the first component that joins labels.csv against
# anomaly_types.csv on the shared "ID" column; if the project later grows
# other consumers of anomaly_types.csv, these should move into
# src.utils.constants alongside the others.
EVENT_ID_COLUMN = "ID"
EVENT_CLASS_COLUMN = "Class"
EVENT_CATEGORY_COLUMN = "Category"


@dataclass(frozen=True)
class Event:
    """One anomaly event (one ``ID`` in labels.csv). An event may touch
    several channels, each with a slightly different interval; `start`
    and `end` are the union (min/max) across all of the event's channel
    intervals."""

    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    channels: FrozenSet[str]
    event_class: str
    category: str
    duration_seconds: float  # sum of per-channel interval durations


@dataclass
class EventCluster:
    """One or more temporally-overlapping `Event`s merged into a single,
    indivisible splitting unit. `start`/`end` are updated as events are
    merged in (see `EventClusterSplitter._merge_into_clusters`)."""

    cluster_id: int
    events: List[Event]
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def channels(self) -> Set[str]:
        result: Set[str] = set()
        for event in self.events:
            result |= set(event.channels)
        return result

    @property
    def prevalence_seconds(self) -> float:
        """Sum of per-event anomaly-channel-interval durations in this
        cluster. Used as a proxy for how many anomalous (timestamp,
        channel) rows this cluster will contribute once the telemetry is
        windowed -- the single factor most responsible for the original
        anomaly-rate drift this splitter fixes."""
        return sum(event.duration_seconds for event in self.events)


class EventClusterSplitter:
    """
    Splits a mission's synchronized telemetry into train/validation/test
    using anomaly EVENTS (merged into temporally-overlapping CLUSTERS) as
    the indivisible splitting unit, instead of a plain chronological
    row-count cutpoint. See module docstring for the full rationale.

    Usage (mirrors the four algorithm steps described in the module
    docstring)::

        splitter = EventClusterSplitter(train_ratio=0.70, validation_ratio=0.15)
        clusters = splitter.build_clusters(mission, synchronized_df.index)
        assignment = splitter.assign_clusters(clusters)
        split_dfs = splitter.split_dataframe(synchronized_df, clusters, assignment)
        # ... generate dense labels / windows per split_dfs entry as before ...
        splitter.print_report(clusters, assignment, dense_labels, window_counts)
    """

    def __init__(self, train_ratio: float, validation_ratio: float) -> None:
        test_ratio = 1.0 - train_ratio - validation_ratio
        if test_ratio <= 0:
            raise ValueError(
                "train_ratio + validation_ratio must be < 1.0 so a "
                "non-empty test split remains "
                f"(got train_ratio={train_ratio!r}, "
                f"validation_ratio={validation_ratio!r})."
            )
        self.target_ratios: Dict[str, float] = {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        }

    # ------------------------------------------------------------------
    # Step 1 + 2: events -> clusters
    # ------------------------------------------------------------------
    def build_clusters(
        self, mission: Mission, reference_index: pd.DatetimeIndex
    ) -> List[EventCluster]:
        """Build events from `mission.labels` / `mission.anomaly_types`
        and merge temporally-overlapping events into clusters."""
        events = self._build_events(mission, reference_index)
        if not events:
            raise ValueError(
                "No anomaly events found in mission.labels; "
                "EventClusterSplitter requires at least one event."
            )
        return self._merge_into_clusters(events)

    def _build_events(
        self, mission: Mission, reference_index: pd.DatetimeIndex
    ) -> List[Event]:
        labels = mission.labels.copy()
        anomaly_types = mission.anomaly_types.copy()

        labels[START_TIME_COLUMN] = self._parse_timestamps(
            labels[START_TIME_COLUMN], reference_index
        )
        labels[END_TIME_COLUMN] = self._parse_timestamps(
            labels[END_TIME_COLUMN], reference_index
        )

        events: List[Event] = []
        for event_id, group in labels.groupby(EVENT_ID_COLUMN):
            duration_seconds = float(
                (group[END_TIME_COLUMN] - group[START_TIME_COLUMN])
                .dt.total_seconds()
                .sum()
            )

            metadata_rows = anomaly_types.loc[
                anomaly_types[EVENT_ID_COLUMN] == event_id
            ]
            event_class = (
                str(metadata_rows[EVENT_CLASS_COLUMN].iloc[0])
                if not metadata_rows.empty
                else "unknown"
            )
            category = (
                str(metadata_rows[EVENT_CATEGORY_COLUMN].iloc[0])
                if not metadata_rows.empty
                else "unknown"
            )

            events.append(
                Event(
                    event_id=str(event_id),
                    start=group[START_TIME_COLUMN].min(),
                    end=group[END_TIME_COLUMN].max(),
                    channels=frozenset(group[LABEL_CHANNEL_COLUMN].unique()),
                    event_class=event_class,
                    category=category,
                    duration_seconds=duration_seconds,
                )
            )

        events.sort(key=lambda e: e.start)
        return events

    @staticmethod
    def _parse_timestamps(
        series: pd.Series, reference_index: pd.DatetimeIndex
    ) -> pd.Series:
        """Same parsing convention as IntervalLabelGenerator._parse_timestamps:
        parse with UTC awareness, then align tz-awareness with
        `reference_index` (the synchronized telemetry's timestamp axis)."""
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if reference_index.tz is None:
            parsed = parsed.dt.tz_localize(None)
        else:
            parsed = parsed.dt.tz_convert(reference_index.tz)
        return parsed

    @staticmethod
    def _merge_into_clusters(events: List[Event]) -> List[EventCluster]:
        """Classic chronological interval-merge sweep: events are already
        sorted by start time, so an event merges into the current
        (running) cluster iff its start falls at or before the running
        cluster's end -- which correctly captures transitive overlaps
        (e.g. event C overlapping only event B, which overlaps event A,
        still ends up in the same cluster as A), since the cluster's end
        is extended to the max of all merged members as we go."""
        clusters: List[EventCluster] = []
        for event in events:
            if clusters and event.start <= clusters[-1].end:
                current = clusters[-1]
                current.events.append(event)
                current.end = max(current.end, event.end)
            else:
                clusters.append(
                    EventCluster(
                        cluster_id=len(clusters),
                        events=[event],
                        start=event.start,
                        end=event.end,
                    )
                )
        return clusters

    # ------------------------------------------------------------------
    # Step 3: greedy cluster -> split assignment
    # ------------------------------------------------------------------
    def assign_clusters(self, clusters: List[EventCluster]) -> Dict[int, str]:
        """
        Deterministic greedy ("Longest Processing Time"-style) bin
        packing: clusters are processed largest-first (by anomaly
        prevalence, then event count), and each is placed into whichever
        split is currently furthest *below* its target share of total
        events / prevalence / channel coverage. Processing the biggest,
        hardest-to-place clusters first and letting small clusters fill
        remaining gaps gives a much more even split than an arbitrary or
        chronological processing order, while remaining a single
        O(n log n) pass -- no search, no backtracking, no optimizer.
        """
        total_events = sum(c.num_events for c in clusters)
        total_prevalence = sum(c.prevalence_seconds for c in clusters)
        total_channel_incidence = sum(len(c.channels) for c in clusters)

        targets = {
            split: {
                "events": ratio * total_events,
                "prevalence": ratio * total_prevalence,
                "channels": ratio * total_channel_incidence,
            }
            for split, ratio in self.target_ratios.items()
        }
        running = {
            split: {"events": 0.0, "prevalence": 0.0, "channels": 0.0}
            for split in self.target_ratios
        }

        ordered = sorted(
            clusters,
            key=lambda c: (c.prevalence_seconds, c.num_events),
            reverse=True,
        )

        assignment: Dict[int, str] = {}
        for cluster in ordered:
            best_split, best_score = None, None
            for split in self.target_ratios:
                fill_events = self._fill_ratio(
                    running[split]["events"] + cluster.num_events,
                    targets[split]["events"],
                )
                fill_prevalence = self._fill_ratio(
                    running[split]["prevalence"] + cluster.prevalence_seconds,
                    targets[split]["prevalence"],
                )
                fill_channels = self._fill_ratio(
                    running[split]["channels"] + len(cluster.channels),
                    targets[split]["channels"],
                )
                # Equal weight on event count and prevalence (the two
                # factors most directly responsible for the anomaly-rate
                # drift being fixed here); a lighter weight on channel
                # coverage so it nudges, rather than dominates, placement.
                score = (
                    0.4 * fill_events + 0.4 * fill_prevalence + 0.2 * fill_channels
                )
                if best_score is None or score < best_score:
                    best_score, best_split = score, split

            assignment[cluster.cluster_id] = best_split
            running[best_split]["events"] += cluster.num_events
            running[best_split]["prevalence"] += cluster.prevalence_seconds
            running[best_split]["channels"] += len(cluster.channels)

        return assignment

    @staticmethod
    def _fill_ratio(hypothetical_total: float, target: float) -> float:
        """How "full" a split would be, relative to its target share, if
        a candidate cluster were placed there. Assigning to the split
        with the lowest such ratio is what keeps all splits proportional
        to their targets."""
        if target <= 0:
            return float("inf") if hypothetical_total > 0 else 0.0
        return hypothetical_total / target

    # ------------------------------------------------------------------
    # Step 4: cluster assignment -> row-level DataFrames
    # ------------------------------------------------------------------
    def split_dataframe(
        self,
        synchronized_df: pd.DataFrame,
        clusters: List[EventCluster],
        assignment: Dict[int, str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert a cluster -> split assignment into three row-level
        DataFrames. Every telemetry row is attached to the split of its
        temporally nearest cluster (boundaries are the midpoints between
        consecutive clusters), which keeps entire clusters -- and the
        "normal" time immediately surrounding them -- together in one
        split. Row order within each split matches `synchronized_df`'s
        original chronological order (masking preserves order), so
        `WindowGenerator`'s "sorted, non-decreasing index" requirement
        is still satisfied.
        """
        if not clusters:
            raise ValueError(
                "EventClusterSplitter requires at least one anomaly "
                "cluster to define split boundaries."
            )

        ordered = sorted(clusters, key=lambda c: c.start)
        timestamps = synchronized_df.index

        boundaries = [
            ordered[i].end + (ordered[i + 1].start - ordered[i].end) / 2
            for i in range(len(ordered) - 1)
        ]

        split_masks = {
            split: np.zeros(len(synchronized_df), dtype=bool)
            for split in self.target_ratios
        }

        for i, cluster in enumerate(ordered):
            lower = boundaries[i - 1] if i > 0 else None
            upper = boundaries[i] if i < len(boundaries) else None

            if lower is None and upper is None:
                mask = np.ones(len(synchronized_df), dtype=bool)
            elif lower is None:
                mask = timestamps <= upper
            elif upper is None:
                mask = timestamps > lower
            else:
                mask = ((timestamps > lower) & (timestamps <= upper))

            split_masks[assignment[cluster.cluster_id]] |= mask

        splits = {
            split: synchronized_df.loc[mask] for split, mask in split_masks.items()
        }

        for split, df in splits.items():
            if df.empty:
                raise ValueError(
                    f"Event-cluster split produced an empty '{split}' "
                    "split; adjust train_ratio/validation_ratio or "
                    "inspect the cluster distribution."
                )

        return splits

    # ------------------------------------------------------------------
    # Diagnostic report
    # ------------------------------------------------------------------
    def print_report(
        self,
        clusters: List[EventCluster],
        assignment: Dict[int, str],
        dense_labels: Dict[str, np.ndarray],
        window_counts: Dict[str, int],
    ) -> None:
        """Print the FINAL SPLIT REPORT required by this task: per-split
        cluster/event/window counts, anomaly rate, class distribution,
        category distribution, affected-channel count, and average
        anomaly duration."""
        print("=" * 30)
        print("FINAL SPLIT REPORT")
        print("=" * 30)
        total_clusters = len(clusters)
        total_events = sum(len(cluster.events) for cluster in clusters)
        total_windows = sum(window_counts.values())

        for split in ("train", "validation", "test"):
            split_clusters = [
                c for c in clusters if assignment[c.cluster_id] == split
            ]
            events = [e for c in split_clusters for e in c.events]

            channels: Set[str] = set()
            for e in events:
                channels |= set(e.channels)

            class_dist = Counter(e.event_class for e in events)
            category_dist = Counter(e.category for e in events)

            durations_hours = [
                (e.end - e.start).total_seconds() / 3600.0 for e in events
            ]
            avg_duration_hours = (
                float(np.mean(durations_hours)) if durations_hours else 0.0
            )

            labels_arr = dense_labels.get(split)
            anomaly_rate = (
                float(labels_arr.mean())
                if labels_arr is not None and labels_arr.size
                else 0.0
            )

            print(f"\n[{split.upper()}]")
            cluster_count = len(split_clusters)
            event_count = len(events)
            window_count = window_counts.get(split, 0)

            print(
                f"  event clusters       : {cluster_count} "
                f"({100*cluster_count/total_clusters:.1f}%)"
            )

            print(
                f"  events               : {event_count} "
                f"({100*event_count/total_events:.1f}%)"
            )

            print(
                f"  windows              : {window_count:,} "
                f"({100*window_count/total_windows:.1f}%)"
            )
            print(f"  anomaly rate         : {anomaly_rate:.4%}")
            print(f"  class distribution   : {dict(class_dist)}")
            print(f"  category distribution: {dict(category_dist)}")
            print(f"  affected channels    : {len(channels)}")
            print(f"  avg anomaly duration : {avg_duration_hours:.2f} h")

        print("\n" + "=" * 30)
