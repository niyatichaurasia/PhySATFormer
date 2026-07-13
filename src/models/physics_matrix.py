# src/models/physics_matrix.py

import torch

from src.core.mission import Mission


class PhysicsRelationshipMatrix:
    """
    Builds a static physics-relationship matrix between telemetry channels
    based purely on mission metadata (subsystem membership).

    Rule:
        - matrix[i][j] = 1.0 if channel i and channel j belong to the same
          subsystem (including i == j)
        - matrix[i][j] = 0.0 otherwise
        - Diagonal is always 1.0

    This class never loads telemetry data; it only reads mission metadata,
    which is exposed as a pandas DataFrame via `mission.channels`.
    """

    def __init__(self, mission: Mission):
        self.mission = mission

    def _get_channel_subsystems(self):
        """
        Extract the subsystem label for each channel from the metadata
        DataFrame exposed by the Mission object.
        """
        channels = self.mission.channels
        if channels.empty:
            raise ValueError("Mission contains no channel metadata.")

        if "subsystem" not in channels.columns:
            raise KeyError(
                "Mission.channels DataFrame does not contain a "
                "'subsystem' column."
            )

        return channels["subsystem"].tolist()

    def build(self) -> torch.FloatTensor:
        subsystems = self._get_channel_subsystems()
        num_channels = len(subsystems)

        matrix = torch.zeros((num_channels, num_channels), dtype=torch.float32)

        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    matrix[i, j] = 1.0
                elif (
                    subsystems[i] == subsystems[j]
                    and subsystems[i] is not None
                ):
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = 0.0

        return matrix
    
    @property
    def num_channels(self) -> int:
        """Return the number of telemetry channels."""
        return len(self.mission.channels)