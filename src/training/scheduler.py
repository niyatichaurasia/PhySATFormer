"""Learning rate scheduler factory for PhySATFormer.

This module provides a small, reusable factory for constructing
``torch.optim.lr_scheduler.CosineAnnealingLR`` schedulers with
validated configuration. It contains no training-loop logic; it is
strictly responsible for scheduler construction.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


class SchedulerFactory:
    """Factory for constructing a configured ``CosineAnnealingLR`` scheduler.

    This class encapsulates the configuration and validation logic
    required to build a ``torch.optim.lr_scheduler.CosineAnnealingLR``
    instance for a given optimizer. It performs no training, no
    optimizer construction, and no state management beyond holding its
    own configuration.

    Parameters
    ----------
    T_max : int, optional
        Maximum number of iterations (or epochs, depending on how the
        scheduler's ``step`` method is invoked) for one cosine
        annealing half-cycle, by default 50. Must be a positive
        integer.
    eta_min : float, optional
        Minimum learning rate reached at the trough of the cosine
        annealing schedule, by default 0.0. Must be non-negative.

    Attributes
    ----------
    T_max : int
        Maximum number of iterations for one cosine annealing
        half-cycle.
    eta_min : float
        Minimum learning rate reached at the trough of the schedule.

    Examples
    --------
    >>> import torch
    >>> model = torch.nn.Linear(4, 2)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> factory = SchedulerFactory(T_max=100, eta_min=1e-6)
    >>> scheduler = factory.create_scheduler(optimizer)
    >>> isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    True
    """

    def __init__(
        self,
        T_max: int = 50,
        eta_min: float = 0.0,
    ) -> None:
        self._validate_t_max(T_max)
        self._validate_eta_min(eta_min)

        self.T_max = T_max
        self.eta_min = eta_min

    @staticmethod
    def _validate_t_max(T_max: int) -> None:
        """Validate the ``T_max`` configuration value.

        Parameters
        ----------
        T_max : int
            Candidate maximum number of iterations for one cosine
            annealing half-cycle.

        Raises
        ------
        TypeError
            If `T_max` is not an ``int`` (booleans are rejected).
        ValueError
            If `T_max` is not positive.
        """
        if not isinstance(T_max, int) or isinstance(T_max, bool):
            raise TypeError(
                f"`T_max` must be an int, got {type(T_max).__name__}."
            )
        if T_max <= 0:
            raise ValueError(f"`T_max` must be positive, got {T_max}.")

    @staticmethod
    def _validate_eta_min(eta_min: float) -> None:
        """Validate the ``eta_min`` configuration value.

        Parameters
        ----------
        eta_min : float
            Candidate minimum learning rate.

        Raises
        ------
        TypeError
            If `eta_min` is not a real number (``int`` or ``float``;
            booleans are rejected).
        ValueError
            If `eta_min` is negative.
        """
        if not isinstance(eta_min, (int, float)) or isinstance(eta_min, bool):
            raise TypeError(
                f"`eta_min` must be a float, got {type(eta_min).__name__}."
            )
        if eta_min < 0:
            raise ValueError(f"`eta_min` must be non-negative, got {eta_min}.")

    @staticmethod
    def _validate_optimizer(optimizer: torch.optim.Optimizer) -> None:
        """Validate that `optimizer` is a ``torch.optim.Optimizer``.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Candidate optimizer instance.

        Raises
        ------
        TypeError
            If `optimizer` is not an instance of
            ``torch.optim.Optimizer``.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "`optimizer` must be a torch.optim.Optimizer, got "
                f"{type(optimizer).__name__}."
            )

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> CosineAnnealingLR:
        """Create a configured ``CosineAnnealingLR`` scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer instance whose learning rate will be scheduled.

        Returns
        -------
        torch.optim.lr_scheduler.CosineAnnealingLR
            A cosine annealing learning rate scheduler configured with
            this factory's ``T_max`` and ``eta_min``, bound to
            `optimizer`.

        Raises
        ------
        TypeError
            If `optimizer` is not a ``torch.optim.Optimizer``.
        """
        self._validate_optimizer(optimizer)

        return CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
        )