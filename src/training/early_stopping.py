"""Early stopping utility for PhySATFormer training.

This module implements a single, reusable utility class,
:class:`EarlyStopping`, that monitors a validation metric across epochs
and decides whether training should be halted.

This module intentionally contains no training-loop logic (no
optimizer stepping, no checkpointing, no data loading, no model
references). It is a pure decision-making utility: given a stream of
scalar validation scores fed to it via repeated calls to
:meth:`EarlyStopping.update`, it tracks the best score seen so far and
signals, via :attr:`EarlyStopping.should_stop`, when training should
be terminated because the monitored metric has stopped improving.
"""

from __future__ import annotations

import numbers


class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    On each call to :meth:`update`, the current score is compared
    against the best score observed so far. A score is considered an
    improvement if it exceeds (for ``mode="max"``) or falls below (for
    ``mode="min"``) the best score by more than `min_delta`. If the
    score improves, the internal patience counter is reset to zero and
    the new score becomes the best score. Otherwise, the counter is
    incremented. Once the counter exceeds `patience`, :attr:`should_stop`
    becomes ``True`` and remains ``True`` for the lifetime of the
    instance (until :meth:`reset` is called, if ever).

    Parameters
    ----------
    patience : int, optional
        Number of consecutive non-improving updates to tolerate before
        signaling that training should stop, by default 10. Must be
        positive.
    min_delta : float, optional
        Minimum change in the monitored score required to qualify as
        an improvement, by default 0.0. Must be non-negative.
    mode : str, optional
        One of ``"max"`` or ``"min"``. Use ``"max"`` when higher scores
        are better (e.g. F1-score, accuracy, AUROC) and ``"min"`` when
        lower scores are better (e.g. validation loss). Defaults to
        ``"max"``, since PhySATFormer monitors validation F1-score.

    Attributes
    ----------
    patience : int
        Number of consecutive non-improving updates tolerated.
    min_delta : float
        Minimum change required to qualify as an improvement.
    mode : str
        Either ``"max"`` or ``"min"``.

    Examples
    --------
    >>> stopper = EarlyStopping(patience=2, min_delta=0.0, mode="max")
    >>> stopper.update(0.70)
    True
    >>> stopper.update(0.71)
    True
    >>> stopper.update(0.70)
    False
    >>> stopper.update(0.69)
    False
    >>> stopper.should_stop
    True
    >>> stopper.best_score
    0.71
    """

    _VALID_MODES = ("max", "min")

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self._validate_patience(patience)
        self._validate_min_delta(min_delta)
        self._validate_mode(mode)

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best_score: float | None = None
        self._counter: int = 0
        self._should_stop: bool = False

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_patience(patience: int) -> None:
        """Validate the `patience` constructor argument.

        Parameters
        ----------
        patience : int
            Candidate patience value.

        Raises
        ------
        TypeError
            If `patience` is not an ``int``.
        ValueError
            If `patience` is not positive.
        """
        if not isinstance(patience, int) or isinstance(patience, bool):
            raise TypeError(
                f"`patience` must be an int, got {type(patience).__name__}."
            )
        if patience <= 0:
            raise ValueError(f"`patience` must be positive, got {patience}.")

    @staticmethod
    def _validate_min_delta(min_delta: float) -> None:
        """Validate the `min_delta` constructor argument.

        Parameters
        ----------
        min_delta : float
            Candidate min_delta value.

        Raises
        ------
        TypeError
            If `min_delta` is not numeric.
        ValueError
            If `min_delta` is negative.
        """
        if not isinstance(min_delta, numbers.Real) or isinstance(min_delta, bool):
            raise TypeError(
                f"`min_delta` must be numeric, got {type(min_delta).__name__}."
            )
        if min_delta < 0.0:
            raise ValueError(f"`min_delta` must be non-negative, got {min_delta}.")

    @classmethod
    def _validate_mode(cls, mode: str) -> None:
        """Validate the `mode` constructor argument.

        Parameters
        ----------
        mode : str
            Candidate mode value.

        Raises
        ------
        TypeError
            If `mode` is not a ``str``.
        ValueError
            If `mode` is not one of ``"max"`` or ``"min"``.
        """
        if not isinstance(mode, str):
            raise TypeError(f"`mode` must be a str, got {type(mode).__name__}.")
        if mode not in cls._VALID_MODES:
            raise ValueError(
                f"`mode` must be one of {cls._VALID_MODES!r}, got {mode!r}."
            )

    @staticmethod
    def _validate_score(score: float) -> None:
        """Validate a score passed to :meth:`update`.

        Parameters
        ----------
        score : float
            Candidate score value.

        Raises
        ------
        TypeError
            If `score` is not numeric.
        ValueError
            If `score` is NaN or infinite.
        """
        if not isinstance(score, numbers.Real) or isinstance(score, bool):
            raise TypeError(
                f"`score` must be numeric, got {type(score).__name__}."
            )
        if score != score:  # NaN check without requiring math/numpy import.
            raise ValueError("`score` must not be NaN.")
        if score in (float("inf"), float("-inf")):
            raise ValueError("`score` must be finite, got an infinite value.")

    # ------------------------------------------------------------------
    # Internal comparison logic
    # ------------------------------------------------------------------
    def _is_improvement(self, score: float) -> bool:
        """Determine whether `score` improves on the current best score.

        Parameters
        ----------
        score : float
            Candidate score to compare against :attr:`best_score`.

        Returns
        -------
        bool
            ``True`` if `score` is an improvement (including the very
            first score ever seen), ``False`` otherwise.
        """
        if self._best_score is None:
            return True

        if self.mode == "max":
            return score > self._best_score + self.min_delta

        return score < self._best_score - self.min_delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, score: float) -> bool:
        """Update internal state with a new validation score.

        Parameters
        ----------
        score : float
            The latest monitored validation metric value (e.g. this
            epoch's validation F1-score).

        Returns
        -------
        bool
            ``True`` if `score` is the new best score observed so far,
            in which case the internal patience counter is reset.
            ``False`` if `score` did not improve on the best score, in
            which case the patience counter is incremented and
            :attr:`should_stop` may become ``True``.

        Raises
        ------
        TypeError
            If `score` is not numeric.
        ValueError
            If `score` is NaN or infinite.
        """
        self._validate_score(score)

        if self._is_improvement(score):
            self._best_score = float(score)
            self._counter = 0
            return True

        self._counter += 1
        if self._counter > self.patience:
            self._should_stop = True
        return False

    def reset(self) -> None:
        """Reset all internal state to its initial (pre-training) values.

        After calling this method, :attr:`best_score` is ``None``,
        :attr:`should_stop` is ``False``, and the patience counter is
        zero, as if no calls to :meth:`update` had ever been made.
        """
        self._best_score = None
        self._counter = 0
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        """bool: Whether training should stop, based on scores seen so far."""
        return self._should_stop

    @property
    def best_score(self) -> float | None:
        """float or None: The best score observed so far, or ``None`` if
        :meth:`update` has not yet been called."""
        return self._best_score

    @property
    def counter(self) -> int:
        """int: Number of consecutive non-improving updates so far."""
        return self._counter

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(patience={self.patience!r}, "
            f"min_delta={self.min_delta!r}, mode={self.mode!r}, "
            f"best_score={self._best_score!r}, counter={self._counter!r}, "
            f"should_stop={self._should_stop!r})"
        )