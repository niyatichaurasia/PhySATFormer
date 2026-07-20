"""AdamW optimizer factory for PhySATFormer.

This module implements a production-quality optimizer factory that
constructs a :class:`torch.optim.AdamW` optimizer for any PhySATFormer
model variant (or, indeed, any ``nn.Module``), following standard
Transformer optimization practice: weight decay is applied to
"weight-like" matrix parameters (linear/projection/attention/
feed-forward weights) but is *not* applied to biases or to
LayerNorm parameters (weight and bias), since decaying those tends to
destabilize training without a corresponding generalization benefit.

The parameter-grouping logic is module-aware rather than name-based:
it walks the model's module tree and classifies each parameter
according to the type of module that owns it (e.g. any parameter
belonging to an ``nn.LayerNorm`` is exempted from weight decay,
regardless of what the parameter happens to be named). This makes the
grouping robust to naming variations across the different attention
and encoder-block implementations used throughout the project
(``MultiHeadSelfAttention``, ``TemporalAttention``,
``PhysicsGuidedChannelAttention``, etc.), all of which are built from
plain ``nn.Linear`` and ``nn.LayerNorm`` submodules.

This module contains no training-loop logic; it is strictly
responsible for constructing a correctly-configured optimizer instance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class OptimizerFactory:
    """Factory for constructing an ``AdamW`` optimizer for PhySATFormer.

    Parameters passed to the underlying model are split into two
    non-overlapping groups:

    - **Decay group**: parameters that behave like dense weight
      matrices, i.e. the ``weight`` parameter of ``nn.Linear`` (and any
      subclass thereof) layers, covering linear layers, projection
      layers, attention projections (query/key/value/output), and
      feed-forward layers. Weight decay is applied to this group.
    - **No-decay group**: all bias parameters (regardless of owning
      module type), plus both the ``weight`` and ``bias`` parameters of
      normalization layers (``nn.LayerNorm``, ``nn.BatchNorm*``,
      ``nn.GroupNorm``), plus any other parameter with fewer than 2
      dimensions (e.g. learnable per-head scalar weights such as
      ``PhysicsGuidedChannelAttention.physics_weight``). Weight decay
      is *not* applied to this group.

    Parameters
    ----------
    learning_rate : float, optional
        The base learning rate used by AdamW, by default ``1e-3``.
        Must be strictly positive.
    weight_decay : float, optional
        The weight decay (L2 penalty) coefficient applied to the decay
        parameter group, by default ``0.01``. Must be non-negative.
    betas : Tuple[float, float], optional
        Coefficients used for computing running averages of the
        gradient and its square, by default ``(0.9, 0.999)``. Must be
        a length-2 tuple of floats.
    eps : float, optional
        Term added to the denominator to improve numerical stability,
        by default ``1e-8``. Must be strictly positive.

    Attributes
    ----------
    learning_rate : float
        The configured base learning rate.
    weight_decay : float
        The configured weight decay coefficient.
    betas : Tuple[float, float]
        The configured Adam beta coefficients.
    eps : float
        The configured numerical-stability epsilon.

    Examples
    --------
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
    >>> factory = OptimizerFactory(learning_rate=3e-4, weight_decay=0.05)
    >>> optimizer = factory.create_optimizer(model)
    >>> isinstance(optimizer, torch.optim.AdamW)
    True
    """

    # Module types whose parameters are always exempted from weight
    # decay, regardless of the parameter's name.
    _NO_DECAY_MODULE_TYPES: Tuple[type, ...] = (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    # Module types whose "weight" parameter is treated as a dense
    # weight matrix eligible for weight decay.
    _DECAY_MODULE_TYPES: Tuple[type, ...] = (nn.Linear,)

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self._validate_learning_rate(learning_rate)
        self._validate_weight_decay(weight_decay)
        self._validate_betas(betas)
        self._validate_eps(eps)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_learning_rate(learning_rate: float) -> None:
        """Validate the learning rate.

        Raises
        ------
        TypeError
            If `learning_rate` is not a real number.
        ValueError
            If `learning_rate` is not strictly positive.
        """
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise TypeError(
                "`learning_rate` must be a float, got "
                f"{type(learning_rate).__name__}."
            )
        if learning_rate <= 0:
            raise ValueError(
                f"`learning_rate` must be positive, got {learning_rate}."
            )

    @staticmethod
    def _validate_weight_decay(weight_decay: float) -> None:
        """Validate the weight decay coefficient.

        Raises
        ------
        TypeError
            If `weight_decay` is not a real number.
        ValueError
            If `weight_decay` is negative.
        """
        if isinstance(weight_decay, bool) or not isinstance(
            weight_decay, (int, float)
        ):
            raise TypeError(
                "`weight_decay` must be a float, got "
                f"{type(weight_decay).__name__}."
            )
        if weight_decay < 0:
            raise ValueError(
                f"`weight_decay` must be non-negative, got {weight_decay}."
            )

    @staticmethod
    def _validate_betas(betas: Tuple[float, float]) -> None:
        """Validate the Adam beta coefficients.

        Raises
        ------
        TypeError
            If `betas` is not a tuple, or contains non-numeric values.
        ValueError
            If `betas` does not have exactly two elements, or if either
            element is not in the range ``[0, 1)``.
        """
        if not isinstance(betas, tuple):
            raise TypeError(f"`betas` must be a tuple, got {type(betas).__name__}.")
        if len(betas) != 2:
            raise ValueError(
                f"`betas` must be a tuple of length 2, got length {len(betas)}."
            )
        for beta in betas:
            if isinstance(beta, bool) or not isinstance(beta, (int, float)):
                raise TypeError(
                    f"Elements of `betas` must be floats, got {type(beta).__name__}."
                )
            if not 0.0 <= beta < 1.0:
                raise ValueError(
                    f"Elements of `betas` must be in [0, 1), got {beta}."
                )

    @staticmethod
    def _validate_eps(eps: float) -> None:
        """Validate the numerical-stability epsilon.

        Raises
        ------
        TypeError
            If `eps` is not a real number.
        ValueError
            If `eps` is not strictly positive.
        """
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(f"`eps` must be a float, got {type(eps).__name__}.")
        if eps <= 0:
            raise ValueError(f"`eps` must be positive, got {eps}.")

    @staticmethod
    def _validate_model(model: nn.Module) -> None:
        """Validate that `model` is an ``nn.Module``.

        Raises
        ------
        TypeError
            If `model` is not an instance of ``torch.nn.Module``.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"`model` must be a torch.nn.Module, got {type(model).__name__}."
            )

    # ------------------------------------------------------------------
    # Parameter grouping
    # ------------------------------------------------------------------
    def _build_parameter_groups(
        self, model: nn.Module
    ) -> List[Dict[str, Any]]:
        """Split model parameters into decay / no-decay groups.

        Parameters are classified by walking the module tree: each
        parameter is attributed to the (innermost) module that directly
        owns it, and classified according to that module's type. This
        avoids relying on parameter-name conventions, which can vary
        across the different attention and encoder-block
        implementations in this project.

        Classification rules (evaluated in order):

        1. Any parameter belonging to a normalization module (see
           ``_NO_DECAY_MODULE_TYPES``) -> no decay.
        2. Any parameter named ``bias`` -> no decay.
        3. The ``weight`` parameter of a module in
           ``_DECAY_MODULE_TYPES`` (e.g. ``nn.Linear``) -> decay.
        4. Any remaining parameter with fewer than 2 dimensions (e.g.
           learnable scalar/vector weights such as
           ``physics_weight``) -> no decay.
        5. Anything else -> decay (conservative default for dense,
           multi-dimensional weight tensors from custom modules).

        Parameters
        ----------
        model : torch.nn.Module
            The model whose parameters will be grouped.

        Returns
        -------
        List[Dict[str, Any]]
            A list of two parameter-group dictionaries suitable for
            passing directly to ``torch.optim.AdamW``: one with
            ``weight_decay=self.weight_decay`` and one with
            ``weight_decay=0.0``. Empty groups (e.g. a model with no
            biases) are omitted.

        Raises
        ------
        ValueError
            If the model has no trainable parameters, or if a
            parameter is unexpectedly assigned to both groups (an
            internal consistency check).
        """
        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []
        seen_param_ids: set = set()

        for module in model.modules():
            is_no_decay_module = isinstance(module, self._NO_DECAY_MODULE_TYPES)
            is_decay_module = isinstance(module, self._DECAY_MODULE_TYPES)

            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                param_id = id(param)
                if param_id in seen_param_ids:
                    # Guards against double-counting parameters that are
                    # shared/tied across multiple modules.
                    continue
                seen_param_ids.add(param_id)

                if is_no_decay_module:
                    no_decay_params.append(param)
                elif param_name.endswith("bias"):
                    no_decay_params.append(param)
                elif is_decay_module and param_name == "weight":
                    decay_params.append(param)
                elif param.dim() < 2:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        if not decay_params and not no_decay_params:
            raise ValueError(
                "`model` has no trainable parameters; cannot construct an "
                "optimizer."
            )

        parameter_groups: List[Dict[str, Any]] = []
        if decay_params:
            parameter_groups.append(
                {"params": decay_params, "weight_decay": self.weight_decay}
            )
        if no_decay_params:
            parameter_groups.append(
                {"params": no_decay_params, "weight_decay": 0.0}
            )

        return parameter_groups

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_optimizer(self, model: nn.Module) -> torch.optim.AdamW:
        """Create an ``AdamW`` optimizer for the given model.

        Model parameters are split into weight-decay and no-weight-decay
        groups as described in :meth:`_build_parameter_groups`, and an
        ``AdamW`` optimizer is constructed over both groups using this
        factory's configured learning rate, betas, and epsilon.

        Parameters
        ----------
        model : torch.nn.Module
            The model to optimize. May be any ``nn.Module``, including
            ``PhySATFormer`` or ``BaselineTransformer``.

        Returns
        -------
        torch.optim.AdamW
            A fully initialized AdamW optimizer configured with
            per-group weight decay.

        Raises
        ------
        TypeError
            If `model` is not a ``torch.nn.Module``.
        ValueError
            If `model` has no trainable parameters.
        """
        self._validate_model(model)

        parameter_groups = self._build_parameter_groups(model)

        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
        )

        return optimizer