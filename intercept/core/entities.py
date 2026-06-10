"""Engagement entities (interceptors, targets, decoys) and their control interface.

An :class:`Entity` couples a dynamics model with a *controller* — a callable
``controller(t, own_state, world) -> control``. Guidance laws, scripted evaders, RL policies,
and cooperative allocators all conform to this single signature, so the engagement loop never
needs to know which paradigm is driving an entity. ``world`` is a read-only snapshot mapping
entity name -> state, giving controllers access to (estimated, later) opponent states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from intercept.core.dynamics import Dynamics

Array = NDArray[np.float64]

#: A controller maps (time, own state, world snapshot) to a control vector.
Controller = Callable[[float, Array, Mapping[str, Array]], Array]


def zero_controller(dim: int) -> Controller:
    """Return a controller that always commands zero control (coasting)."""
    z = np.zeros(dim, dtype=float)

    def _c(t: float, own_state: Array, world: Mapping[str, Array]) -> Array:
        return z.copy()

    return _c


@dataclass
class Entity:
    """A simulated body with its own dynamics, state, and controller.

    Parameters
    ----------
    name:
        Unique identifier used as the key in the ``world`` snapshot.
    dynamics:
        The continuous-time plant for this entity.
    state:
        Initial state (copied on construction).
    controller:
        Callable producing the control each step. Defaults to a zero (coasting) controller.
    role:
        Free-form tag (e.g. ``"interceptor"``, ``"target"``, ``"decoy"``) for bookkeeping/plots.
    """

    name: str
    dynamics: Dynamics
    state: Array
    controller: Controller | None = None
    role: str = "entity"
    _last_control: Array = field(init=False, repr=False, default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        self.state = np.array(self.state, dtype=float).copy()
        if self.state.shape != (self.dynamics.state_dim,):
            raise ValueError(
                f"Entity '{self.name}': state has shape {self.state.shape}, "
                f"expected ({self.dynamics.state_dim},)"
            )
        if self.controller is None:
            self.controller = zero_controller(self.dynamics.control_dim)
        self._last_control = np.zeros(self.dynamics.control_dim, dtype=float)

    @property
    def position(self) -> Array:
        return self.dynamics.position(self.state)

    @property
    def velocity(self) -> Array:
        return self.dynamics.velocity(self.state)

    @property
    def last_control(self) -> Array:
        """The control commanded on the most recent step (for logging effort metrics)."""
        return self._last_control

    def compute_control(self, t: float, world: Mapping[str, Array]) -> Array:
        assert self.controller is not None  # set in __post_init__
        u = np.asarray(self.controller(t, self.state, world), dtype=float)
        if u.shape != (self.dynamics.control_dim,):
            raise ValueError(
                f"Entity '{self.name}': controller returned shape {u.shape}, "
                f"expected ({self.dynamics.control_dim},)"
            )
        self._last_control = u
        return u
