# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Double-integrator prediction model and the per-agent local QP.

This module owns the "x-update" of consensus ADMM: given the consensus trajectories
``z^j`` and the scaled duals ``lam_i^j`` handed down by :mod:`consensus_admm`, agent ``i``
solves a small strictly convex QP over its own inputs and its local copies of the
neighbor trajectories.

Local problem solved by agent ``i`` (see ``docs/derivations/consensus_admm_derivation.tex``)
-------------------------------------------------------------------------------------

    minimise    J_track(U_i, y_i^i) + J_input(U_i) + J_form({y_i^j})
                + (rho/2) * sum_{j in Ncl(i)} || y_i^j - z^j + lam_i^j ||_F^2

    subject to  y_i^i = Phi_p x_i(0) + Gamma_p U_i          (own dynamics, condensed)
                |u_i(t)| <= u_max                            (elementwise, all t)
                |v_i(t)| <= v_max                            (elementwise, all t)
                p_min <= p_i(t) <= p_max                     (optional workspace box)

Only ``y_i^i`` is dynamically constrained. The copies ``y_i^j`` for ``j != i`` are free
variables — the consensus penalty is the *only* thing that pins them down, which is what
makes the problem separable across agents.

Dimensions: ``U_i`` is ``(T, 2)``, every ``y_i^j`` is ``(T, 2)``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AgentCostWeights",
    "AgentLimits",
    "CvxpyAgentSolver",
    "DoubleIntegrator",
    "LocalProblemData",
    "LocalSolution",
    "PerAgentSolver",
]


@cache
def _condensed_prediction(
    dt: float, dim: int, horizon: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build ``(Phi, Gamma)`` for the discrete double integrator, cached on ``(dt, dim, horizon)``.

    ``Phi`` block row ``r`` (time ``t = r + 1``) is ``A^(r+1)``; ``Gamma`` block ``(r, s)``
    is ``A^(r-s) B`` for ``s <= r`` (and zero above the diagonal). ``A`` powers are built
    iteratively so the recursion is directly checkable against the forward rollout.
    """
    a = np.zeros((2 * dim, 2 * dim))
    a[0:dim, 0:dim] = np.eye(dim)
    a[0:dim, dim : 2 * dim] = dt * np.eye(dim)
    a[dim : 2 * dim, dim : 2 * dim] = np.eye(dim)

    b = np.zeros((2 * dim, dim))
    b[0:dim, :] = 0.5 * dt**2 * np.eye(dim)
    b[dim : 2 * dim, :] = dt * np.eye(dim)

    n = 2 * dim
    powers = [np.eye(n)]
    for _ in range(horizon):
        powers.append(a @ powers[-1])

    phi = np.vstack([powers[r + 1] for r in range(horizon)])
    gamma = np.zeros((horizon * n, horizon * dim))
    for r in range(horizon):
        for s in range(r + 1):
            gamma[r * n : (r + 1) * n, s * dim : (s + 1) * dim] = powers[r - s] @ b
    return phi, gamma


@dataclass(frozen=True)
class DoubleIntegrator:
    """Discrete-time double integrator in ``dim`` spatial dimensions.

    Continuous dynamics ``pdot = v``, ``vdot = u``, discretised exactly under a
    zero-order hold with sample time ``dt``::

        A = [[I, dt*I], [0, I]]        B = [[0.5*dt^2*I], [dt*I]]

    State ordering is ``[p_1..p_dim, v_1..v_dim]``.
    """

    dt: float
    dim: int = 2

    @property
    def n_states(self) -> int:
        """``2 * dim``."""
        return 2 * self.dim

    @property
    def n_inputs(self) -> int:
        """``dim``."""
        return self.dim

    def matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(A, B)`` with shapes ``(2*dim, 2*dim)`` and ``(2*dim, dim)``."""
        dim = self.dim
        a = np.zeros((2 * dim, 2 * dim))
        a[0:dim, 0:dim] = np.eye(dim)
        a[0:dim, dim : 2 * dim] = self.dt * np.eye(dim)
        a[dim : 2 * dim, dim : 2 * dim] = np.eye(dim)

        b = np.zeros((2 * dim, dim))
        b[0:dim, :] = 0.5 * self.dt**2 * np.eye(dim)
        b[dim : 2 * dim, :] = self.dt * np.eye(dim)
        return a, b

    def position_selector(self) -> NDArray[np.float64]:
        """``C_p`` with ``p = C_p x``; shape ``(dim, 2*dim)``."""
        selector = np.zeros((self.dim, 2 * self.dim))
        selector[:, : self.dim] = np.eye(self.dim)
        return selector

    def velocity_selector(self) -> NDArray[np.float64]:
        """``C_v`` with ``v = C_v x``; shape ``(dim, 2*dim)``."""
        selector = np.zeros((self.dim, 2 * self.dim))
        selector[:, self.dim :] = np.eye(self.dim)
        return selector

    def prediction_matrices(self, horizon: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Condensed prediction ``X = Phi x0 + Gamma U`` over ``t = 1..T``.

        Returns ``(Phi, Gamma)`` with shapes ``(T*2*dim, 2*dim)`` and
        ``(T*2*dim, T*dim)``. ``Gamma`` is block lower-triangular with
        ``Gamma[r, s] = A^{r-s} B`` for ``s <= r``.

        Note the horizon starts at ``t = 1``: ``x0`` itself is *not* part of ``X``.
        """
        return _condensed_prediction(self.dt, self.dim, horizon)

    def position_prediction_matrices(
        self, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rows of :meth:`prediction_matrices` that produce positions only.

        Returns ``(Phi_p, Gamma_p)`` with shapes ``(T*dim, 2*dim)`` and ``(T*dim, T*dim)``.
        This is the map used in the equality constraint ``y_i^i = Phi_p x0 + Gamma_p U_i``.
        Cache the result — it depends only on ``(dt, dim, horizon)``.
        """
        phi, gamma = _condensed_prediction(self.dt, self.dim, horizon)
        n = 2 * self.dim
        idx = np.concatenate([np.arange(t * n, t * n + self.dim) for t in range(horizon)])
        return phi[idx], gamma[idx]

    def velocity_prediction_matrices(
        self, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rows of :meth:`prediction_matrices` that produce velocities only.

        Returns ``(Phi_v, Gamma_v)`` with shapes ``(T*dim, 2*dim)`` and ``(T*dim, T*dim)``.
        This is the map used in the velocity box ``|V_i| <= v_max``.
        """
        phi, gamma = _condensed_prediction(self.dt, self.dim, horizon)
        n = 2 * self.dim
        idx = np.concatenate(
            [np.arange(t * n + self.dim, t * n + 2 * self.dim) for t in range(horizon)]
        )
        return phi[idx], gamma[idx]

    def simulate(self, x0: NDArray[np.float64], inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Roll the true plant forward. ``inputs`` is ``(T, dim)``; returns ``(T+1, 2*dim)``
        including ``x0`` as row 0."""
        a, b = self.matrices()
        x0 = np.asarray(x0, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        horizon = inputs.shape[0]
        states = np.zeros((horizon + 1, self.n_states))
        states[0] = x0
        for t in range(horizon):
            states[t + 1] = a @ states[t] + b @ inputs[t]
        return states


@dataclass
class AgentLimits:
    """Box constraints for one agent. ``None`` disables the corresponding constraint."""

    u_max: float | None = 3.0
    """Max magnitude of each acceleration component (infinity-norm, not 2-norm)."""

    v_max: float | None = 2.0
    """Max magnitude of each velocity component."""

    p_min: NDArray[np.float64] | None = None
    """Lower workspace corner, shape ``(dim,)``."""

    p_max: NDArray[np.float64] | None = None
    """Upper workspace corner, shape ``(dim,)``."""

    def validate(self, dim: int) -> None:
        """Raise ``ValueError`` on shape/sign mismatches. Call once at solver setup."""
        if self.u_max is not None and self.u_max <= 0:
            raise ValueError(f"u_max must be positive, got {self.u_max}")
        if self.v_max is not None and self.v_max <= 0:
            raise ValueError(f"v_max must be positive, got {self.v_max}")
        for name, bound in (("p_min", self.p_min), ("p_max", self.p_max)):
            if bound is not None and np.asarray(bound).shape != (dim,):
                raise ValueError(f"{name} must have shape ({dim},), got {np.asarray(bound).shape}")
        if (
            self.p_min is not None
            and self.p_max is not None
            and np.any(np.asarray(self.p_min) > np.asarray(self.p_max))
        ):
            raise ValueError("p_min must be component-wise <= p_max")


@dataclass
class AgentCostWeights:
    """Stage and terminal weights for one agent's local objective.

    All weights are scalars multiplying identity blocks; that keeps the QP data trivially
    positive semidefinite and keeps the notebooks readable. Swap to full matrices only if
    an experiment needs anisotropic weighting.
    """

    q_position: float = 1.0
    """Tracking weight on ``|| p_i(t) - p_ref(t) ||^2``. Set to 0 for pure followers."""

    q_velocity: float = 0.1
    """Damping weight on ``|| v_i(t) ||^2`` (or velocity tracking if a reference has one)."""

    r_input: float = 0.05
    """Effort weight on ``|| u_i(t) ||^2``."""

    r_rate: float = 0.0
    """Smoothness weight on ``|| u_i(t) - u_i(t-1) ||^2``. Needs ``u_prev`` in the data."""

    p_terminal: float = 5.0
    """Terminal weight multiplying the position error at ``t = T``."""

    w_formation: float = 10.0
    """Weight on each formation edge residual ``|| (y_i^i - y_i^j) - d_ij ||^2``."""

    def validate(self) -> None:
        """Raise ``ValueError`` if any weight is negative."""
        for name in (
            "q_position",
            "q_velocity",
            "r_input",
            "r_rate",
            "p_terminal",
            "w_formation",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass
class LocalProblemData:
    """Everything agent ``i`` needs to run one x-update.

    Fields split into three groups: *structural* (fixed for the whole run — used to build
    the parametrised CVXPY problem once), *episodic* (changes each control step) and
    *iterative* (changes each ADMM iteration).
    """

    # --- structural -----------------------------------------------------------------
    agent_id: int
    horizon: int
    model: DoubleIntegrator
    limits: AgentLimits
    weights: AgentCostWeights
    neighborhood: tuple[int, ...]
    """``closed_neighborhood(i)``, sorted; defines the ordering of ``y``, ``z``, ``lam``."""

    offsets: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    """``{j: d_ij}`` desired ``p_i - p_j`` for each formation edge ``j`` in ``neighbors(i)``."""

    # --- episodic -------------------------------------------------------------------
    x0: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    """Current measured state, shape ``(2*dim,)``."""

    reference: NDArray[np.float64] | None = None
    """Position reference over the horizon, shape ``(T, dim)``. ``None`` -> no tracking term."""

    u_prev: NDArray[np.float64] | None = None
    """Input applied at the previous control step, shape ``(dim,)``; only used if ``r_rate > 0``."""

    # --- iterative ------------------------------------------------------------------
    rho: float = 1.0
    z: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    """``{j: z^j}``, each ``(T, dim)``; keys must cover ``neighborhood``."""

    lam: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    """``{j: lam_i^j}``, each ``(T, dim)``; scaled duals (already divided by rho)."""

    def validate(self) -> None:
        """Check that ``z`` and ``lam`` cover ``neighborhood`` with the right shapes."""
        shape = (self.horizon, self.model.dim)
        for j in self.neighborhood:
            for name, mapping in (("z", self.z), ("lam", self.lam)):
                if j not in mapping:
                    raise KeyError(f"neighbor {j} missing from {name} for agent {self.agent_id}")
                value = np.asarray(mapping[j])
                if value.shape != shape:
                    raise ValueError(f"{name}[{j}] must have shape {shape}, got {value.shape}")


@dataclass
class LocalSolution:
    """Result of one x-update."""

    agent_id: int
    inputs: NDArray[np.float64]
    """``U_i``, shape ``(T, dim)``."""

    copies: dict[int, NDArray[np.float64]]
    """``{j: y_i^j}``, each ``(T, dim)``. ``copies[agent_id]`` is the agent's own trajectory."""

    states: NDArray[np.float64]
    """Predicted full state trajectory ``(T, 2*dim)`` implied by ``inputs``."""

    local_objective: float
    """Value of ``f_i`` *excluding* the augmented-Lagrangian penalty. Summing this over
    agents gives the objective curve plotted in the convergence figures."""

    solve_time: float
    status: str

    @property
    def own_trajectory(self) -> NDArray[np.float64]:
        """Shorthand for ``copies[agent_id]``."""
        return self.copies[self.agent_id]


class PerAgentSolver(ABC):
    """Interface for the x-update. Implementations must be stateless across agents but may
    cache compiled problem structure for their own ``agent_id``."""

    # Shared state every concrete solver exposes to the consensus loop.
    _model: DoubleIntegrator
    _limits: AgentLimits
    _weights: AgentCostWeights

    @abstractmethod
    def solve(self, data: LocalProblemData) -> LocalSolution:
        """Solve agent ``data.agent_id``'s local QP. Must not mutate ``data``."""

    @abstractmethod
    def warm_start(self, solution: LocalSolution) -> None:
        """Seed the next solve from a previous one (typically the previous ADMM iterate)."""

    def reset(self) -> None:
        """Drop any warm-start state. Default is a no-op."""
        return None


class CvxpyAgentSolver(PerAgentSolver):
    """CVXPY/OSQP implementation of the local QP.

    Performance contract
    --------------------
    The CVXPY problem is compiled **once** in :meth:`__init__` with every quantity that
    changes between solves declared as a ``cp.Parameter`` (``x0``, ``reference``, ``rho``,
    and one ``z``/``lam`` parameter per neighbor). :meth:`solve` only assigns
    ``.value`` and calls ``problem.solve(warm_start=True)``. Rebuilding the problem inside
    ``solve`` makes a 4-agent run roughly two orders of magnitude slower and is the single
    most common way to get this module wrong.

    Because ``rho`` multiplies a squared norm, declare it as a ``cp.Parameter(nonneg=True)``
    so DPP-compliance is preserved; otherwise CVXPY re-canonicalises on every solve and you
    lose the benefit. If DPP still complains, substitute ``sqrt_rho`` as the parameter and
    penalise ``|| sqrt_rho * (y - z + lam) ||^2``.
    """

    def __init__(
        self,
        agent_id: int,
        horizon: int,
        model: DoubleIntegrator,
        limits: AgentLimits,
        weights: AgentCostWeights,
        neighborhood: tuple[int, ...],
        offsets: dict[int, NDArray[np.float64]] | None = None,
        solver: str = "OSQP",
        solver_options: dict[str, object] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._horizon = horizon
        self._model = model
        self._limits = limits
        self._weights = weights
        self._neighborhood = tuple(neighborhood)
        self._offsets = dict(offsets) if offsets is not None else {}
        self._solver = solver
        self._solver_options = dict(solver_options) if solver_options is not None else {}

        limits.validate(model.dim)
        weights.validate()
        if agent_id not in self._neighborhood:
            raise ValueError(f"agent_id {agent_id} must belong to its own closed neighborhood")

        t_steps, dim = horizon, model.dim
        phi_p, gamma_p = model.position_prediction_matrices(horizon)
        phi_v, gamma_v = model.velocity_prediction_matrices(horizon)

        # --- variables ---------------------------------------------------------
        self._U = cp.Variable(t_steps * dim, name="U")
        self._y = {j: cp.Variable(t_steps * dim, name=f"y_{j}") for j in self._neighborhood}

        # --- parameters --------------------------------------------------------
        self._x0_p = cp.Parameter(model.n_states, name="x0")
        self._ref_p = cp.Parameter(t_steps * dim, name="ref")
        self._rho_p = cp.Parameter(nonneg=True, name="rho")
        self._w_p = {j: cp.Parameter(t_steps * dim, name=f"w_{j}") for j in self._neighborhood}
        self._u_prev_p = cp.Parameter(dim, name="u_prev")
        self._u_prev_active = cp.Parameter(nonneg=True, name="u_prev_active")

        # Formation edges enter as runtime parameters, not as baked-in constants, so that
        # ``LocalProblemData.offsets`` genuinely takes effect and a morph event
        # (:func:`formation_constraints.interpolate_formations`) needs no solver rebuild.
        # This mirrors ``PerAgentQp::setOffsets`` on the C++ side, which likewise updates
        # only the linear term. ``_wf_p[j]`` is the effective weight for edge ``j`` (zero
        # when that edge carries no offset) and ``_wd_p[j]`` is ``w * d_ij`` tiled.
        self._formation_edges = tuple(j for j in self._neighborhood if j != agent_id)
        self._wf_p = {j: cp.Parameter(nonneg=True, name=f"wf_{j}") for j in self._formation_edges}
        self._wd_p = {j: cp.Parameter(t_steps * dim, name=f"wd_{j}") for j in self._formation_edges}

        y_self = self._y[agent_id]

        # --- constraints -------------------------------------------------------
        constraints = []
        constraints.append(y_self == phi_p @ self._x0_p + gamma_p @ self._U)

        if limits.u_max is not None:
            constraints.append(cp.abs(self._U) <= limits.u_max)

        velocity = phi_v @ self._x0_p + gamma_v @ self._U
        if limits.v_max is not None:
            constraints.append(cp.abs(velocity) <= limits.v_max)

        if limits.p_min is not None:
            constraints.append(y_self >= np.tile(limits.p_min, t_steps))
        if limits.p_max is not None:
            constraints.append(y_self <= np.tile(limits.p_max, t_steps))

        # --- objective ---------------------------------------------------------
        objective = weights.q_position * cp.sum_squares(y_self - self._ref_p)
        objective += weights.p_terminal * cp.sum_squares(
            y_self[(t_steps - 1) * dim :] - self._ref_p[(t_steps - 1) * dim :]
        )
        objective += weights.q_velocity * cp.sum_squares(velocity)
        objective += weights.r_input * cp.sum_squares(self._U)

        if weights.r_rate > 0:
            objective += weights.r_rate * (
                cp.sum_squares(self._U[dim:] - self._U[:-dim])
                + self._u_prev_active * cp.sum_squares(self._U[:dim] - self._u_prev_p)
            )

        for j in self._offsets:
            if j != agent_id and j not in self._neighborhood:
                raise ValueError(
                    f"offset key {j} is not in the closed neighborhood of agent {agent_id}"
                )

        # DPP-safe expansion of w*||y_self - y_j - d||^2, dropping the w*||d||^2 constant
        # (it does not change the argmin; ``_compute_local_objective`` adds it back
        # numerically when the objective value itself is reported):
        #     w*||y_self - y_j||^2 - 2*<y_self - y_j, w*d>
        # The quadratic term is parameter * parameter-free expression and the linear term
        # is affine in the parameter, so both are DPP.
        for j in self._formation_edges:
            difference = y_self - self._y[j]
            objective += self._wf_p[j] * cp.sum_squares(difference)
            objective -= 2.0 * (difference @ self._wd_p[j])

        # Consensus penalty, DPP-safe expansion of (rho/2)||y - z + lam||^2:
        #     (rho/2)||y||^2 - <y, rho*(z - lam)> + const
        # The linear term is affine in the single parameter w_p[j] = rho*(z - lam).
        for j in self._neighborhood:
            objective += 0.5 * self._rho_p * cp.sum_squares(self._y[j])
            objective -= self._y[j] @ self._w_p[j]

        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self._problem.is_dpp():
            raise RuntimeError(
                "local QP is not DPP-compliant; it would re-canonicalise on every solve"
            )

    def solve(self, data: LocalProblemData) -> LocalSolution:
        self._assign_parameters(data)
        start = time.perf_counter()
        self._problem.solve(solver=self._solver, warm_start=True, **self._solver_options)
        solve_time = time.perf_counter() - start
        # Accept the OSQP "user_limit" (max-iterations) status: under packet loss or a
        # poorly scaled penalty the subproblem may not converge, but its last iterate is
        # still a valid, finite step and crashing would break the outer ADMM loop.
        if self._problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT):
            raise RuntimeError(
                f"local QP for agent {data.agent_id} returned non-optimal status "
                f"{self._problem.status!r}"
            )
        return self._extract_solution(data, solve_time)

    def warm_start(self, solution: LocalSolution) -> None:
        self._U.value = solution.inputs.ravel()
        for j, value in solution.copies.items():
            if j in self._y:
                self._y[j].value = value.ravel()

    def reset(self) -> None:
        self._U.value = None
        for j in self._neighborhood:
            self._y[j].value = None

    # ------------------------------------------------------------------ internals

    def _effective_offsets(self, data: LocalProblemData) -> dict[int, NDArray[np.float64]]:
        """Formation offsets in force for this solve.

        ``data.offsets`` wins when it is non-empty; otherwise the constructor defaults
        apply. Passing offsets at solve time used to be silently ignored — the formation
        cost was compiled from the constructor argument alone — which produced a run that
        converged cleanly on the *uncoupled* problem. Keys outside the closed
        neighborhood now raise instead of being dropped.
        """
        source = data.offsets if data.offsets else self._offsets
        effective: dict[int, NDArray[np.float64]] = {}
        for j, offset in source.items():
            if j == self._agent_id:
                continue
            if j not in self._neighborhood:
                raise ValueError(
                    f"agent {self._agent_id}: offset key {j} is not in its closed neighborhood "
                    f"{self._neighborhood}"
                )
            value = np.asarray(offset, dtype=np.float64).reshape(-1)
            if value.size != self._model.dim:
                raise ValueError(
                    f"agent {self._agent_id}: offset[{j}] must have {self._model.dim} "
                    f"components, got {value.size}"
                )
            effective[j] = value
        return effective

    def _assign_parameters(self, data: LocalProblemData) -> None:
        """Copy ``data`` into the CVXPY parameters. Raise if a neighbor key is missing."""
        data.validate()
        self._x0_p.value = np.asarray(data.x0, dtype=np.float64)

        offsets = self._effective_offsets(data)
        weight = self._weights.w_formation
        for j in self._formation_edges:
            offset = offsets.get(j)
            if offset is None:
                self._wf_p[j].value = 0.0
                self._wd_p[j].value = np.zeros(self._horizon * self._model.dim)
            else:
                self._wf_p[j].value = weight
                self._wd_p[j].value = weight * np.tile(offset, self._horizon)

        if data.reference is None:
            if self._weights.q_position > 0 or self._weights.p_terminal > 0:
                raise ValueError(
                    f"agent {data.agent_id}: reference is None but tracking weights are non-zero"
                )
            self._ref_p.value = np.zeros(self._horizon * self._model.dim)
        else:
            self._ref_p.value = np.asarray(data.reference, dtype=np.float64).ravel()

        self._rho_p.value = float(data.rho)

        for j in self._neighborhood:
            if j not in data.z:
                raise KeyError(f"agent {data.agent_id}: missing z[{j}]")
            if j not in data.lam:
                raise KeyError(f"agent {data.agent_id}: missing lam[{j}]")
            z = np.asarray(data.z[j], dtype=np.float64)
            lam = np.asarray(data.lam[j], dtype=np.float64)
            self._w_p[j].value = float(data.rho) * (z - lam).ravel()

        if self._weights.r_rate > 0:
            if data.u_prev is None:
                self._u_prev_p.value = np.zeros(self._model.dim)
                self._u_prev_active.value = 0.0
            else:
                self._u_prev_p.value = np.asarray(data.u_prev, dtype=np.float64)
                self._u_prev_active.value = 1.0

    def _extract_solution(self, data: LocalProblemData, solve_time: float) -> LocalSolution:
        """Read variable values back out and compute ``local_objective``."""
        t_steps, dim = self._horizon, self._model.dim
        inputs = np.asarray(self._U.value, dtype=np.float64).reshape(t_steps, dim)
        copies = {
            j: np.asarray(self._y[j].value, dtype=np.float64).reshape(t_steps, dim)
            for j in self._neighborhood
        }
        phi, gamma = self._model.prediction_matrices(t_steps)
        x0 = np.asarray(data.x0, dtype=np.float64)
        states = (phi @ x0 + gamma @ inputs.ravel()).reshape(t_steps, self._model.n_states)
        local_objective = self._compute_local_objective(data, inputs, copies)
        return LocalSolution(
            agent_id=self._agent_id,
            inputs=inputs,
            copies=copies,
            states=states,
            local_objective=local_objective,
            solve_time=solve_time,
            status=self._problem.status,
        )

    def _compute_local_objective(
        self,
        data: LocalProblemData,
        inputs: NDArray[np.float64],
        copies: dict[int, NDArray[np.float64]],
    ) -> float:
        """Evaluate ``f_i`` (without the augmented-Lagrangian penalty) numerically."""
        t_steps, dim = self._horizon, self._model.dim
        weights = self._weights
        y_self = copies[self._agent_id]

        value = 0.0
        if data.reference is not None:
            ref = np.asarray(data.reference, dtype=np.float64).reshape(t_steps, dim)
            value += weights.q_position * float(np.sum((y_self - ref) ** 2))
            value += weights.p_terminal * float(np.sum((y_self[-1] - ref[-1]) ** 2))

        phi_v, gamma_v = self._model.velocity_prediction_matrices(t_steps)
        x0 = np.asarray(data.x0, dtype=np.float64)
        velocity = (phi_v @ x0 + gamma_v @ inputs.ravel()).reshape(t_steps, dim)
        value += weights.q_velocity * float(np.sum(velocity**2))
        value += weights.r_input * float(np.sum(inputs**2))

        if weights.r_rate > 0:
            previous = (
                np.asarray(data.u_prev, dtype=np.float64) if data.u_prev is not None else inputs[0]
            )
            seq = np.vstack([previous[None, :], inputs])
            value += weights.r_rate * float(np.sum((seq[1:] - seq[:-1]) ** 2))

        for j, offset in self._effective_offsets(data).items():
            value += weights.w_formation * float(np.sum((y_self - copies[j] - offset) ** 2))

        return value


def build_reference_trajectory(
    waypoints: NDArray[np.float64],
    horizon: int,
    start_step: int,
    dt: float,
) -> NDArray[np.float64]:
    """Slice a global reference into the ``(T, dim)`` window seen at control step ``start_step``.

    Clamps past the end of ``waypoints`` (hold-last), which keeps the terminal cost
    well-defined when the horizon runs off the end of the mission.
    """
    idx = np.clip(np.arange(start_step + 1, start_step + horizon + 1), 0, len(waypoints) - 1)
    return waypoints[idx]
