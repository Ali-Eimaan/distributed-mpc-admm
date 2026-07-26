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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
        raise NotImplementedError

    @property
    def n_inputs(self) -> int:
        """``dim``."""
        raise NotImplementedError

    def matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(A, B)`` with shapes ``(2*dim, 2*dim)`` and ``(2*dim, dim)``."""
        raise NotImplementedError

    def position_selector(self) -> NDArray[np.float64]:
        """``C_p`` with ``p = C_p x``; shape ``(dim, 2*dim)``."""
        raise NotImplementedError

    def velocity_selector(self) -> NDArray[np.float64]:
        """``C_v`` with ``v = C_v x``; shape ``(dim, 2*dim)``."""
        raise NotImplementedError

    def prediction_matrices(
        self, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Condensed prediction ``X = Phi x0 + Gamma U`` over ``t = 1..T``.

        Returns ``(Phi, Gamma)`` with shapes ``(T*2*dim, 2*dim)`` and
        ``(T*2*dim, T*dim)``. ``Gamma`` is block lower-triangular with
        ``Gamma[t, s] = A^{t-s-1} B`` for ``s <= t``.

        Note the horizon starts at ``t = 1``: ``x0`` itself is *not* part of ``X``.
        """
        raise NotImplementedError

    def position_prediction_matrices(
        self, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Rows of :meth:`prediction_matrices` that produce positions only.

        Returns ``(Phi_p, Gamma_p)`` with shapes ``(T*dim, 2*dim)`` and ``(T*dim, T*dim)``.
        This is the map used in the equality constraint ``y_i^i = Phi_p x0 + Gamma_p U_i``.
        Cache the result — it depends only on ``(dt, dim, horizon)``.
        """
        raise NotImplementedError

    def simulate(
        self, x0: NDArray[np.float64], inputs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Roll the true plant forward. ``inputs`` is ``(T, dim)``; returns ``(T+1, 2*dim)``
        including ``x0`` as row 0."""
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


class PerAgentSolver(ABC):
    """Interface for the x-update. Implementations must be stateless across agents but may
    cache compiled problem structure for their own ``agent_id``."""

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
        # TODO [GUIDE 4.3]: build the parametrised problem here. Concretely:
        #   1. variables: U (T, dim); y = {j: cp.Variable((T, dim)) for j in neighborhood}
        #   2. parameters: x0_p (2*dim,), ref_p (T, dim), rho_p (nonneg scalar),
        #      z_p[j] (T, dim), lam_p[j] (T, dim), u_prev_p (dim,)
        #   3. constraint: cp.vec(y[agent_id]) == Phi_p @ x0_p + Gamma_p @ cp.vec(U)
        #      -- fix the vec order (CVXPY uses Fortran/column-major by default; pass
        #      order="C" or build Phi_p/Gamma_p to match, and assert it in a unit test)
        #   4. constraints: cp.abs(U) <= u_max; cp.abs(V) <= v_max via the velocity rows
        #      of the full-state prediction; workspace box on y[agent_id]
        #   5. objective: tracking + input + rate + terminal + formation + consensus penalty
        raise NotImplementedError

    def solve(self, data: LocalProblemData) -> LocalSolution:
        raise NotImplementedError

    def warm_start(self, solution: LocalSolution) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ internals

    def _assign_parameters(self, data: LocalProblemData) -> None:
        """Copy ``data`` into the CVXPY parameters. Raise if a neighbor key is missing."""
        raise NotImplementedError

    def _extract_solution(self, data: LocalProblemData, solve_time: float) -> LocalSolution:
        """Read variable values back out and compute ``local_objective``."""
        raise NotImplementedError


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
    raise NotImplementedError
