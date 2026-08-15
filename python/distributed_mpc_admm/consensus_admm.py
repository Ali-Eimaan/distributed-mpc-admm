# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""General-form consensus ADMM over a communication graph, and the closed-loop driver.

Algorithm (scaled dual form, Boyd et al. 2011, sections 7.2 and 3.4.1)
----------------------------------------------------------------------
For ADMM iteration ``k``:

1. **x-update** (fully parallel, one QP per agent)::

       (U_i, y_i) <- argmin  f_i(U_i, y_i)
                             + (rho/2) sum_{j in Ncl(i)} || y_i^j - z^j_k + lam_i^j_k ||_F^2

2. **relaxation** (optional, ``alpha`` in ``[1, 2)``)::

       yhat_i^j <- alpha * y_i^j + (1 - alpha) * z^j_k

3. **z-update** (neighborhood averaging; computed *by agent j* from what its
   contributors report, then broadcast back)::

       z^j_{k+1} <- (1 / |C_j|) * sum_{i in C_j} ( yhat_i^j + lam_i^j_k )

   where ``C_j = contributors(j) = Ncl(j)``. This step is what makes the method
   distributed: it needs one round of neighbor-to-neighbor messages, nothing global.

4. **dual update**::

       lam_i^j_{k+1} <- lam_i^j_k + yhat_i^j - z^j_{k+1}

Residuals and stopping
----------------------
::

    r_k = sqrt( sum_i sum_{j in Ncl(i)} || y_i^j - z^j_k ||_F^2 )         (primal)
    s_k = rho * sqrt( sum_i sum_{j in Ncl(i)} || z^j_k - z^j_{k-1} ||_F^2 ) (dual)

    eps_pri  = sqrt(n_dual) * eps_abs + eps_rel * max(||y||, ||z||)
    eps_dual = sqrt(n_dual) * eps_abs + eps_rel * rho * ||lam||

with ``n_dual = sum_i |Ncl(i)| * T * dim``. Stop when ``r_k <= eps_pri`` **and**
``s_k <= eps_dual``.

Adaptive rho (residual balancing)
---------------------------------
If ``r_k > mu * s_k`` then ``rho <- tau * rho`` and ``lam <- lam / tau``; if
``s_k > mu * r_k`` then ``rho <- rho / tau`` and ``lam <- lam * tau``. The dual rescaling
is mandatory — the scaled duals absorb ``rho``, so changing ``rho`` without rescaling
silently changes the dual iterate and can stall convergence. Defaults ``mu = 10``,
``tau = 2``. Disable adaptation before quoting any convergence-rate claim, since the
standard rate proof assumes fixed ``rho``.

Asynchrony
----------
When a :class:`~distributed_mpc_admm.communication_graph.LossyChannel` is supplied, an
agent that fails to receive ``z^j`` reuses its last known value. The iteration is then no
longer the textbook ADMM and the convergence guarantee does not apply — measuring exactly
where it breaks is the point of ``notebooks/04_switching_topology.ipynb``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .communication_graph import CommunicationGraph, LossyChannel, Message, TimeVaryingGraph
from .formation_constraints import FormationSpec, LeaderFollowerSpec, formation_error, settling_step
from .per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    CvxpyAgentSolver,
    DoubleIntegrator,
    LocalProblemData,
    LocalSolution,
    PerAgentSolver,
    build_reference_trajectory,
)

__all__ = [
    "ADMMHistory",
    "ADMMOptions",
    "ADMMResult",
    "ConsensusADMM",
    "DistributedMPC",
    "MPCOptions",
    "SimulationLog",
]

TrajectoryMap = dict[int, NDArray[np.float64]]
"""``{agent_id: (T, dim) trajectory}``."""


@dataclass
class ADMMOptions:
    """Tuning knobs for one ADMM solve."""

    rho: float = 1.0
    """Initial penalty parameter. Sensitivity to this is the headline result of
    ``notebooks/05_convergence_analysis.ipynb``."""

    max_iterations: int = 200
    eps_abs: float = 1e-4
    eps_rel: float = 1e-3

    alpha: float = 1.0
    """Over-relaxation. ``1.0`` disables it; ``1.5``-``1.8`` typically buys 20-40 percent
    fewer iterations at no per-iteration cost."""

    adaptive_rho: bool = False
    mu: float = 10.0
    """Residual-imbalance threshold that triggers a rho update."""

    tau: float = 2.0
    """Multiplicative rho step."""

    rho_min: float = 1e-4
    rho_max: float = 1e4

    warm_start: bool = True
    """Carry ``(y, z, lam)`` across control steps (time-shifted by one). Cuts closed-loop
    iteration counts by roughly an order of magnitude after the first step."""

    check_every: int = 1
    """Evaluate residuals every ``check_every`` iterations (cheap here; matters in C++)."""

    record_objective: bool = False
    """Sum the per-agent ``local_objective`` each iteration. Off by default because it is
    only meaningful once the primal residual is small."""

    verbose: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` on ``alpha`` outside ``[1, 2)``, non-positive rho, etc."""
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.eps_abs < 0:
            raise ValueError(f"eps_abs must be non-negative, got {self.eps_abs}")
        if self.eps_rel < 0:
            raise ValueError(f"eps_rel must be non-negative, got {self.eps_rel}")
        if not (1.0 <= self.alpha < 2.0):
            raise ValueError(f"alpha must be in [1, 2), got {self.alpha}")
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.tau <= 1:
            raise ValueError(f"tau must be > 1 for adaptive rho, got {self.tau}")
        if self.rho_min <= 0:
            raise ValueError(f"rho_min must be positive, got {self.rho_min}")
        if self.rho_max < self.rho_min:
            raise ValueError(f"rho_max must be >= rho_min, got [{self.rho_min}, {self.rho_max}]")
        if self.check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {self.check_every}")


@dataclass
class ADMMHistory:
    """Per-iteration diagnostics. Every list has one entry per *checked* iteration."""

    primal_residual: list[float] = field(default_factory=list)
    dual_residual: list[float] = field(default_factory=list)
    eps_primal: list[float] = field(default_factory=list)
    eps_dual: list[float] = field(default_factory=list)
    rho: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    consensus_gap: list[float] = field(default_factory=list)
    """``max_j max_i || y_i^j - z^j ||_inf`` — the infinity-norm disagreement, which is
    the quantity a reviewer will ask about (the 2-norm residual can hide one bad agent)."""

    iteration_time: list[float] = field(default_factory=list)
    messages_sent: list[int] = field(default_factory=list)
    messages_dropped: list[int] = field(default_factory=list)

    def append(self, **kwargs: float | int) -> None:
        """Push one row; missing keys append ``nan``/``0`` so all lists stay equal length."""
        float_fields = (
            "primal_residual",
            "dual_residual",
            "eps_primal",
            "eps_dual",
            "rho",
            "objective",
            "consensus_gap",
            "iteration_time",
        )
        int_fields = ("messages_sent", "messages_dropped")
        for name in float_fields:
            getattr(self, name).append(kwargs.get(name, float("nan")))
        for name in int_fields:
            getattr(self, name).append(kwargs.get(name, 0))

    def as_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Numpy view for plotting."""
        return {
            name: np.asarray(getattr(self, name), dtype=np.float64)
            for name in self.__dataclass_fields__
        }

    def empirical_rate(self, skip_initial: int = 5) -> float:
        """Least-squares slope of ``log(primal_residual)`` versus iteration.

        Reported as the measured linear rate and compared against the
        ``lambda_2``-dependent bound in ``docs/derivations/convergence_proof.tex``.
        """
        residual = np.asarray(self.primal_residual, dtype=np.float64)
        if residual.size <= skip_initial:
            return float("nan")
        iterations = np.arange(residual.size, dtype=np.float64)
        mask = (iterations >= skip_initial) & (residual > 0)
        if np.count_nonzero(mask) < 2:
            return float("nan")
        slope = np.polyfit(iterations[mask], np.log(residual[mask]), 1)[0]
        return float(slope)


@dataclass
class ADMMResult:
    """Output of :meth:`ConsensusADMM.solve`."""

    inputs: NDArray[np.float64]
    """``(N, T, dim)`` optimal input sequences."""

    trajectories: NDArray[np.float64]
    """``(N, T, dim)`` consensus position trajectories, i.e. ``z^j`` stacked over ``j``."""

    states: NDArray[np.float64]
    """``(N, T, 2*dim)`` predicted full states."""

    local_copies: dict[int, TrajectoryMap]
    """``{i: {j: y_i^j}}`` at termination; needed to warm-start the next control step."""

    duals: dict[int, TrajectoryMap]
    """``{i: {j: lam_i^j}}`` at termination."""

    consensus: TrajectoryMap
    """``{j: z^j}`` at termination."""

    history: ADMMHistory
    iterations: int
    converged: bool
    rho_final: float
    solve_time: float

    def first_inputs(self) -> NDArray[np.float64]:
        """``(N, dim)`` — the receding-horizon control actually applied."""
        return self.inputs[:, 0, :]

    def shifted(self) -> tuple[dict[int, TrajectoryMap], TrajectoryMap, dict[int, TrajectoryMap]]:
        """Time-shift ``(local_copies, consensus, duals)`` by one step for warm starting.

        Drop the first row and repeat the last one. Duals are shifted the same way;
        zeroing them instead throws away most of the warm-start benefit.
        """

        def _shift(trajectory: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.concatenate([trajectory[1:], trajectory[-1:]], axis=0)

        def _shift_map(mapping: TrajectoryMap) -> TrajectoryMap:
            return {k: _shift(np.asarray(v)) for k, v in mapping.items()}

        local_copies = {i: _shift_map(copies) for i, copies in self.local_copies.items()}
        consensus = _shift_map(self.consensus)
        duals = {i: _shift_map(lam) for i, lam in self.duals.items()}
        return local_copies, consensus, duals


class ConsensusADMM:
    """One open-loop distributed solve: the ADMM iteration itself.

    The class is a *simulation* of a distributed algorithm — it runs the agent updates in a
    loop in one process, but no update is allowed to read data that would be unavailable to
    a real agent. Keep that invariant: any global quantity (residual norms, objective) is
    computed for logging only and must never feed back into an agent update.

    Parameters
    ----------
    graph:
        Fixed topology for this solve. Under a switching schedule, the caller rebuilds a
        :class:`ConsensusADMM` (or calls :meth:`set_graph`) at each switch instant.
    solvers:
        ``{agent_id: PerAgentSolver}``. Each solver must have been constructed with the
        same ``neighborhood`` that ``graph`` implies.
    options:
        See :class:`ADMMOptions`.
    channel:
        Optional lossy channel. ``None`` means perfect synchronous communication.
    """

    def __init__(
        self,
        graph: CommunicationGraph,
        solvers: Mapping[int, PerAgentSolver],
        horizon: int,
        dim: int = 2,
        options: ADMMOptions | None = None,
        channel: LossyChannel | None = None,
    ) -> None:
        self._graph = graph
        self._solvers = dict(solvers)
        self._horizon = int(horizon)
        self._dim = int(dim)
        self._opts = options if options is not None else ADMMOptions()
        self._opts.validate()
        self._channel = channel

        self._validate_solvers(graph)

        # Allocate the iteration state and the per-agent "last known z" cache.
        self._y: dict[int, TrajectoryMap] = {}
        self._lam: dict[int, TrajectoryMap] = {}
        self._z: TrajectoryMap = {}
        self._z_view: dict[int, TrajectoryMap] = {}
        self._z_last_known: dict[int, dict[int, NDArray[np.float64]]] = {}

        for i in range(graph.n_agents):
            neighborhood = graph.closed_neighborhood(i)
            self._y[i] = {j: np.zeros((self._horizon, self._dim)) for j in neighborhood}
            self._lam[i] = {j: np.zeros((self._horizon, self._dim)) for j in neighborhood}
            self._z_last_known[i] = {}
        for j in range(graph.n_agents):
            self._z[j] = np.zeros((self._horizon, self._dim))
        self._z_view = {i: dict(self._z) for i in range(graph.n_agents)}
        self._latest_solutions: dict[int, LocalSolution] = {}

    def _validate_solvers(self, graph: CommunicationGraph) -> None:
        """Raise if ``self._solvers`` does not cover every agent with the right neighborhood."""
        expected = set(range(graph.n_agents))
        actual = set(self._solvers)
        if actual != expected:
            raise ValueError(
                f"solvers must cover agents 0..{graph.n_agents - 1}; "
                f"missing {sorted(expected - actual)}, unexpected {sorted(actual - expected)}"
            )
        for i in range(graph.n_agents):
            neighborhood = tuple(graph.closed_neighborhood(i))
            solver_neighborhood = getattr(self._solvers[i], "_neighborhood", None)
            if solver_neighborhood is not None and tuple(solver_neighborhood) != neighborhood:
                raise ValueError(
                    f"solver {i} was built for closed neighborhood {tuple(solver_neighborhood)}, "
                    f"but the graph implies {neighborhood}"
                )

    # ------------------------------------------------------------------ public API

    def solve(
        self,
        x0: NDArray[np.float64],
        references: Mapping[int, NDArray[np.float64]] | None = None,
        offsets: Mapping[int, dict[int, NDArray[np.float64]]] | None = None,
        initial_guess: (
            tuple[dict[int, TrajectoryMap], TrajectoryMap, dict[int, TrajectoryMap]] | None
        ) = None,
    ) -> ADMMResult:
        """Run the ADMM iteration to convergence or ``max_iterations``.

        Parameters
        ----------
        x0:
            ``(N, 2*dim)`` current states.
        references:
            ``{i: (T, dim)}`` position references; agents absent from the mapping get none.
        offsets:
            ``{i: {j: d_ij}}`` formation offsets, normally from
            :meth:`FormationSpec.edge_offsets`.
        initial_guess:
            ``(local_copies, consensus, duals)``, typically ``ADMMResult.shifted()`` from
            the previous control step.
        """
        graph = self._graph
        n_agents = graph.n_agents
        opts = self._opts
        start_time = time.perf_counter()

        if self._channel is not None:
            self._channel.clear_messages()

        if initial_guess is not None:
            self._apply_initial_guess(*initial_guess)
        else:
            self._z_view = {i: dict(self._z) for i in range(n_agents)}
            for i in range(n_agents):
                for j in graph.closed_neighborhood(i):
                    self._z_last_known[i][j] = self._z[j].copy()

        history = ADMMHistory()
        rho = opts.rho
        converged = False

        for iteration in range(opts.max_iterations):
            iter_start = time.perf_counter()
            z_prev = {j: self._z[j].copy() for j in range(n_agents)}

            # --- x-update ------------------------------------------------------
            solutions = self._x_update(x0, references, offsets, rho)
            self._y = {
                i: {j: solutions[i].copies[j].copy() for j in graph.closed_neighborhood(i)}
                for i in range(n_agents)
            }
            self._latest_solutions = solutions

            # --- over-relaxation (identity for alpha == 1) ---------------------
            y_hat = self._relax(self._y, self._z) if opts.alpha != 1.0 else self._y

            # --- z-update + broadcast + dual update ---------------------------
            channel_sent_before = self._channel.stats.sent if self._channel is not None else 0
            channel_dropped_before = self._channel.stats.dropped if self._channel is not None else 0

            self._z = self._z_update(y_hat, self._lam, iteration)
            self._z_view = self._broadcast_consensus(self._z, iteration)
            self._lam = self._dual_update(self._lam, y_hat, self._z_view)

            messages_sent = (
                self._channel.stats.sent - channel_sent_before if self._channel is not None else 0
            )
            messages_dropped = (
                self._channel.stats.dropped - channel_dropped_before
                if self._channel is not None
                else 0
            )

            # --- diagnostics ---------------------------------------------------
            primal_residual, dual_residual = self._residuals(self._y, self._z, z_prev, rho)
            eps_primal, eps_dual = self._tolerances(self._y, self._z, self._lam, rho)

            if opts.adaptive_rho:
                rho = self._update_rho(rho, primal_residual, dual_residual, self._lam)

            converged = primal_residual <= eps_primal and dual_residual <= eps_dual

            row: dict[str, float | int] = {
                "primal_residual": primal_residual,
                "dual_residual": dual_residual,
                "eps_primal": eps_primal,
                "eps_dual": eps_dual,
                "rho": rho,
                "messages_sent": messages_sent,
                "messages_dropped": messages_dropped,
                "iteration_time": time.perf_counter() - iter_start,
            }
            if opts.record_objective:
                row["objective"] = float(sum(s.local_objective for s in solutions.values()))
                row["consensus_gap"] = float(
                    max(
                        np.max(np.abs(self._y[i][j] - self._z[j]))
                        for i in range(n_agents)
                        for j in graph.closed_neighborhood(i)
                    )
                )
            history.append(**row)

            if opts.verbose:
                print(
                    f"[ADMM] iter {iteration:4d}  r={primal_residual:.3e}  "
                    f"s={dual_residual:.3e}  eps={eps_primal:.3e}/{eps_dual:.3e}  "
                    f"rho={rho:.3f}"
                )

            if converged:
                break

        # Package the final iterate. ``inputs`` stacks each agent's own control plan.
        inputs = np.stack([np.asarray(self._latest_solutions[i].inputs) for i in range(n_agents)])
        trajectories = np.stack([self._z[j] for j in range(n_agents)])
        states = np.stack([np.asarray(self._latest_solutions[i].states) for i in range(n_agents)])
        local_copies: dict[int, TrajectoryMap] = {
            i: {j: self._y[i][j].copy() for j in graph.closed_neighborhood(i)}
            for i in range(n_agents)
        }
        consensus = {j: self._z[j].copy() for j in range(n_agents)}
        duals: dict[int, TrajectoryMap] = {
            i: {j: self._lam[i][j].copy() for j in graph.closed_neighborhood(i)}
            for i in range(n_agents)
        }

        return ADMMResult(
            inputs=inputs,
            trajectories=trajectories,
            states=states,
            local_copies=local_copies,
            duals=duals,
            consensus=consensus,
            history=history,
            iterations=iteration + 1,
            converged=converged,
            rho_final=rho,
            solve_time=time.perf_counter() - start_time,
        )

    def _apply_initial_guess(
        self,
        local_copies: dict[int, TrajectoryMap],
        consensus: TrajectoryMap,
        duals: dict[int, TrajectoryMap],
    ) -> None:
        """Overwrite ``(y, z, lam)`` and the per-agent z views from a previous result."""
        graph = self._graph
        n_agents = graph.n_agents
        for i in range(n_agents):
            for j in graph.closed_neighborhood(i):
                if j in local_copies.get(i, {}) and local_copies[i][j].shape == self._y[i][j].shape:
                    self._y[i][j] = np.asarray(local_copies[i][j], dtype=np.float64).copy()
                if j in duals.get(i, {}) and duals[i][j].shape == self._lam[i][j].shape:
                    self._lam[i][j] = np.asarray(duals[i][j], dtype=np.float64).copy()
        for j in range(n_agents):
            if j in consensus and consensus[j].shape == self._z[j].shape:
                self._z[j] = np.asarray(consensus[j], dtype=np.float64).copy()
        self._z_view = {i: dict(self._z) for i in range(n_agents)}
        for i in range(n_agents):
            for j in graph.closed_neighborhood(i):
                self._z_last_known[i][j] = self._z[j].copy()

    def set_graph(self, graph: CommunicationGraph) -> None:
        """Swap the topology between solves.

        Agents whose neighborhood changed need their solver rebuilt (the QP structure
        depends on ``|Ncl(i)|``); this method must raise if the supplied solvers no longer
        match, rather than silently solving the wrong problem.
        """
        self._validate_solvers(graph)
        self._graph = graph
        if self._channel is not None:
            self._channel.set_graph(graph)

        # Rebuild the iteration-state dicts for the new neighborhoods, preserving values
        # for entries whose neighborhood membership is unchanged.
        y: dict[int, TrajectoryMap] = {}
        lam: dict[int, TrajectoryMap] = {}
        z_last_known: dict[int, dict[int, NDArray[np.float64]]] = {}
        for i in range(graph.n_agents):
            neighborhood = graph.closed_neighborhood(i)
            y[i] = {}
            lam[i] = {}
            z_last_known[i] = {}
            for j in neighborhood:
                old_y = self._y.get(i, {}).get(j)
                old_lam = self._lam.get(i, {}).get(j)
                y[i][j] = (
                    old_y.copy()
                    if old_y is not None and old_y.shape == (self._horizon, self._dim)
                    else np.zeros((self._horizon, self._dim))
                )
                lam[i][j] = (
                    old_lam.copy()
                    if old_lam is not None and old_lam.shape == (self._horizon, self._dim)
                    else np.zeros((self._horizon, self._dim))
                )
                z_last_known[i][j] = self._z.get(j, np.zeros((self._horizon, self._dim))).copy()
        self._y = y
        self._lam = lam
        self._z_last_known = z_last_known
        self._z_view = {i: dict(self._z) for i in range(graph.n_agents)}

    def reset(self) -> None:
        """Zero ``(y, z, lam)`` and clear the channel."""
        graph = self._graph
        for i in range(graph.n_agents):
            for j in graph.closed_neighborhood(i):
                self._y[i][j] = np.zeros((self._horizon, self._dim))
                self._lam[i][j] = np.zeros((self._horizon, self._dim))
        for j in range(graph.n_agents):
            self._z[j] = np.zeros((self._horizon, self._dim))
        self._z_view = {i: dict(self._z) for i in range(graph.n_agents)}
        self._z_last_known = {
            i: {j: self._z[j].copy() for j in graph.closed_neighborhood(i)}
            for i in range(graph.n_agents)
        }
        if self._channel is not None:
            self._channel.reset()

    # ------------------------------------------------------------------ ADMM steps

    def _x_update(
        self,
        x0: NDArray[np.float64],
        references: Mapping[int, NDArray[np.float64]] | None,
        offsets: Mapping[int, dict[int, NDArray[np.float64]]] | None,
        rho: float,
    ) -> dict[int, LocalSolution]:
        """Solve every agent's local QP. Embarrassingly parallel by construction.

        Keep it a plain loop. A ``ProcessPoolExecutor`` here costs more in pickling than
        it saves for ``N <= 8``; if a parallel variant is added, put it behind a flag and
        assert bit-identical results against the serial path.
        """
        graph = self._graph
        n_agents = graph.n_agents
        solutions: dict[int, LocalSolution] = {}
        for i in range(n_agents):
            solver = self._solvers[i]
            neighborhood = tuple(graph.closed_neighborhood(i))
            agent_offsets = (
                offsets.get(i) if offsets is not None else getattr(solver, "_offsets", {})
            )
            data = LocalProblemData(
                agent_id=i,
                horizon=self._horizon,
                model=solver._model,
                limits=solver._limits,
                weights=solver._weights,
                neighborhood=neighborhood,
                offsets=dict(agent_offsets) if agent_offsets else {},
                x0=np.asarray(x0[i], dtype=np.float64),
                reference=references.get(i) if references is not None else None,
                u_prev=None,
                rho=rho,
                z={j: self._z_view[i][j] for j in neighborhood},
                lam={j: self._lam[i][j] for j in neighborhood},
            )
            solutions[i] = solver.solve(data)
        return solutions

    def _relax(self, y: dict[int, TrajectoryMap], z: TrajectoryMap) -> dict[int, TrajectoryMap]:
        """Apply over-relaxation ``alpha * y + (1 - alpha) * z``. Identity when ``alpha == 1``."""
        alpha = self._opts.alpha
        return {i: {j: alpha * y[i][j] + (1.0 - alpha) * z[j] for j in y[i]} for i in y}

    def _z_update(
        self,
        y_hat: dict[int, TrajectoryMap],
        lam: dict[int, TrajectoryMap],
        iteration: int,
    ) -> TrajectoryMap:
        """Neighborhood averaging, executed *at the subject agent*.

        For each ``j``: gather ``y_i^j + lam_i^j`` from every ``i in contributors(j)`` and
        average. When a channel is present, a contribution that fails to arrive is simply
        excluded from that iteration's average and the divisor shrinks accordingly — do
        **not** substitute a stale value here, because averaging stale and fresh terms
        biases ``z`` in a way that is much harder to analyse than a missing term.
        """
        graph = self._graph
        n_agents = graph.n_agents
        channel = self._channel

        # Phase 1: contributors push their (y + lam) contributions toward the subject.
        if channel is not None:
            for j in range(n_agents):
                for i in graph.contributors(j):
                    if i == j:
                        continue
                    payload = y_hat[i][j] + lam[i][j]
                    channel.send(
                        Message(
                            sender=i,
                            receiver=j,
                            subject=j,
                            admm_iteration=iteration,
                            payload=payload,
                        ),
                        iteration,
                    )
            channel.advance(iteration)

        # Phase 2: each subject averages the contributions that actually arrived.
        z: TrajectoryMap = {}
        for j in range(n_agents):
            contributions = [y_hat[j][j] + lam[j][j]]  # self-contribution is always local
            for i in graph.contributors(j):
                if i == j:
                    continue
                if channel is None:
                    contributions.append(y_hat[i][j] + lam[i][j])
                else:
                    message = channel.receive(j, subject=j, sender=i, iteration=iteration)
                    if message is not None:
                        contributions.append(message.payload)
            z[j] = np.mean(np.stack(contributions), axis=0)
        return z

    def _dual_update(
        self,
        lam: dict[int, TrajectoryMap],
        y_hat: dict[int, TrajectoryMap],
        z: dict[int, TrajectoryMap],
    ) -> dict[int, TrajectoryMap]:
        """``lam += y_hat - z``, using each agent's *received* copy of ``z``."""
        return {i: {j: lam[i][j] + y_hat[i][j] - z[i][j] for j in lam[i]} for i in lam}

    # ------------------------------------------------------------------ diagnostics

    def _residuals(
        self,
        y: dict[int, TrajectoryMap],
        z: TrajectoryMap,
        z_prev: TrajectoryMap,
        rho: float,
    ) -> tuple[float, float]:
        """Return ``(primal_residual, dual_residual)`` per the module docstring."""
        primal_sq = 0.0
        dual_sq = 0.0
        for i in y:
            for j in y[i]:
                primal_sq += float(np.sum((y[i][j] - z[j]) ** 2))
                dual_sq += float(np.sum((z[j] - z_prev[j]) ** 2))
        primal = float(np.sqrt(primal_sq))
        dual = float(rho * np.sqrt(dual_sq))
        return primal, dual

    def _tolerances(
        self,
        y: dict[int, TrajectoryMap],
        z: TrajectoryMap,
        lam: dict[int, TrajectoryMap],
        rho: float,
    ) -> tuple[float, float]:
        """Return ``(eps_primal, eps_dual)``."""
        opts = self._opts
        n_dual = 0
        y_norm_sq = 0.0
        z_norm_sq = 0.0
        lam_norm_sq = 0.0
        for i in y:
            for j in y[i]:
                n_dual += y[i][j].size
                y_norm_sq += float(np.sum(y[i][j] ** 2))
                z_norm_sq += float(np.sum(z[j] ** 2))
                lam_norm_sq += float(np.sum(lam[i][j] ** 2))
        sqrt_n = float(np.sqrt(n_dual))
        eps_primal = sqrt_n * opts.eps_abs + opts.eps_rel * max(
            float(np.sqrt(y_norm_sq)), float(np.sqrt(z_norm_sq))
        )
        eps_dual = sqrt_n * opts.eps_abs + opts.eps_rel * rho * float(np.sqrt(lam_norm_sq))
        return eps_primal, eps_dual

    def _update_rho(
        self, rho: float, primal: float, dual: float, lam: dict[int, TrajectoryMap]
    ) -> float:
        """Residual balancing. **Mutates ``lam`` in place** by the reciprocal factor."""
        opts = self._opts
        if primal > opts.mu * dual:
            factor = opts.tau
        elif dual > opts.mu * primal:
            factor = 1.0 / opts.tau
        else:
            factor = 1.0
        new_rho = float(np.clip(rho * factor, opts.rho_min, opts.rho_max))
        actual = new_rho / rho
        if actual != 1.0:
            for i in lam:
                for j in lam[i]:
                    lam[i][j] = lam[i][j] / actual
        return new_rho

    def _broadcast_consensus(self, z: TrajectoryMap, iteration: int) -> dict[int, TrajectoryMap]:
        """Deliver ``z`` through the channel; returns each agent's *received* view of it.

        With no channel this is ``{i: z for all i}`` (shared reference is fine, the
        downstream code must not mutate it). With a channel, missing entries fall back to
        the per-agent last-known cache.
        """
        graph = self._graph
        n_agents = graph.n_agents
        channel = self._channel

        if channel is None:
            return {i: dict(z) for i in range(n_agents)}

        # Phase 1: subjects broadcast their fresh z to their neighbors.
        for j in range(n_agents):
            channel.broadcast(
                sender=j,
                subject=j,
                payload=z[j],
                iteration=iteration,
                receivers=graph.neighbors(j),
            )
        channel.advance(iteration)

        # Phase 2: each agent reads the freshest z about each neighbor, falling back to
        # the last-known value and updating the cache as fresh values arrive.
        z_view: dict[int, TrajectoryMap] = {}
        for i in range(n_agents):
            z_view[i] = {}
            for j in graph.closed_neighborhood(i):
                if j == i:
                    z_view[i][j] = z[i]
                    self._z_last_known[i][j] = z[i].copy()
                    continue
                message = channel.receive(i, subject=j, sender=j, iteration=iteration)
                if message is not None:
                    z_view[i][j] = message.payload
                    self._z_last_known[i][j] = message.payload.copy()
                else:
                    z_view[i][j] = self._z_last_known[i][j]
        return z_view


@dataclass
class MPCOptions:
    """Closed-loop settings for :class:`DistributedMPC`."""

    horizon: int = 15
    dt: float = 0.1
    n_steps: int = 100
    process_noise_std: float = 0.0
    """Std of zero-mean Gaussian noise added to velocity states each step. Nonzero values
    are what make the warm-start and recursive-feasibility discussion non-trivial."""

    measurement_noise_std: float = 0.0
    seed: int | None = 0

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.process_noise_std < 0:
            raise ValueError(
                f"process_noise_std must be non-negative, got {self.process_noise_std}"
            )
        if self.measurement_noise_std < 0:
            raise ValueError(
                f"measurement_noise_std must be non-negative, got {self.measurement_noise_std}"
            )


@dataclass
class SimulationLog:
    """Closed-loop record. All arrays are indexed by control step ``k``."""

    time: NDArray[np.float64]
    """``(K,)`` seconds."""

    states: NDArray[np.float64]
    """``(K+1, N, 2*dim)`` realised states, including the initial condition."""

    inputs: NDArray[np.float64]
    """``(K, N, dim)`` applied inputs."""

    predictions: list[NDArray[np.float64]]
    """Per step, the ``(N, T, dim)`` predicted trajectory — used for the horizon overlay
    in the animation."""

    admm_iterations: NDArray[np.int_]
    """``(K,)`` iterations used per control step."""

    admm_converged: NDArray[np.bool_]
    formation_error: NDArray[np.float64]
    """``(K,)`` edge-RMS from :func:`formation_constraints.formation_error`."""

    graphs: list[CommunicationGraph]
    """Topology active at each step; drives the edge overlay in the animation."""

    histories: list[ADMMHistory]
    solve_times: NDArray[np.float64]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def positions(self) -> NDArray[np.float64]:
        """``(K+1, N, dim)`` view of the position rows of ``states``."""
        dim = self.states.shape[2] // 2
        return self.states[:, :, :dim]

    def summary(self) -> dict[str, float]:
        """Headline numbers for the README table: mean/max ADMM iterations, final
        formation error, settling step, total solve time, convergence failure count."""
        iterations = np.asarray(self.admm_iterations)
        final_formation_error = (
            float(self.formation_error[-1]) if self.formation_error.size else float("nan")
        )
        settling = (
            settling_step(self.formation_error, tolerance=0.05)
            if self.formation_error.size
            else None
        )
        return {
            "mean_admm_iterations": float(np.mean(iterations)) if iterations.size else float("nan"),
            "max_admm_iterations": float(np.max(iterations)) if iterations.size else float("nan"),
            "final_formation_error": final_formation_error,
            "settling_step": float(settling) if settling is not None else float("nan"),
            "total_solve_time": float(np.sum(self.solve_times)),
            "convergence_failures": float(np.count_nonzero(~self.admm_converged)),
        }

    def save(self, path: str) -> None:
        """Write to a compressed ``.npz`` so notebooks can reload without re-simulating."""
        k = len(self.graphs)
        n_agents = self.graphs[0].n_agents if k else 0
        adjacency = np.zeros((k, n_agents, n_agents), dtype=np.int64)
        for step, graph in enumerate(self.graphs):
            for edge in graph.edges:
                adjacency[step, edge[0], edge[1]] = 1
                adjacency[step, edge[1], edge[0]] = 1

        # Flatten histories into stacked arrays + per-step length indices.
        history_lengths = np.asarray([len(h.primal_residual) for h in self.histories])
        max_len = int(history_lengths.max()) if history_lengths.size else 0

        def _stack(names: Sequence[str]) -> NDArray[np.float64]:
            arrays = np.full((k, max_len), np.nan)
            for step, h in enumerate(self.histories):
                for name in names:
                    data = np.asarray(getattr(h, name), dtype=np.float64)
                    arrays[step, : len(data)] = data
            return arrays

        primal = _stack(["primal_residual"])
        dual = _stack(["dual_residual"])
        eps_primal = _stack(["eps_primal"])
        eps_dual = _stack(["eps_dual"])
        rho = _stack(["rho"])
        iteration_time = _stack(["iteration_time"])

        np.savez_compressed(
            path,
            time=self.time,
            states=self.states,
            inputs=self.inputs,
            predictions=np.stack(self.predictions) if self.predictions else np.empty((0,)),
            admm_iterations=self.admm_iterations,
            admm_converged=self.admm_converged,
            formation_error=self.formation_error,
            adjacency=adjacency,
            solve_times=self.solve_times,
            history_lengths=history_lengths,
            primal=primal,
            dual=dual,
            eps_primal=eps_primal,
            eps_dual=eps_dual,
            rho=rho,
            iteration_time=iteration_time,
        )

    @classmethod
    def load(cls, path: str) -> SimulationLog:
        data = np.load(path, allow_pickle=False)

        predictions: list[NDArray[np.float64]] = []
        if data["predictions"].ndim > 1:
            predictions = [data["predictions"][i] for i in range(data["predictions"].shape[0])]

        adjacency = data["adjacency"]
        graphs: list[CommunicationGraph] = []
        for step in range(adjacency.shape[0]):
            n_agents = adjacency.shape[1]
            edges = {
                (i, j)
                for i in range(n_agents)
                for j in range(i + 1, n_agents)
                if adjacency[step, i, j] != 0
            }
            graphs.append(CommunicationGraph(n_agents, edges))

        histories: list[ADMMHistory] = []
        lengths = np.asarray(data["history_lengths"], dtype=int)
        for step in range(len(graphs)):
            h = ADMMHistory()
            n = int(lengths[step])
            h.primal_residual = list(np.asarray(data["primal"][step, :n], dtype=float))
            h.dual_residual = list(np.asarray(data["dual"][step, :n], dtype=float))
            h.eps_primal = list(np.asarray(data["eps_primal"][step, :n], dtype=float))
            h.eps_dual = list(np.asarray(data["eps_dual"][step, :n], dtype=float))
            h.rho = list(np.asarray(data["rho"][step, :n], dtype=float))
            h.iteration_time = list(np.asarray(data["iteration_time"][step, :n], dtype=float))
            histories.append(h)

        return cls(
            time=np.asarray(data["time"], dtype=np.float64),
            states=np.asarray(data["states"], dtype=np.float64),
            inputs=np.asarray(data["inputs"], dtype=np.float64),
            predictions=predictions,
            admm_iterations=np.asarray(data["admm_iterations"], dtype=int),
            admm_converged=np.asarray(data["admm_converged"], dtype=bool),
            formation_error=np.asarray(data["formation_error"], dtype=np.float64),
            graphs=graphs,
            histories=histories,
            solve_times=np.asarray(data["solve_times"], dtype=np.float64),
        )


class DistributedMPC:
    """Receding-horizon driver: solve, apply the first input, shift, repeat.

    Each control step ``k``:

    1. Look up the active topology ``graph = schedule.at(k)``; rebuild solvers if the
       neighborhoods changed.
    2. Build the per-agent references from ``leader_follower``.
    3. Run :meth:`ConsensusADMM.solve`, warm-started from step ``k-1``.
    4. Apply ``result.first_inputs()`` to the plant model, add noise, log.
    """

    def __init__(
        self,
        model: DoubleIntegrator,
        graph: CommunicationGraph | TimeVaryingGraph,
        formation: FormationSpec,
        limits: AgentLimits,
        weights: AgentCostWeights | Sequence[AgentCostWeights],
        admm_options: ADMMOptions | None = None,
        mpc_options: MPCOptions | None = None,
        leader_follower: LeaderFollowerSpec | None = None,
        channel: LossyChannel | None = None,
    ) -> None:
        self._model = model
        self._formation = formation
        self._limits = limits
        self._admm_options = admm_options if admm_options is not None else ADMMOptions()
        self._admm_options.validate()
        self._mpc_options = mpc_options if mpc_options is not None else MPCOptions()
        self._mpc_options.validate()
        self._leader_follower = leader_follower
        self._channel = channel

        self._schedule = (
            graph if isinstance(graph, TimeVaryingGraph) else TimeVaryingGraph([graph], mode="hold")
        )

        n_agents = formation.n_agents
        if isinstance(weights, AgentCostWeights):
            base_weights = [replace(weights) for _ in range(n_agents)]
        else:
            base_weights = [replace(w) for w in weights]
            if len(base_weights) != n_agents:
                raise ValueError(f"expected {n_agents} weight sets, got {len(base_weights)}")

        self._weights: list[AgentCostWeights] = []
        for i in range(n_agents):
            w = base_weights[i]
            if leader_follower is not None:
                w = replace(
                    w,
                    q_position=leader_follower.weight_for(i, w.q_position),
                    p_terminal=leader_follower.weight_for(i, w.p_terminal),
                )
            self._weights.append(w)

        initial_graph = self._schedule.at(0)
        self._dropped_edges: frozenset[tuple[int, int]] = frozenset()
        try:
            self._formation.validate_against(initial_graph)
        except ValueError:
            self._dropped_edges = self._missing_formation_edges(initial_graph)
        if leader_follower is not None:
            leader_follower.validate_against(initial_graph)

        self._solver_cache: dict[tuple[int, tuple[int, ...]], CvxpyAgentSolver] = {}
        self._rebuild_solvers(initial_graph)
        solvers = {
            i: self._solver_cache[(i, tuple(initial_graph.closed_neighborhood(i)))]
            for i in range(n_agents)
        }
        self._admm = ConsensusADMM(
            initial_graph,
            solvers,
            self._mpc_options.horizon,
            model.dim,
            self._admm_options,
            channel,
        )
        self._active_graph = initial_graph
        self._last_guess: (
            tuple[dict[int, TrajectoryMap], TrajectoryMap, dict[int, TrajectoryMap]] | None
        ) = None

    def step(self, k: int, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], ADMMResult]:
        """One control step. Returns ``((N, dim) applied input, result)``."""
        graph = self._schedule.at(k)
        if graph != self._active_graph:
            self._rebuild_solvers(graph)
            for i in range(graph.n_agents):
                self._admm._solvers[i] = self._solver_cache[
                    (i, tuple(graph.closed_neighborhood(i)))
                ]
            self._admm.set_graph(graph)
            self._active_graph = graph
            try:
                self._formation.validate_against(graph)
                self._dropped_edges = frozenset()
            except ValueError:
                self._dropped_edges = self._missing_formation_edges(graph)

        references = self._references_at(k)
        offsets = {i: self._formation.edge_offsets(i) for i in range(graph.n_agents)}
        result = self._admm.solve(
            x0=np.asarray(x, dtype=np.float64),
            references=references,
            offsets=offsets,
            initial_guess=self._last_guess,
        )
        self._last_guess = result.shifted()
        return result.first_inputs(), result

    def run(self, x0: NDArray[np.float64], n_steps: int | None = None) -> SimulationLog:
        """Full closed-loop simulation from ``x0`` of shape ``(N, 2*dim)``."""
        opts = self._mpc_options
        n_steps = opts.n_steps if n_steps is None else int(n_steps)
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        n_agents = self._formation.n_agents
        dim = self._model.dim
        rng = np.random.default_rng(opts.seed)

        x_true = np.asarray(x0, dtype=np.float64).copy()
        states = np.zeros((n_steps + 1, n_agents, 2 * dim))
        states[0] = x_true

        inputs = np.zeros((n_steps, n_agents, dim))
        predictions: list[NDArray[np.float64]] = []
        admm_iterations = np.zeros(n_steps, dtype=int)
        admm_converged = np.zeros(n_steps, dtype=bool)
        formation_error_log = np.zeros(n_steps)
        graphs: list[CommunicationGraph] = []
        histories: list[ADMMHistory] = []
        solve_times = np.zeros(n_steps)
        rng = np.random.default_rng(opts.seed)

        for k in range(n_steps):
            x_measured = x_true.copy()
            if opts.measurement_noise_std > 0:
                x_measured[:dim] += rng.normal(0.0, opts.measurement_noise_std, size=dim)

            applied, result = self.step(k, x_measured)

            inputs[k] = applied
            predictions.append(np.stack([result.local_copies[i][i] for i in range(n_agents)]))
            admm_iterations[k] = result.iterations
            admm_converged[k] = result.converged
            solve_times[k] = result.solve_time
            graphs.append(self._schedule.at(k))
            histories.append(result.history)

            # Advance the true plant by one step (per-agent), then add process noise.
            x_next = np.empty_like(x_true)
            for i in range(n_agents):
                x_next[i] = self._model.simulate(x_true[i], applied[i][None, :])[1]
            x_true = x_next
            if opts.process_noise_std > 0:
                x_true[:, dim:] += rng.normal(0.0, opts.process_noise_std, size=(n_agents, dim))
            states[k + 1] = x_true

            formation_error_log[k] = formation_error(x_true[:, :dim], self._formation).edge_rms

        metadata: dict[str, object] = {}
        if self._dropped_edges:
            metadata["dropped_formation_edges"] = sorted(self._dropped_edges)
        switch_steps = self._schedule.switch_steps(0, n_steps)
        if switch_steps:
            metadata["switch_steps"] = switch_steps

        return SimulationLog(
            time=np.asarray(opts.dt * np.arange(n_steps), dtype=np.float64),
            states=states,
            inputs=inputs,
            predictions=predictions,
            admm_iterations=admm_iterations,
            admm_converged=admm_converged,
            formation_error=formation_error_log,
            graphs=graphs,
            histories=histories,
            solve_times=solve_times,
            metadata=metadata,
        )

    def _rebuild_solvers(self, graph: CommunicationGraph) -> None:
        """Recreate only the solvers whose closed neighborhood changed."""
        for i in range(graph.n_agents):
            key = (i, tuple(graph.closed_neighborhood(i)))
            if key in self._solver_cache:
                continue
            comm_neighbors = set(key[1])
            offsets = {
                j: d for j, d in self._formation.edge_offsets(i).items() if j in comm_neighbors
            }
            self._solver_cache[key] = CvxpyAgentSolver(
                agent_id=i,
                horizon=self._mpc_options.horizon,
                model=self._model,
                limits=self._limits,
                weights=self._weights[i],
                neighborhood=key[1],
                offsets=offsets,
            )

    def _missing_formation_edges(self, graph: CommunicationGraph) -> frozenset[tuple[int, int]]:
        """Formation edges absent from the active communication graph."""
        missing: set[tuple[int, int]] = set()
        for a, b in self._formation.graph.edges:
            if not graph.has_edge(a, b):
                missing.add((a, b) if a <= b else (b, a))
        return frozenset(missing)

    def _references_at(self, k: int) -> dict[int, NDArray[np.float64]]:
        """Slice the leader reference into per-agent ``(T, dim)`` windows."""
        if self._leader_follower is None:
            return {}
        window = build_reference_trajectory(
            self._leader_follower.reference,
            self._mpc_options.horizon,
            k,
            self._mpc_options.dt,
        )
        return {i: window for i in range(self._formation.n_agents)}
