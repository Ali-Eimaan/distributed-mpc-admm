# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-decomposition (subgradient dual ascent) baseline for the ADMM comparison.

This module exists for exactly one purpose: to give ``docs/COMPARISON_VS_DUAL_DECOMP.md``
measured numbers rather than textbook claims. It splits the *same* coupled problem across
the *same* graph as :mod:`consensus_admm`, but drops the augmented (quadratic) term and
climbs the dual with a fixed-step subgradient ascent instead.

Algorithm (contrast with :mod:`consensus_admm`)
------------------------------------------------
For iteration ``k``, dual decomposition does:

1. **x-update** (parallel):  ``y_i <- argmin  f_i(y_i) + sum_{j in Ncl(i)} nu_i^j . y_i^j``
   — a linear dual term, *no* quadratic penalty.
2. **z-update**:  ``z^j <- (1/|C(j)|) sum_{i in C(j)} y_i^j``  — plain average, *no* dual term.
3. **dual update**:  ``nu_i^j <- nu_i^j + step * (y_i^j - z^j)``  — subgradient ascent.

The only difference from ADMM is the missing ``(rho/2)||y - z + lam||^2`` term: ADMM's
quadratic penalty and its ``+lam`` inside the average both collapse. That single term is
what makes ADMM's iteration a contraction; without it the subgradient step must use a
diminishing (or carefully hand-tuned constant) step size, and the practical convergence
rate drops from linear to sublinear.

Two structural weaknesses, made concrete:

* The x-update is unbounded below (or multi-valued) unless ``f_i`` is strictly convex in
  every free copy ``y_i^j``. With the default weights the tracking term pins ``y_i^i`` and
  ``w_formation`` pins each copy, so the problem is well-posed; set ``q_position``,
  ``q_velocity``, ``r_input`` and ``p_terminal`` all to zero and the subproblem has a null
  direction, which the driver below demonstrates.
* The step size must be tuned to a Lipschitz constant nobody knows a priori; ADMM's
  ``rho`` plays the same role but its good region is dramatically wider (see the
  U-curve comparison in the notebook/script).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from .communication_graph import CommunicationGraph
from .consensus_admm import ADMMHistory
from .per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    DoubleIntegrator,
    LocalProblemData,
    LocalSolution,
)

__all__ = [
    "DualDecomposition",
    "DualDecompositionAgentSolver",
    "DualDecompositionOptions",
    "DualDecompositionResult",
]

TrajectoryMap = dict[int, NDArray[np.float64]]


@dataclass
class DualDecompositionOptions:
    """Tuning knobs for one dual-decomposition solve."""

    step_size: float = 0.5
    """Fixed subgradient step. There is no adaptive variant — that is part of the point."""

    max_iterations: int = 2000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    verbose: bool = False

    def validate(self) -> None:
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.eps_abs < 0 or self.eps_rel < 0:
            raise ValueError("eps_abs and eps_rel must be non-negative")


@dataclass
class DualDecompositionResult:
    """Output of :meth:`DualDecomposition.solve`."""

    inputs: NDArray[np.float64]
    trajectories: NDArray[np.float64]
    states: NDArray[np.float64]
    local_copies: dict[int, TrajectoryMap]
    duals: dict[int, TrajectoryMap]
    consensus: TrajectoryMap
    history: ADMMHistory
    iterations: int
    converged: bool
    solve_time: float


class DualDecompositionAgentSolver:
    """Per-agent QP for dual decomposition: ``min  f_i(y_i) + sum_j nu_i^j . y_i^j``.

    Structurally identical to :class:`~distributed_mpc_admm.per_agent_solver.CvxpyAgentSolver`
    except that the augmented-Lagrangian penalty is replaced by a linear dual term. The
    unscaled dual ``nu_i^j`` is passed through ``LocalProblemData.lam`` (the field is just a
    per-neighbor array); ``rho`` and ``z`` are ignored.
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
        self._nu_p = {j: cp.Parameter(t_steps * dim, name=f"nu_{j}") for j in self._neighborhood}
        self._u_prev_p = cp.Parameter(dim, name="u_prev")
        self._u_prev_active = cp.Parameter(nonneg=True, name="u_prev_active")

        y_self = self._y[agent_id]

        # --- constraints (identical to CvxpyAgentSolver) ----------------------
        constraints = [y_self == phi_p @ self._x0_p + gamma_p @ self._U]
        if limits.u_max is not None:
            constraints.append(cp.abs(self._U) <= limits.u_max)
        velocity = phi_v @ self._x0_p + gamma_v @ self._U
        if limits.v_max is not None:
            constraints.append(cp.abs(velocity) <= limits.v_max)
        if limits.p_min is not None:
            constraints.append(y_self >= np.tile(limits.p_min, t_steps))
        if limits.p_max is not None:
            constraints.append(y_self <= np.tile(limits.p_max, t_steps))

        # --- objective: f_i (no penalty) + linear dual term -------------------
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

        for j, offset in self._offsets.items():
            if j == agent_id:
                continue
            if j not in self._neighborhood:
                raise ValueError(
                    f"offset key {j} is not in the closed neighborhood of agent {agent_id}"
                )
            d_full = np.tile(np.asarray(offset, dtype=np.float64), t_steps)
            objective += weights.w_formation * cp.sum_squares(y_self - self._y[j] - d_full)

        for j in self._neighborhood:
            objective += self._nu_p[j] @ self._y[j]

        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self._problem.is_dpp():
            raise RuntimeError("dual-decomposition local QP is not DPP-compliant")

    def solve(self, data: LocalProblemData) -> LocalSolution:
        data.validate()
        self._x0_p.value = np.asarray(data.x0, dtype=np.float64)
        if data.reference is None:
            if self._weights.q_position > 0 or self._weights.p_terminal > 0:
                raise ValueError(
                    f"agent {data.agent_id}: reference is None but tracking weights are non-zero"
                )
            self._ref_p.value = np.zeros(self._horizon * self._model.dim)
        else:
            self._ref_p.value = np.asarray(data.reference, dtype=np.float64).ravel()

        for j in self._neighborhood:
            self._nu_p[j].value = np.asarray(data.lam[j], dtype=np.float64).ravel()

        if self._weights.r_rate > 0:
            if data.u_prev is None:
                self._u_prev_p.value = np.zeros(self._model.dim)
                self._u_prev_active.value = 0.0
            else:
                self._u_prev_p.value = np.asarray(data.u_prev, dtype=np.float64)
                self._u_prev_active.value = 1.0

        start = time.perf_counter()
        self._problem.solve(solver=self._solver, warm_start=True, **self._solver_options)
        solve_time = time.perf_counter() - start
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

    def _extract_solution(self, data: LocalProblemData, solve_time: float) -> LocalSolution:
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
        """Evaluate ``f_i`` *without* the linear dual term, matching ADMM's convention."""
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

        for j, offset in self._offsets.items():
            if j == self._agent_id:
                continue
            d_full = np.asarray(offset, dtype=np.float64)
            value += weights.w_formation * float(np.sum((y_self - copies[j] - d_full) ** 2))

        return value


class DualDecomposition:
    """One open-loop dual-decomposition solve over a fixed graph.

    Mirrors :class:`~distributed_mpc_admm.consensus_admm.ConsensusADMM` so the two can be
    benchmarked on identical instances. No channel/loss model is provided — the ADMM
    comparison is performed on perfect synchronous communication, where dual decomposition
    is already at its most favourable.
    """

    def __init__(
        self,
        graph: CommunicationGraph,
        solvers: Mapping[int, DualDecompositionAgentSolver],
        horizon: int,
        dim: int = 2,
        options: DualDecompositionOptions | None = None,
    ) -> None:
        self._graph = graph
        self._solvers = dict(solvers)
        self._horizon = int(horizon)
        self._dim = int(dim)
        self._opts = options if options is not None else DualDecompositionOptions()
        self._opts.validate()

        expected = set(range(graph.n_agents))
        if set(self._solvers) != expected:
            raise ValueError("solvers must cover every agent")

        self._y: dict[int, TrajectoryMap] = {}
        self._nu: dict[int, TrajectoryMap] = {}
        self._z: TrajectoryMap = {}
        for i in range(graph.n_agents):
            neighborhood = graph.closed_neighborhood(i)
            self._y[i] = {j: np.zeros((self._horizon, self._dim)) for j in neighborhood}
            self._nu[i] = {j: np.zeros((self._horizon, self._dim)) for j in neighborhood}
        for j in range(graph.n_agents):
            self._z[j] = np.zeros((self._horizon, self._dim))
        self._latest_solutions: dict[int, LocalSolution] = {}

    def solve(
        self,
        x0: NDArray[np.float64],
        references: Mapping[int, NDArray[np.float64]] | None = None,
        offsets: Mapping[int, dict[int, NDArray[np.float64]]] | None = None,
    ) -> DualDecompositionResult:
        graph = self._graph
        n_agents = graph.n_agents
        opts = self._opts
        start_time = time.perf_counter()

        history = ADMMHistory()
        converged = False

        for iteration in range(opts.max_iterations):
            iter_start = time.perf_counter()
            z_prev = {j: self._z[j].copy() for j in range(n_agents)}

            # --- x-update (parallel) ------------------------------------------
            solutions = self._x_update(x0, references, offsets)
            self._y = {
                i: {j: solutions[i].copies[j].copy() for j in graph.closed_neighborhood(i)}
                for i in range(n_agents)
            }
            self._latest_solutions = solutions

            # --- z-update: plain average, no dual term ------------------------
            for j in range(n_agents):
                self._z[j] = np.mean(
                    np.stack([self._y[i][j] for i in graph.contributors(j)]), axis=0
                )

            # --- dual update: subgradient ascent ------------------------------
            step = opts.step_size
            for i in range(n_agents):
                for j in graph.closed_neighborhood(i):
                    self._nu[i][j] = self._nu[i][j] + step * (self._y[i][j] - self._z[j])

            primal_residual, dual_residual = self._residuals(self._y, self._z, z_prev)
            eps_primal, eps_dual = self._tolerances(self._y, self._z, self._nu)
            converged = primal_residual <= eps_primal and dual_residual <= eps_dual

            history.append(
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                eps_primal=eps_primal,
                eps_dual=eps_dual,
                rho=step,
                objective=float(sum(s.local_objective for s in solutions.values())),
                iteration_time=time.perf_counter() - iter_start,
            )

            if opts.verbose and (iteration % 100 == 0 or converged):
                print(
                    f"[DD] iter {iteration:5d}  r={primal_residual:.3e}  "
                    f"s={dual_residual:.3e}  eps={eps_primal:.3e}/{eps_dual:.3e}"
                )

            if converged:
                break

        inputs = np.stack([np.asarray(self._latest_solutions[i].inputs) for i in range(n_agents)])
        trajectories = np.stack([self._z[j] for j in range(n_agents)])
        states = np.stack([np.asarray(self._latest_solutions[i].states) for i in range(n_agents)])
        local_copies = {
            i: {j: self._y[i][j].copy() for j in graph.closed_neighborhood(i)}
            for i in range(n_agents)
        }
        duals = {
            i: {j: self._nu[i][j].copy() for j in graph.closed_neighborhood(i)}
            for i in range(n_agents)
        }

        return DualDecompositionResult(
            inputs=inputs,
            trajectories=trajectories,
            states=states,
            local_copies=local_copies,
            duals=duals,
            consensus={j: self._z[j].copy() for j in range(n_agents)},
            history=history,
            iterations=iteration + 1,
            converged=converged,
            solve_time=time.perf_counter() - start_time,
        )

    def _x_update(
        self,
        x0: NDArray[np.float64],
        references: Mapping[int, NDArray[np.float64]] | None,
        offsets: Mapping[int, dict[int, NDArray[np.float64]]] | None,
    ) -> dict[int, LocalSolution]:
        graph = self._graph
        solutions: dict[int, LocalSolution] = {}
        for i in range(graph.n_agents):
            solver = self._solvers[i]
            neighborhood = tuple(graph.closed_neighborhood(i))
            agent_offsets = offsets.get(i) if offsets is not None else {}
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
                rho=1.0,
                z={j: self._z[j] for j in neighborhood},
                lam={j: self._nu[i][j] for j in neighborhood},
            )
            solutions[i] = solver.solve(data)
        return solutions

    def _residuals(
        self,
        y: dict[int, TrajectoryMap],
        z: TrajectoryMap,
        z_prev: TrajectoryMap,
    ) -> tuple[float, float]:
        primal_sq = 0.0
        dual_sq = 0.0
        for i in y:
            for j in y[i]:
                primal_sq += float(np.sum((y[i][j] - z[j]) ** 2))
                dual_sq += float(np.sum((z[j] - z_prev[j]) ** 2))
        return float(np.sqrt(primal_sq)), float(np.sqrt(dual_sq))

    def _tolerances(
        self,
        y: dict[int, TrajectoryMap],
        z: TrajectoryMap,
        nu: dict[int, TrajectoryMap],
    ) -> tuple[float, float]:
        opts = self._opts
        n_dual = 0
        y_norm_sq = 0.0
        z_norm_sq = 0.0
        nu_norm_sq = 0.0
        for i in y:
            for j in y[i]:
                n_dual += y[i][j].size
                y_norm_sq += float(np.sum(y[i][j] ** 2))
                z_norm_sq += float(np.sum(z[j] ** 2))
                nu_norm_sq += float(np.sum(nu[i][j] ** 2))
        eps_pri = np.sqrt(n_dual) * opts.eps_abs + opts.eps_rel * max(
            np.sqrt(y_norm_sq), np.sqrt(z_norm_sq)
        )
        eps_dual = np.sqrt(n_dual) * opts.eps_abs + opts.eps_rel * np.sqrt(nu_norm_sq)
        return float(eps_pri), float(eps_dual)
