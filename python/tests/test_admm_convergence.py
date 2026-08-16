# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Convergence properties of the consensus ADMM loop.

These tests are the contract for :mod:`distributed_mpc_admm.consensus_admm`. They are
written against *properties* (monotone residuals, agreement with a centralised solve,
invariance to agent relabelling) rather than golden numbers, so they survive re-tuning.

The centralised reference used throughout is the single QP that stacks all agents'
variables and imposes the coupling exactly. If ADMM does not reproduce it to a few times
the solver tolerance, everything downstream in the repo is meaningless — so
:func:`test_matches_centralized_solution` is the one test that must never be skipped or
loosened.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from distributed_mpc_admm.communication_graph import CommunicationGraph, LossyChannel
from distributed_mpc_admm.consensus_admm import ADMMOptions, ConsensusADMM
from distributed_mpc_admm.formation_constraints import FormationSpec
from distributed_mpc_admm.per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    CvxpyAgentSolver,
    DoubleIntegrator,
)

# --------------------------------------------------------------------------- fixtures


def _build_solvers(graph, model, horizon, weights, limits, offsets=None):
    """Build one ``CvxpyAgentSolver`` per agent, baking in the formation offsets."""
    return {
        i: CvxpyAgentSolver(
            agent_id=i,
            horizon=horizon,
            model=model,
            limits=limits,
            weights=weights,
            neighborhood=tuple(graph.closed_neighborhood(i)),
            offsets=dict(offsets[i]) if offsets and offsets.get(i) else None,
        )
        for i in range(graph.n_agents)
    }


def _zero_references(n_agents, horizon, dim):
    """Rendezvous/tracking references: every agent tracks the origin."""
    return {i: np.zeros((horizon, dim)) for i in range(n_agents)}


def _random_x0(rng, n_agents, model, velocity_scale=0.5):
    """Random initial states with gentler velocities so ``v_max`` stays feasible."""
    x0 = rng.normal(size=(n_agents, model.n_states))
    x0[:, model.dim :] *= velocity_scale
    return x0


@pytest.fixture
def model() -> DoubleIntegrator:
    """dt = 0.1 s, 2D."""
    return DoubleIntegrator(dt=0.1, dim=2)


@pytest.fixture
def horizon() -> int:
    """Short horizon (T = 10) so the whole suite runs in well under a minute."""
    return 10


@pytest.fixture
def four_agent_setup(model, horizon):
    """4 agents on a cycle, rendezvous formation, random but seeded initial states.

    Returns a small namespace with ``graph``, ``solvers``, ``x0``, ``offsets``,
    ``references``.
    """
    graph = CommunicationGraph.cycle(4)
    weights = AgentCostWeights()
    limits = AgentLimits()
    rng = np.random.default_rng(0)
    x0 = _random_x0(rng, 4, model)
    offsets = {i: {j: np.zeros(model.dim) for j in graph.neighbors(i)} for i in range(4)}
    solvers = _build_solvers(graph, model, horizon, weights, limits, offsets)
    references = _zero_references(4, horizon, model.dim)
    return types.SimpleNamespace(
        graph=graph,
        model=model,
        horizon=horizon,
        weights=weights,
        limits=limits,
        solvers=solvers,
        x0=x0,
        offsets=offsets,
        references=references,
    )


def solve_centralized(x0, graph, model, horizon, weights, limits, offsets):
    """Reference: one monolithic CVXPY problem over all agents.

    Build the *same* objective ADMM targets — per-agent tracking/effort plus the formation
    edge costs — with the coupling imposed directly instead of through local copies. No
    ADMM, no local copies, no duals.
    """
    import cvxpy as cp

    n_agents = graph.n_agents
    t_steps, dim = horizon, model.dim
    phi_p, gamma_p = model.position_prediction_matrices(horizon)
    phi_v, gamma_v = model.velocity_prediction_matrices(horizon)

    U = cp.Variable((n_agents, t_steps * dim))
    P = [phi_p @ x0[i] + gamma_p @ U[i] for i in range(n_agents)]
    V = [phi_v @ x0[i] + gamma_v @ U[i] for i in range(n_agents)]

    objective = 0.0
    constraints: list = []
    for i in range(n_agents):
        objective += weights.q_position * cp.sum_squares(P[i])
        objective += weights.p_terminal * cp.sum_squares(P[i][(t_steps - 1) * dim :])
        objective += weights.q_velocity * cp.sum_squares(V[i])
        objective += weights.r_input * cp.sum_squares(U[i])
        if limits.u_max is not None:
            constraints.append(cp.abs(U[i]) <= limits.u_max)
        if limits.v_max is not None:
            constraints.append(cp.abs(V[i]) <= limits.v_max)

    for i, neighbors in offsets.items():
        for j, d in neighbors.items():
            if j == i:
                continue
            d_full = np.tile(np.asarray(d, dtype=float), t_steps)
            objective += weights.w_formation * cp.sum_squares(P[i] - P[j] - d_full)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="OSQP", eps_abs=1e-6, eps_rel=1e-6, max_iter=100000)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"centralized solve failed: {problem.status}")
    return np.asarray(U.value).reshape(n_agents, t_steps, dim)


# --------------------------------------------------------------------------- core


def test_prediction_matrices_match_rollout(model, horizon):
    """``Phi x0 + Gamma U`` must equal an explicit forward simulation.

    Catches the two classic bugs at once: an off-by-one in the horizon indexing (does
    ``X`` start at ``t=0`` or ``t=1``?) and a transposed vec order in ``Gamma``.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=model.n_states)
    U = rng.normal(size=(horizon, model.dim))
    phi, gamma = model.prediction_matrices(horizon)
    predicted = (phi @ x0 + gamma @ U.ravel()).reshape(horizon, model.n_states)
    rollout = model.simulate(x0, U)[1:]
    assert np.allclose(predicted, rollout, atol=1e-10)


def test_position_prediction_is_row_subset(model, horizon):
    """``Phi_p``/``Gamma_p`` are exactly the position rows of ``Phi``/``Gamma``."""
    phi, gamma = model.prediction_matrices(horizon)
    phi_p, gamma_p = model.position_prediction_matrices(horizon)
    n = model.n_states
    idx = np.concatenate([np.arange(t * n, t * n + model.dim) for t in range(horizon)])
    assert np.array_equal(phi_p, phi[idx])
    assert np.array_equal(gamma_p, gamma[idx])


def test_single_agent_admm_equals_local_qp(model, horizon):
    """With ``N = 1`` and no edges, ADMM must reproduce the plain local QP.

    The consensus penalty is self-referential here: ``z^0 = y_0^0`` at the fixed point and
    the dual is zero, so the converged iterate is exactly the unconstrained local QP
    minimizer.
    """
    graph = CommunicationGraph(1)
    weights = AgentCostWeights()
    limits = AgentLimits()
    solvers = _build_solvers(graph, model, horizon, weights, limits)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(1, model.n_states))
    reference = np.zeros((horizon, model.dim))
    admm = ConsensusADMM(
        graph,
        solvers,
        horizon,
        dim=model.dim,
        options=ADMMOptions(max_iterations=2000, eps_abs=1e-8, eps_rel=1e-8),
    )
    result = admm.solve(x0, references={0: reference})
    assert result.converged
    centralized = solve_centralized(x0, graph, model, horizon, weights, limits, {})
    assert np.allclose(result.inputs[0], centralized[0], atol=1e-5)


def test_matches_centralized_solution(four_agent_setup):
    """ADMM converges to the centralised optimum.

    Tolerance: ``1e-3`` on the stacked input vector with ``eps_abs = 1e-6``,
    ``eps_rel = 1e-6``, ``max_iterations = 2000``. Compare inputs, not just cost — equal
    cost with different inputs means the problem is not strictly convex and the weights
    need fixing.
    """
    s = four_agent_setup
    options = ADMMOptions(max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6)
    admm = ConsensusADMM(s.graph, s.solvers, s.horizon, dim=s.model.dim, options=options)
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert result.converged
    centralized = solve_centralized(
        s.x0, s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets
    )
    assert np.allclose(result.inputs, centralized, atol=1e-3)


def test_solve_time_offsets_take_effect(model, horizon):
    """Offsets supplied to ``solve()`` must couple the agents, not be silently dropped.

    Regression test. The formation cost used to be compiled from the ``CvxpyAgentSolver``
    constructor argument alone, so ``ConsensusADMM.solve(offsets=...)`` — a documented
    public parameter — was accepted and ignored. The run then converged cleanly on the
    *uncoupled* problem: no error, no warning, plausible trajectories, wrong answer.

    Three checks, because any one alone would pass for the wrong reason:
      1. solve-time offsets match constructor offsets (the two paths agree),
      2. they differ from the no-formation run (the coupling is actually present),
      3. a solve-time offset outside the closed neighborhood raises.
    """
    graph = CommunicationGraph.cycle(4)
    weights = AgentCostWeights()
    limits = AgentLimits()
    x0 = _random_x0(np.random.default_rng(7), 4, model)
    references = _zero_references(4, horizon, model.dim)
    square = FormationSpec.regular_polygon(4, radius=1.0, graph=graph)
    offsets = {i: square.edge_offsets(i) for i in range(4)}
    options = ADMMOptions(max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6)

    def run(constructor_offsets, solve_offsets):
        solvers = _build_solvers(graph, model, horizon, weights, limits, constructor_offsets)
        admm = ConsensusADMM(graph, solvers, horizon, dim=model.dim, options=options)
        return admm.solve(x0, references=references, offsets=solve_offsets)

    at_construction = run(offsets, None)
    at_solve_time = run(None, offsets)
    uncoupled = run(None, None)

    assert at_construction.converged and at_solve_time.converged
    assert np.allclose(at_solve_time.inputs, at_construction.inputs, atol=1e-4)
    # The formation must actually change the answer, or check 1 proves nothing.
    assert not np.allclose(at_solve_time.inputs, uncoupled.inputs, atol=1e-2)

    bad = {i: dict(offsets[i]) for i in range(4)}
    bad[0][2] = np.zeros(model.dim)  # agent 2 is not a neighbour of agent 0 on a cycle
    with pytest.raises(ValueError, match="closed neighborhood"):
        run(None, bad)


def test_residuals_decrease(four_agent_setup):
    """Primal residual decreases over a moving window and ends below tolerance.

    Do not assert strict monotonicity per iteration: with over-relaxation or adaptive rho
    the primal residual legitimately increases on individual steps. Assert instead that
    the min over each successive 10-iteration window is non-increasing.
    """
    s = four_agent_setup
    admm = ConsensusADMM(
        s.graph, s.solvers, s.horizon, dim=s.model.dim, options=ADMMOptions(max_iterations=2000)
    )
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert result.converged

    primal = np.asarray(result.history.primal_residual, dtype=float)
    window = 10
    minima = [primal[i : i + window].min() for i in range(0, len(primal) - window + 1, window)]
    assert all(minima[k + 1] <= minima[k] + 1e-12 for k in range(len(minima) - 1))
    assert result.history.primal_residual[-1] <= result.history.eps_primal[-1]


def test_converges_on_all_topologies(model, horizon):
    """Complete, cycle, path and star graphs all reach tolerance within ``max_iterations``.

    With a shared reference and no formation offsets the global problem is topology
    independent, so every topology must converge to the *same* inputs. (Iteration
    counts are not compared directly: the stopping tolerance scales with the number of
    consensus terms, which grows with graph density.)
    """
    builders = {
        "complete": CommunicationGraph.complete,
        "cycle": CommunicationGraph.cycle,
        "path": CommunicationGraph.path,
        "star": CommunicationGraph.star,
    }
    weights = AgentCostWeights()
    limits = AgentLimits()
    rng = np.random.default_rng(1)
    x0 = _random_x0(rng, 4, model)
    references = _zero_references(4, horizon, model.dim)
    solutions: dict[str, np.ndarray] = {}
    for name, builder in builders.items():
        graph = builder(4)
        solvers = _build_solvers(graph, model, horizon, weights, limits)
        admm = ConsensusADMM(
            graph,
            solvers,
            horizon,
            dim=model.dim,
            options=ADMMOptions(max_iterations=2000),
        )
        result = admm.solve(x0, references=references)
        assert result.converged, f"{name} topology did not converge"
        solutions[name] = result.inputs
    # Topology-invariant fixed point: all topologies agree with the cycle solution.
    reference_solution = solutions["cycle"]
    for name, inputs in solutions.items():
        assert np.allclose(inputs, reference_solution, atol=1e-2), name


def test_disconnected_graph_does_not_reach_consensus(model, horizon):
    """Two isolated components converge internally but not to a common formation.

    ``is_connected()`` is ``False`` and ``algebraic_connectivity()`` is ~0; the solver must
    still terminate cleanly rather than hang or emit NaNs.
    """
    graph = CommunicationGraph(4, [(0, 1), (2, 3)])
    assert not graph.is_connected()
    assert graph.algebraic_connectivity() < 1e-9

    weights = AgentCostWeights(q_position=0.0, p_terminal=0.0)
    limits = AgentLimits()
    offsets = {i: {j: np.zeros(model.dim) for j in graph.neighbors(i)} for i in range(4)}
    solvers = _build_solvers(graph, model, horizon, weights, limits, offsets)
    rng = np.random.default_rng(2)
    x0 = _random_x0(rng, 4, model)
    admm = ConsensusADMM(
        graph,
        solvers,
        horizon,
        dim=model.dim,
        options=ADMMOptions(max_iterations=2000),
    )
    result = admm.solve(x0, references=None, offsets=offsets)
    assert np.all(np.isfinite(result.inputs))
    # Each component internally agrees, but the two components need not coincide.
    final_positions = result.trajectories[:, -1, :]
    separation = np.linalg.norm(final_positions[0] - final_positions[2])
    assert separation > 1e-2, "disconnected components unexpectedly met at a common formation"


# --------------------------------------------------------------------------- options


@pytest.mark.parametrize("rho", [0.5, 1.0, 10.0])
def test_converges_for_range_of_rho(four_agent_setup, rho):
    """Same optimum for every rho; only the iteration count changes.

    This is the test that catches a missing dual rescaling in ``_update_rho`` — the fixed
    points differ if rho is folded into the duals incorrectly.
    """
    s = four_agent_setup
    options = ADMMOptions(rho=rho, max_iterations=2000)
    admm = ConsensusADMM(s.graph, s.solvers, s.horizon, dim=s.model.dim, options=options)
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert result.converged
    centralized = solve_centralized(
        s.x0, s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets
    )
    assert np.allclose(result.inputs, centralized, atol=1e-2)


@pytest.mark.parametrize("alpha", [1.0, 1.6])
def test_over_relaxation_preserves_optimum(four_agent_setup, alpha):
    """Over-relaxation changes the path, not the fixed point."""
    s = four_agent_setup
    options = ADMMOptions(alpha=alpha, max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6)
    admm = ConsensusADMM(s.graph, s.solvers, s.horizon, dim=s.model.dim, options=options)
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert result.converged
    centralized = solve_centralized(
        s.x0, s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets
    )
    assert np.allclose(result.inputs, centralized, atol=1e-3)


def test_adaptive_rho_reaches_same_optimum(four_agent_setup):
    """Adaptive rho matches fixed rho to tolerance and does not use more iterations."""
    s = four_agent_setup
    centralized = solve_centralized(
        s.x0, s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets
    )

    fixed = ConsensusADMM(
        s.graph,
        _build_solvers(s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets),
        s.horizon,
        dim=s.model.dim,
        options=ADMMOptions(max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6),
    ).solve(s.x0, references=s.references, offsets=s.offsets)

    adaptive = ConsensusADMM(
        s.graph,
        _build_solvers(s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets),
        s.horizon,
        dim=s.model.dim,
        options=ADMMOptions(adaptive_rho=True, max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6),
    ).solve(s.x0, references=s.references, offsets=s.offsets)

    assert fixed.converged and adaptive.converged
    assert np.allclose(fixed.inputs, adaptive.inputs, atol=1e-3)
    assert np.allclose(fixed.inputs, centralized, atol=1e-3)
    assert adaptive.iterations <= fixed.iterations + 20


def test_warm_start_reduces_iterations(four_agent_setup):
    """Re-solving from a shifted previous solution takes strictly fewer iterations.

    Perturb ``x0`` slightly between the two solves so the test is not trivially satisfied
    by an unchanged problem.
    """
    s = four_agent_setup
    admm = ConsensusADMM(
        s.graph,
        s.solvers,
        s.horizon,
        dim=s.model.dim,
        options=ADMMOptions(max_iterations=2000),
    )
    cold = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert cold.converged

    x0_perturbed = s.x0 + 0.05 * np.random.default_rng(0).normal(size=s.x0.shape)
    warm = admm.solve(
        x0_perturbed, references=s.references, offsets=s.offsets, initial_guess=cold.shifted()
    )
    assert warm.converged
    assert warm.iterations < cold.iterations


def test_solution_invariant_to_agent_relabelling(four_agent_setup):
    """Permuting agent ids permutes the solution and leaves the iteration count alone.

    Guards against any accidental dependence on iteration order in ``_z_update`` — the
    single most likely way to break the "no global information" invariant.
    """
    s = four_agent_setup
    perm = np.array([2, 0, 3, 1])
    options = ADMMOptions(max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6)

    original = ConsensusADMM(
        s.graph,
        _build_solvers(s.graph, s.model, s.horizon, s.weights, s.limits, s.offsets),
        s.horizon,
        dim=s.model.dim,
        options=options,
    ).solve(s.x0, references=s.references, offsets=s.offsets)
    assert original.converged

    relabelled_edges = [(int(perm[i]), int(perm[j])) for i, j in s.graph.edges]
    graph_perm = CommunicationGraph(4, relabelled_edges)
    x0_perm = np.empty_like(s.x0)
    x0_perm[perm] = s.x0
    references_perm = {int(perm[i]): s.references[i] for i in range(4)}
    offsets_perm = {
        int(perm[i]): {int(perm[j]): d for j, d in neighbors.items()}
        for i, neighbors in s.offsets.items()
    }
    solvers_perm = _build_solvers(graph_perm, s.model, s.horizon, s.weights, s.limits, offsets_perm)

    relabelled = ConsensusADMM(
        graph_perm,
        solvers_perm,
        s.horizon,
        dim=s.model.dim,
        options=options,
    ).solve(x0_perm, references=references_perm, offsets=offsets_perm)
    assert relabelled.converged
    assert np.allclose(relabelled.inputs[perm], original.inputs, atol=1e-3)
    assert relabelled.iterations == pytest.approx(original.iterations, abs=5)


# --------------------------------------------------------------------------- robustness


def test_input_limits_respected(four_agent_setup):
    """No returned input exceeds ``u_max`` beyond solver tolerance, from an initial
    condition aggressive enough to actually saturate."""
    s = four_agent_setup
    aggressive = np.zeros_like(s.x0)
    aggressive[:, : s.model.dim] = 50.0
    options = ADMMOptions(max_iterations=2000, eps_abs=1e-6, eps_rel=1e-6)
    admm = ConsensusADMM(s.graph, s.solvers, s.horizon, dim=s.model.dim, options=options)
    result = admm.solve(aggressive, references=s.references, offsets=s.offsets)
    assert result.converged
    assert np.max(np.abs(result.inputs)) <= s.limits.u_max + 1e-5
    assert np.max(np.abs(result.inputs)) > 0.5 * s.limits.u_max


def test_max_iterations_reported_not_raised(four_agent_setup):
    """Hitting the iteration cap sets ``converged = False`` and returns the last iterate.

    An MPC loop must never crash because one step ran out of iterations; it degrades.
    """
    s = four_agent_setup
    options = ADMMOptions(max_iterations=3, eps_abs=1e-12, eps_rel=1e-12)
    admm = ConsensusADMM(s.graph, s.solvers, s.horizon, dim=s.model.dim, options=options)
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert not result.converged
    assert result.iterations == 3
    assert np.all(np.isfinite(result.inputs))
    assert len(result.history.primal_residual) == 3


def test_packet_loss_degrades_gracefully(four_agent_setup):
    """Under Bernoulli loss the residual stays bounded and no NaN appears.

    Do not assert convergence to tolerance: the whole point of this experiment is that the
    guarantee is lost. Assert boundedness and that the final residual degrades with
    ``loss_prob`` across seeds.
    """
    s = four_agent_setup
    finals: dict[float, list[float]] = {}
    for loss_prob in (0.0, 0.1, 0.3):
        finals[loss_prob] = []
        for seed in range(2):
            channel = LossyChannel(s.graph, loss_prob=loss_prob, rng=seed)
            admm = ConsensusADMM(
                s.graph,
                s.solvers,
                s.horizon,
                dim=s.model.dim,
                options=ADMMOptions(max_iterations=300),
                channel=channel,
            )
            result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
            primal = np.asarray(result.history.primal_residual, dtype=float)
            assert np.all(np.isfinite(primal))
            assert primal[-1] < 1e3
            finals[loss_prob].append(primal[-1])
    medians = {p: float(np.median(v)) for p, v in finals.items()}
    assert medians[0.0] < medians[0.3]


@pytest.mark.slow
def test_scales_to_eight_agents(model, horizon):
    """8 agents on a random connected graph converge within the iteration cap."""
    graph = CommunicationGraph.random_connected(8, rng=0)
    weights = AgentCostWeights()
    limits = AgentLimits()
    offsets = {i: {j: np.zeros(model.dim) for j in graph.neighbors(i)} for i in range(8)}
    solvers = _build_solvers(graph, model, horizon, weights, limits, offsets)
    rng = np.random.default_rng(3)
    x0 = _random_x0(rng, 8, model)
    references = _zero_references(8, horizon, model.dim)
    admm = ConsensusADMM(
        graph,
        solvers,
        horizon,
        dim=model.dim,
        options=ADMMOptions(max_iterations=2000),
    )
    result = admm.solve(x0, references=references, offsets=offsets)
    assert result.converged
    assert np.all(np.isfinite(result.inputs))


def test_no_global_information_used(four_agent_setup, monkeypatch):
    """Structural guard: an agent's update must not read another agent's private state.

    Wrap each solver's ``solve`` and assert the ``LocalProblemData`` it receives has keys
    only within that agent's closed neighborhood. Cheap to write, and it is the property
    the entire repo claims.
    """
    s = four_agent_setup
    neighborhoods = {i: set(s.graph.closed_neighborhood(i)) for i in range(4)}

    for i, solver in s.solvers.items():
        original_solve = solver.solve

        def wrapped(data, *, _i=i, _original=original_solve):
            assert (
                set(data.z) <= neighborhoods[_i]
            ), f"agent {_i} received z for {set(data.z) - neighborhoods[_i]}"
            assert (
                set(data.lam) <= neighborhoods[_i]
            ), f"agent {_i} received lam for {set(data.lam) - neighborhoods[_i]}"
            return _original(data)

        monkeypatch.setattr(solver, "solve", wrapped)

    admm = ConsensusADMM(
        s.graph, s.solvers, s.horizon, dim=s.model.dim, options=ADMMOptions(max_iterations=50)
    )
    result = admm.solve(s.x0, references=s.references, offsets=s.offsets)
    assert result.iterations >= 1
