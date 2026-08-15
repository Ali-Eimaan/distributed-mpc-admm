# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Formation geometry, leader-follower behaviour, and closed-loop convergence.

Where ``test_admm_convergence.py`` tests the optimiser, this file tests that the
optimiser is being pointed at the right problem: that the geometry helpers are internally
consistent, and that a closed-loop run actually achieves and holds the formation.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from distributed_mpc_admm.communication_graph import CommunicationGraph, TimeVaryingGraph
from distributed_mpc_admm.consensus_admm import (
    ADMMOptions,
    DistributedMPC,
    MPCOptions,
    SimulationLog,
)
from distributed_mpc_admm.formation_constraints import (
    FormationSpec,
    LeaderFollowerSpec,
    formation_error,
    settling_step,
)
from distributed_mpc_admm.per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    DoubleIntegrator,
)

# --------------------------------------------------------------------------- helpers


def _builtin_formations() -> list[FormationSpec]:
    """A representative spread of the factory-built formations."""
    return [
        FormationSpec.regular_polygon(4, radius=1.0),
        FormationSpec.regular_polygon(5, radius=1.5),
        FormationSpec.line(4, spacing=1.0),
        FormationSpec.line(5, spacing=1.0, heading=0.4),
        FormationSpec.v_shape(5, spacing=1.0),
        FormationSpec.grid(2, 3, spacing=1.0),
        FormationSpec.grid(3, 3, spacing=1.5),
    ]


def _scattered_x0(rng, n_agents: int, model: DoubleIntegrator) -> np.ndarray:
    """Gentle, deterministic scatter so the initial condition stays feasible."""
    x0 = rng.normal(scale=0.4, size=(n_agents, model.n_states))
    x0[:, model.dim :] *= 0.2
    x0[:, : model.dim] += np.array([0.5, -0.3]) * np.arange(n_agents)[:, None]
    return x0


# --------------------------------------------------------------------------- geometry


def test_relative_offsets_are_antisymmetric():
    """``d_ij == -d_ji`` for every pair in every built-in formation."""
    for spec in _builtin_formations():
        for i in range(spec.n_agents):
            for j in range(spec.n_agents):
                if i == j:
                    continue
                assert np.allclose(spec.relative_offset(i, j), -spec.relative_offset(j, i))


def test_offsets_are_mean_centred():
    """Factory-built formations have zero-mean offsets, so the anchor is the centroid."""
    for spec in _builtin_formations():
        assert np.allclose(spec.anchor_offsets().mean(axis=0), 0.0, atol=1e-12)


def test_relative_offsets_invariant_to_translation():
    """Translating every offset by a constant leaves all ``d_ij`` unchanged.

    This is why the encoding is implementable with only relative measurements.
    """
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    shifted = FormationSpec(offsets=spec.offsets + np.array([3.0, -2.0]), graph=spec.graph)
    for i in range(spec.n_agents):
        for j in range(spec.n_agents):
            if i == j:
                continue
            assert np.allclose(spec.relative_offset(i, j), shifted.relative_offset(i, j))


def test_rotation_rotates_all_offsets():
    """Setting ``rotation`` applies one rigid rotation; pairwise distances are preserved."""
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    rotated = FormationSpec(offsets=spec.offsets, graph=spec.graph, rotation=np.deg2rad(37.0))
    base = spec.anchor_offsets()
    rot = rotated.anchor_offsets()
    for i in range(spec.n_agents):
        for j in range(i + 1, spec.n_agents):
            assert np.isclose(np.linalg.norm(base[i] - base[j]), np.linalg.norm(rot[i] - rot[j]))


@pytest.mark.parametrize("n_agents", [4, 5, 6, 8])
def test_polygon_side_lengths_equal(n_agents):
    """Consecutive agents in a regular polygon are equidistant."""
    spec = FormationSpec.regular_polygon(n_agents, radius=1.0)
    offsets = spec.anchor_offsets()
    sides = np.array(
        [np.linalg.norm(offsets[(k + 1) % n_agents] - offsets[k]) for k in range(n_agents)]
    )
    assert np.allclose(sides, sides[0])
    assert sides[0] > 0.0


def test_formation_error_zero_at_target():
    """``formation_error(spec.target_positions(anchor), spec)`` is zero for any anchor."""
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    for anchor in (np.zeros(2), np.array([1.5, -2.0])):
        err = formation_error(spec.target_positions(anchor), spec)
        assert err.edge_rms < 1e-12
        assert err.edge_max < 1e-12


def test_formation_error_positive_when_perturbed():
    """Displacing one agent raises ``edge_max`` and shows up on that agent's edges only."""
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    positions = spec.target_positions(np.zeros(2))
    positions[0] += np.array([0.3, -0.1])
    err = formation_error(positions, spec)
    assert err.edge_rms > 0.0
    assert err.edge_max > 0.0
    assert err.per_edge[(0, 1)] > 0.0
    assert err.per_edge[(0, 3)] > 0.0
    assert err.per_edge[(1, 2)] == 0.0
    assert err.per_edge[(2, 3)] == 0.0


def test_centroid_error_separates_shape_from_drift():
    """Translating the whole formation leaves ``edge_rms`` at zero but moves
    ``centroid_error`` — the two failure modes must not be conflated."""
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    anchor = np.zeros(2)
    positions = spec.target_positions(anchor)
    translation = np.array([2.0, 1.0])
    err = formation_error(positions + translation, spec, anchor_reference=anchor)
    assert err.edge_rms < 1e-12
    assert err.edge_max < 1e-12
    assert np.isclose(err.centroid_error, np.linalg.norm(translation))


# --------------------------------------------------------------------------- rigidity


def test_complete_graph_formation_is_rigid():
    """A 4-agent complete graph in 2D is infinitesimally rigid."""
    spec = FormationSpec.regular_polygon(4, radius=1.0, graph=CommunicationGraph.complete(4))
    assert spec.is_infinitesimally_rigid()


def test_path_graph_formation_is_not_rigid():
    """A 4-agent path is a flexible framework: rank deficiency exceeds the 3 trivial modes."""
    spec = FormationSpec.regular_polygon(4, radius=1.0, graph=CommunicationGraph.path(4))
    assert not spec.is_infinitesimally_rigid()
    rank = np.linalg.matrix_rank(spec.rigidity_matrix())
    trivial = spec.dim * (spec.dim + 1) // 2  # 3 trivial motions in 2D
    assert (spec.dim * spec.n_agents - rank) > trivial


def test_rigidity_matrix_shape():
    """``R(p)`` is ``(|E|, N*dim)`` and its null space contains the trivial motions
    (two translations and one infinitesimal rotation in 2D)."""
    spec = FormationSpec.regular_polygon(4, radius=1.0)
    rigidity = spec.rigidity_matrix()
    assert rigidity.shape == (len(spec.graph.edges), spec.n_agents * spec.dim)

    # Translations: every agent moves by the same vector.
    for direction in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        assert np.linalg.norm(rigidity @ np.tile(direction, spec.n_agents)) < 1e-10

    # Infinitesimal rotation about the origin: v_i = omega x p_i = [-p_y, p_x].
    rotation_motion = np.zeros(spec.n_agents * spec.dim)
    for i, p in enumerate(spec.anchor_offsets()):
        rotation_motion[i * spec.dim : (i + 1) * spec.dim] = np.array([-p[1], p[0]])
    assert np.linalg.norm(rigidity @ rotation_motion) < 1e-10


# --------------------------------------------------------------------------- validation


def test_formation_edge_missing_from_comm_graph_raises():
    """``validate_against`` rejects a formation edge with no communication link.

    Silently dropping the term is the alternative and it produces a formation that
    converges to the wrong shape with no error message.
    """
    formation = FormationSpec.regular_polygon(4, radius=1.0)  # cycle: 4 edges
    comm = CommunicationGraph.path(4)  # missing edge (0, 3)
    with pytest.raises(ValueError):
        formation.validate_against(comm)


def test_unreachable_follower_raises():
    """A follower with no path to any leader is rejected by
    ``LeaderFollowerSpec.validate_against``."""
    graph = CommunicationGraph(4, [(0, 1), (2, 3)])  # two disconnected components
    spec = LeaderFollowerSpec(leaders=(0,), reference=np.zeros((5, 2)))
    with pytest.raises(ValueError):
        spec.validate_against(graph)


# --------------------------------------------------------------------------- closed loop


@pytest.fixture
def square_formation_setup():
    """4 agents, square formation, cycle graph, scattered initial positions."""
    model = DoubleIntegrator(dt=0.1, dim=2)
    formation = FormationSpec.regular_polygon(4, radius=1.0)
    graph = CommunicationGraph.cycle(4)
    limits = AgentLimits()
    weights = AgentCostWeights(q_position=0.0, p_terminal=0.0)
    admm_options = ADMMOptions(max_iterations=300, eps_abs=2e-2, eps_rel=2e-2)
    mpc_options = MPCOptions(horizon=10, dt=0.1, n_steps=35)
    x0 = _scattered_x0(np.random.default_rng(0), 4, model)
    return types.SimpleNamespace(
        model=model,
        formation=formation,
        graph=graph,
        limits=limits,
        weights=weights,
        admm_options=admm_options,
        mpc_options=mpc_options,
        x0=x0,
    )


def test_agents_reach_formation(square_formation_setup):
    """Final edge-RMS below 5 cm and :func:`settling_step` returns a finite index.

    Assert on the *held* error over the last 20 percent of the run, not just the final
    sample, so an oscillating formation cannot pass.
    """
    s = square_formation_setup
    ctrl = DistributedMPC(
        s.model, s.graph, s.formation, s.limits, s.weights, s.admm_options, s.mpc_options
    )
    log = ctrl.run(s.x0)
    held = log.formation_error[-int(0.2 * s.mpc_options.n_steps) :]
    assert held.max() <= 0.05
    assert settling_step(log.formation_error, tolerance=0.05) is not None


def test_rendezvous_converges_to_common_point(square_formation_setup):
    """With all offsets zero, the inter-agent spread decays below tolerance."""
    s = square_formation_setup
    formation = FormationSpec.rendezvous(4, s.graph)
    ctrl = DistributedMPC(
        s.model, s.graph, formation, s.limits, s.weights, s.admm_options, s.mpc_options
    )
    log = ctrl.run(s.x0)
    spread = np.array(
        [np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1).max() for p in log.positions]
    )
    assert spread[-1] < 0.05
    assert spread[-1] < 0.5 * spread[0]  # genuinely decayed, not just small from the start


def test_formation_holds_while_tracking(square_formation_setup):
    """With a moving leader reference, shape error stays bounded while the centroid tracks.

    The centroid should lag the reference by a bounded amount (this is a finite-horizon
    tracker, not an integrator) — assert bounded lag, not zero lag.
    """
    s = square_formation_setup
    t = 0.1 * np.arange(60)
    reference = np.column_stack([2.0 * np.cos(0.15 * t), 2.0 * np.sin(0.15 * t)])
    leader_follower = LeaderFollowerSpec(leaders=(0,), reference=reference)
    weights = AgentCostWeights(q_position=1.0, p_terminal=5.0)
    ctrl = DistributedMPC(
        s.model,
        s.graph,
        s.formation,
        s.limits,
        weights,
        s.admm_options,
        s.mpc_options,
        leader_follower=leader_follower,
    )
    log = ctrl.run(s.x0)
    held = log.formation_error[-int(0.2 * s.mpc_options.n_steps) :]
    assert held.max() <= 0.05  # shape stays bounded while the whole body moves
    centroid = log.positions.mean(axis=1)
    lag = np.linalg.norm(centroid - reference[: len(centroid)], axis=1)
    assert lag.max() < 2.0  # bounded, nonzero lag


def test_leader_only_tracks_reference(square_formation_setup):
    """Followers have zero tracking weight yet still move — pulled purely by edge costs.

    This is the test that proves the formation coupling, not the reference, is doing the
    work.
    """
    s = square_formation_setup
    t = 0.1 * np.arange(60)
    reference = np.column_stack([2.0 * np.cos(0.15 * t), 2.0 * np.sin(0.15 * t)])
    leader_follower = LeaderFollowerSpec(leaders=(0,), reference=reference)
    weights = AgentCostWeights(q_position=1.0, p_terminal=5.0)
    ctrl = DistributedMPC(
        s.model,
        s.graph,
        s.formation,
        s.limits,
        weights,
        s.admm_options,
        s.mpc_options,
        leader_follower=leader_follower,
    )
    log = ctrl.run(s.x0)
    followers_moved = np.linalg.norm(log.positions[-1, 1:] - log.positions[0, 1:], axis=1)
    assert np.all(followers_moved > 0.1)  # edge costs pulled them, not the reference
    assert log.formation_error[-1] < 0.05  # and the shape actually formed


def test_formation_recovers_after_topology_switch():
    """Dropping an edge mid-run causes a transient that decays back below tolerance.

    Use a :class:`TimeVaryingGraph` that switches from a cycle to a path at the halfway
    point. Assert the error spikes at the switch step and settles again — both halves
    matter; a run with no spike means the switch was not actually applied.
    """
    model = DoubleIntegrator(dt=0.1, dim=2)
    cycle = CommunicationGraph.cycle(4)
    path = CommunicationGraph.path(4)
    formation = FormationSpec.regular_polygon(4, radius=1.0)
    limits = AgentLimits()
    weights = AgentCostWeights(q_position=0.0, p_terminal=0.0)
    admm_options = ADMMOptions(max_iterations=300, eps_abs=2e-2, eps_rel=2e-2)
    mpc_options = MPCOptions(horizon=10, dt=0.1, n_steps=35)
    x0 = _scattered_x0(np.random.default_rng(0), 4, model)

    half = mpc_options.n_steps // 2
    schedule = TimeVaryingGraph.switching([cycle, path], dwell_time=half, mode="hold")
    ctrl = DistributedMPC(model, schedule, formation, limits, weights, admm_options, mpc_options)
    log = ctrl.run(x0)

    err = log.formation_error
    assert log.metadata["switch_steps"] == (half,)
    assert err[half:].max() > err[half - 1]  # a visible transient, not a silent no-op
    held = err[-int(0.2 * mpc_options.n_steps) :]
    assert held.max() <= 0.05
    assert settling_step(err, tolerance=0.05) is not None


def test_switch_to_disconnected_graph_drifts():
    """When the topology splits, the two components hold their own shapes and drift apart.

    The expected-failure documentation test: it pins the behaviour the AHTD analysis in the
    thesis is meant to explain.
    """
    model = DoubleIntegrator(dt=0.1, dim=2)
    cycle = CommunicationGraph.cycle(4)
    split = CommunicationGraph(4, [(0, 1), (2, 3)])
    formation = FormationSpec.regular_polygon(4, radius=1.0)
    limits = AgentLimits()
    weights = AgentCostWeights(q_position=0.0, p_terminal=0.0)
    admm_options = ADMMOptions(max_iterations=300, eps_abs=2e-2, eps_rel=2e-2)
    mpc_options = MPCOptions(horizon=10, dt=0.1, n_steps=20)

    # Two clearly separated halves so the split is observable.
    x0 = np.zeros((4, model.n_states))
    x0[:, : model.dim] = [[-1.5, 0.2], [-1.5, -0.2], [1.5, 0.2], [1.5, -0.2]]

    schedule = TimeVaryingGraph.switching([cycle, split], dwell_time=3, mode="hold")
    ctrl = DistributedMPC(model, schedule, formation, limits, weights, admm_options, mpc_options)
    log = ctrl.run(x0)

    err = formation_error(log.positions[-1], formation)
    # Each component holds its own (single) edge...
    assert err.per_edge[(0, 1)] < 0.05
    assert err.per_edge[(2, 3)] < 0.05
    # ...while the edges spanning the split drift away from their targets.
    assert err.per_edge[(0, 3)] > 0.05
    assert err.per_edge[(1, 2)] > 0.05
    assert log.formation_error[-1] > 0.05  # never settles as one formation


@pytest.mark.slow
def test_eight_agent_formation_closed_loop():
    """8 agents in a two-row grid reach formation within the mission horizon."""
    model = DoubleIntegrator(dt=0.1, dim=2)
    formation = FormationSpec.grid(2, 4, spacing=1.5)
    graph = formation.graph
    limits = AgentLimits()
    weights = AgentCostWeights(q_position=0.0, p_terminal=0.0)
    admm_options = ADMMOptions(max_iterations=500, eps_abs=2e-2, eps_rel=2e-2)
    mpc_options = MPCOptions(horizon=10, dt=0.1, n_steps=60)
    x0 = _scattered_x0(np.random.default_rng(0), 8, model)

    ctrl = DistributedMPC(model, graph, formation, limits, weights, admm_options, mpc_options)
    log = ctrl.run(x0)
    held = log.formation_error[-int(0.2 * mpc_options.n_steps) :]
    assert held.max() <= 0.05
    assert settling_step(log.formation_error, tolerance=0.05) is not None


def test_simulation_log_roundtrip(square_formation_setup, tmp_path):
    """``SimulationLog.save``/``load`` preserves every array exactly."""
    s = square_formation_setup
    ctrl = DistributedMPC(
        s.model, s.graph, s.formation, s.limits, s.weights, s.admm_options, s.mpc_options
    )
    log = ctrl.run(s.x0)
    path = tmp_path / "simulation.npz"
    log.save(str(path))
    loaded = SimulationLog.load(str(path))

    np.testing.assert_allclose(loaded.time, log.time)
    np.testing.assert_allclose(loaded.states, log.states)
    np.testing.assert_allclose(loaded.inputs, log.inputs)
    np.testing.assert_allclose(loaded.formation_error, log.formation_error)
    np.testing.assert_array_equal(loaded.admm_iterations, log.admm_iterations)
    np.testing.assert_array_equal(loaded.admm_converged, log.admm_converged)
    np.testing.assert_allclose(loaded.solve_times, log.solve_times)
    assert loaded.graphs == log.graphs

    assert len(loaded.predictions) == len(log.predictions)
    for expected, actual in zip(log.predictions, loaded.predictions, strict=True):
        np.testing.assert_allclose(actual, expected)

    assert len(loaded.histories) == len(log.histories)
    for expected, actual in zip(log.histories, loaded.histories, strict=True):
        for name in ("primal_residual", "dual_residual", "eps_primal", "eps_dual", "rho"):
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
            )
