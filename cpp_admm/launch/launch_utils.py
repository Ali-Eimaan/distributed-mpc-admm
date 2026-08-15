# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers imported by both launch files.

``build_topology`` and ``build_offsets`` MUST live here, not copied into each launch file:
two divergent copies of the topology definition is the one bug this pair of files cannot
afford, and it is the default outcome of copy-pasting the first file into the second.

``build_topology`` produces edge lists identical to
``python/distributed_mpc_admm/communication_graph.py`` -- same node numbering, same cycle
orientation -- because the C++ and Python demos are compared against each other and a
differently-numbered cycle makes that comparison meaningless while looking fine.

``build_offsets`` mirrors the factories in
``python/distributed_mpc_admm/formation_constraints.py`` exactly (including mean-centring).
"""

from __future__ import annotations

import math


def build_topology(name: str, n_agents: int) -> list[tuple[int, int]]:
    """Edge list for a named topology, matching ``CommunicationGraph`` verbatim."""
    name = name.strip().lower()
    if name == "cycle":
        # Ring 0-1-...-(N-1)-0.
        if n_agents == 1:
            return []
        return [(i, (i + 1) % n_agents) for i in range(n_agents)]
    if name == "complete":
        return [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    if name == "path":
        return [(i, i + 1) for i in range(n_agents - 1)]
    if name == "star":
        return [(0, i) for i in range(n_agents) if i != 0]
    raise ValueError(f"unknown topology: {name!r}")


def _mean_centre(offsets: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Subtract the centroid so the formation anchor is the centre of mass."""
    if not offsets:
        return offsets
    n = len(offsets)
    mx = sum(o[0] for o in offsets) / n
    my = sum(o[1] for o in offsets) / n
    return [(o[0] - mx, o[1] - my) for o in offsets]


def build_offsets(
    name: str, n_agents: int, scale: float = 1.0
) -> dict[int, tuple[float, float]]:
    """Per-agent anchor offsets, mirroring ``FormationSpec`` factories.

    ``square`` -> ``regular_polygon(n_agents, scale)`` (a square for ``n_agents == 4``);
    ``line``   -> ``line(n_agents, scale)``;
    ``v``      -> ``v_shape(n_agents, scale)``.
    """
    name = name.strip().lower()

    if name == "square":
        angles = [2 * math.pi * i / n_agents for i in range(n_agents)]
        raw = [(scale * math.cos(a), scale * math.sin(a)) for a in angles]
        centred = _mean_centre(raw)
    elif name == "line":
        positions = [(i - (n_agents - 1) / 2) * scale for i in range(n_agents)]
        raw = [(p, 0.0) for p in positions]
        centred = _mean_centre(raw)
    elif name == "v":
        half_angle = math.pi / 6
        cos_a = math.cos(half_angle)
        sin_a = math.sin(half_angle)
        raw: list[tuple[float, float]] = [(0.0, 0.0)]
        for k in range(1, n_agents):
            slot = (k + 1) // 2
            side = 1.0 if k % 2 == 1 else -1.0
            dist = slot * scale
            raw.append((-dist * cos_a, side * dist * sin_a))
        centred = _mean_centre(raw)
    else:
        raise ValueError(f"unknown formation: {name!r}")

    return {i: centred[i] for i in range(n_agents)}


def flatten_neighbor_offsets(
    agent: int, neighbors: list[int], offsets: dict[int, tuple[float, float]]
) -> list[float]:
    """Flatten ``d_ij = o_i - o_j`` into ``[j, dx, dy, ...]`` triple list.

    Sorted by neighbour id so the parameter dump is diffable.
    """
    o_i = offsets.get(agent, (0.0, 0.0))
    flat: list[float] = []
    for j in sorted(neighbors):
        o_j = offsets.get(j, (0.0, 0.0))
        flat.extend((float(j), o_i[0] - o_j[0], o_i[1] - o_j[1]))
    return flat


def schedule_edges(
    name: str, n_agents: int, index: int, loss_prob: float = 0.0, seed: int = 0
) -> list[tuple[int, int]]:
    """Edge list for switch number ``index`` of the named time-varying schedule.

    ``alternate``
        Cycle and path, back and forth. Connectivity stays positive throughout.

    ``split_merge``
        Cycle -> two disconnected pairs -> cycle. During the split the two components
        hold their own shapes and drift apart; the merge is the set-valued reset event.

    ``random_failure``
        Each edge of the base cycle independently present with probability
        ``1 - loss_prob``, reseeded per switch. Sometimes disconnected, which is the point.

    Deterministic given ``seed`` (random failures are drawn from ``random.Random(seed + index)``).
    """
    import random

    name = name.strip().lower()
    cycle = build_topology("cycle", n_agents)

    if name == "alternate":
        return cycle if index % 2 == 0 else build_topology("path", n_agents)

    if name == "split_merge":
        phase = index % 3
        if phase == 1:
            # Two disconnected pairs (agents (0,1), (2,3), ...); an odd agent is isolated.
            return [(i, i + 1) for i in range(0, n_agents - 1, 2)]
        return cycle

    if name == "random_failure":
        rng = random.Random(seed + index)
        return [(i, j) for (i, j) in cycle if rng.random() < (1.0 - loss_prob)]

    raise ValueError(f"unknown schedule: {name!r}")
