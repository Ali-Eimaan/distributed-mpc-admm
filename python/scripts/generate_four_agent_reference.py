# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate ``cpp_admm/test/data/four_agent_reference.json``.

This is the *only* supported way to produce the parity fixture for
``test/test_admm_kernel.cpp``. Regenerating it is a deliberate act: if a change makes the
parity test fail, fix the C++ or justify the export -- never regenerate the fixture to
make a red test go green.

The problem is fixed and fully deterministic:

* 4 agents on ``CommunicationGraph.cycle(4)`` (ring topology)
* ``DoubleIntegrator(dt=0.1, dim=2)``, horizon ``T = 10``
* default ``AgentCostWeights`` / ``AgentLimits``
* rendezvous formation (all offsets zero), every agent tracks the origin (reference = 0)
* ``ADMMOptions(rho=1.0, alpha=1.0, adaptive_rho=False)``

The inner OSQP solver is configured to match the C++ ``QpSettings`` exactly (polish off,
adaptive rho off, scaling off, tight tolerance) so the recorded iterates are comparable to
the C++ kernel at ~1e-8 rather than at the looser cvxpy defaults.

Run from the repository root::

    /home/eiman/Documents/tools/xenvs/mpc-admm/bin/python \
        python/scripts/generate_four_agent_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from distributed_mpc_admm.communication_graph import CommunicationGraph
from distributed_mpc_admm.consensus_admm import ADMMOptions, ConsensusADMM
from distributed_mpc_admm.per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    CvxpyAgentSolver,
    DoubleIntegrator,
    LocalProblemData,
)

# --------------------------------------------------------------------------- problem

N_AGENTS = 4
HORIZON = 10
DIM = 2
DT = 0.1
N_ITERATES = 20  # number of per-iteration snapshots recorded for the parity test

graph = CommunicationGraph.cycle(N_AGENTS)
model = DoubleIntegrator(dt=DT, dim=DIM)
weights = AgentCostWeights()
limits = AgentLimits()

rng = np.random.default_rng(0)
x0 = rng.normal(size=(N_AGENTS, model.n_states))
x0[:, DIM:] *= 0.5  # gentler velocities so v_max stays feasible

offsets = {i: {j: np.zeros(DIM) for j in graph.neighbors(i)} for i in range(N_AGENTS)}
references = {i: np.zeros((HORIZON, DIM)) for i in range(N_AGENTS)}

# Mirror the C++ QpSettings: polish off, adaptive rho off, scaling off, tight tolerance.
solver_options = {
    "eps_abs": 1e-8,
    "eps_rel": 1e-8,
    "polish": False,
    "adaptive_rho": False,
    "scaling": 0,
    "max_iter": 100000,
    "verbose": False,
}

solvers = {
    i: CvxpyAgentSolver(
        agent_id=i,
        horizon=HORIZON,
        model=model,
        limits=limits,
        weights=weights,
        neighborhood=tuple(graph.closed_neighborhood(i)),
        offsets=dict(offsets[i]),
        solver_options=solver_options,
    )
    for i in range(N_AGENTS)
}

options = ADMMOptions(
    rho=1.0,
    alpha=1.0,
    adaptive_rho=False,
    max_iterations=2000,
    eps_abs=1e-8,
    eps_rel=1e-8,
)

# Drive the consensus loop manually (same phase order as ConsensusADMM.solve) so we can
# snapshot the exact (z, lam) fed into each x-update and the (U, y) it returns.
admm = ConsensusADMM(graph, solvers, HORIZON, dim=DIM, options=options)
admm._z = {j: np.zeros((HORIZON, DIM)) for j in range(N_AGENTS)}
admm._z_view = {i: dict(admm._z) for i in range(N_AGENTS)}
admm._y = {
    i: {j: np.zeros((HORIZON, DIM)) for j in graph.closed_neighborhood(i)} for i in range(N_AGENTS)
}
admm._lam = {
    i: {j: np.zeros((HORIZON, DIM)) for j in graph.closed_neighborhood(i)} for i in range(N_AGENTS)
}
admm._z_last_known = {i: {} for i in range(N_AGENTS)}

rho = options.rho
iterates: list[dict] = []
last_inputs: list[list[float]] | None = None
converged = False

for iteration in range(options.max_iterations):
    snapshot: dict = {"iteration": iteration, "agents": []}
    for i in range(N_AGENTS):
        neighborhood = tuple(graph.closed_neighborhood(i))
        data = LocalProblemData(
            agent_id=i,
            horizon=HORIZON,
            model=model,
            limits=limits,
            weights=weights,
            neighborhood=neighborhood,
            offsets=dict(offsets[i]),
            x0=np.asarray(x0[i], dtype=np.float64),
            reference=np.asarray(references[i], dtype=np.float64),
            u_prev=None,
            rho=rho,
            z={j: admm._z_view[i][j] for j in neighborhood},
            lam={j: admm._lam[i][j] for j in neighborhood},
        )
        solution = solvers[i].solve(data)
        snapshot["agents"].append(
            {
                "z_in": {str(j): data.z[j].ravel().tolist() for j in neighborhood},
                "lam_in": {str(j): data.lam[j].ravel().tolist() for j in neighborhood},
                "U": solution.inputs.ravel().tolist(),
                "y": {str(j): solution.copies[j].ravel().tolist() for j in neighborhood},
            }
        )
        admm._y[i] = {j: solution.copies[j].copy() for j in neighborhood}

    # The last x-update's inputs are exactly what ConsensusADMM.solve() returns as
    # `inputs` (stack(latest_solutions[i].inputs)); record them for the optimum test.
    last_inputs = [snapshot["agents"][i]["U"] for i in range(N_AGENTS)]

    if iteration < N_ITERATES:
        iterates.append(snapshot)

    # --- relax / z-update / broadcast / dual update (identical to solve()) ---------
    y_hat = admm._relax(admm._y, admm._z) if options.alpha != 1.0 else admm._y
    z_prev = {j: admm._z[j].copy() for j in range(N_AGENTS)}
    admm._z = admm._z_update(y_hat, admm._lam, iteration)
    admm._z_view = admm._broadcast_consensus(admm._z, iteration)
    admm._lam = admm._dual_update(admm._lam, y_hat, admm._z_view)

    primal_residual, dual_residual = admm._residuals(admm._y, admm._z, z_prev, rho)
    eps_primal, eps_dual = admm._tolerances(admm._y, admm._z, admm._lam, rho)
    converged = primal_residual <= eps_primal and dual_residual <= eps_dual
    if converged:
        break

# `final_inputs` is the U from the final x-update (ConsensusADMM.solve()'s `inputs`),
# NOT a re-solve at the final (z, lam) -- the two differ by one dual update.
final_inputs = last_inputs

# Record the final consensus/dual views as well, for fixed-point parity checks.
final_z = {i: admm._z_view[i][i].ravel().tolist() for i in range(N_AGENTS)}
final_lam = {
    i: {str(j): admm._lam[i][j].ravel().tolist() for j in graph.closed_neighborhood(i)}
    for i in range(N_AGENTS)
}

payload = {
    "dt": DT,
    "dim": DIM,
    "horizon": HORIZON,
    "n_agents": N_AGENTS,
    "rho": rho,
    "weights": {
        "q_position": weights.q_position,
        "q_velocity": weights.q_velocity,
        "r_input": weights.r_input,
        "r_rate": weights.r_rate,
        "p_terminal": weights.p_terminal,
        "w_formation": weights.w_formation,
    },
    "limits": {"u_max": limits.u_max, "v_max": limits.v_max},
    "x0": x0.tolist(),
    "reference": [references[i].ravel().tolist() for i in range(N_AGENTS)],
    "neighbors": [list(graph.neighbors(i)) for i in range(N_AGENTS)],
    "iterates": iterates,
    "final_inputs": final_inputs,
    "final_z": final_z,
    "final_lam": final_lam,
    "converged": converged,
    "iterations_run": iteration + 1,
}

out = (
    Path(__file__).resolve().parents[2] / "cpp_admm" / "test" / "data" / "four_agent_reference.json"
)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")

print(f"wrote {out}")
print(f"converged={converged} iterations_run={iteration + 1} final_rho={rho}")
