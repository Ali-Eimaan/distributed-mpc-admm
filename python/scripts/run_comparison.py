# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible driver for ``docs/COMPARISON_VS_DUAL_DECOMP.md``.

Runs consensus ADMM and dual decomposition on *identical* problem instances (same graph,
same seeds, same initial-condition distribution, same tolerances) and prints the measured
numbers that populate section 5 of the comparison document.

Protocol (do not deviate)
-------------------------
* tolerance: ``eps_abs = eps_rel = 1e-4`` (dual decomposition does not reach the ``1e-6``
  used in notebook 05 within a tractable iteration budget -- that gap is part of the result)
* max iterations: 2000 per solve
* metric: ``iterations`` -- the iteration at which each method's own convergence test
  declares success (``r_k <= eps_pri`` **and** ``s_k <= eps_dual``, ``eps_abs =
  eps_rel = 1e-4``). The primal tolerance is identical for both methods; the dual
  tolerance differs only by the documented ``rho``/``nu`` scaling, so the two stop on
  the same residual/tolerance test (see docs/COMPARISON_VS_DUAL_DECOMP.md section 3).
* ADMM rho and dual-decomposition step size are each swept on a coarse grid and the best
  (median over seeds) is used -- i.e. each method is shown at its best tuning

Usage:  COMPARISON_SEEDS=5 python scripts/run_comparison.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distributed_mpc_admm import (
    ADMMOptions,
    ConsensusADMM,
    DualDecomposition,
    DualDecompositionAgentSolver,
    DualDecompositionOptions,
)
from distributed_mpc_admm.communication_graph import CommunicationGraph
from distributed_mpc_admm.per_agent_solver import (
    AgentCostWeights,
    AgentLimits,
    CvxpyAgentSolver,
    DoubleIntegrator,
)

SEEDS = int(os.environ.get("COMPARISON_SEEDS", "3"))
TOL = 1e-4
MAX_ITER = 2000
SWEEP_MAX_ITER = 500

_model = DoubleIntegrator(dt=0.1, dim=2)
_limits = AgentLimits()


def _make_solvers(graph, horizon, *, dual: bool, weights: AgentCostWeights, offsets):
    cls = DualDecompositionAgentSolver if dual else CvxpyAgentSolver
    return {
        i: cls(
            agent_id=i,
            horizon=horizon,
            model=_model,
            limits=_limits,
            weights=weights,
            neighborhood=tuple(graph.closed_neighborhood(i)),
            # Formation cost is compiled in __init__ from these offsets; passing them
            # only to .solve() would silently drop the coupling (and unbounded DD).
            offsets=dict(offsets[i]) if offsets and offsets.get(i) else None,
        )
        for i in range(graph.n_agents)
    }


def _initial_state(graph, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(graph.n_agents, _model.n_states))
    x0[:, _model.dim :] *= 0.5
    return x0


def _instance(graph, horizon):
    references = {i: np.zeros((horizon, _model.dim)) for i in range(graph.n_agents)}
    offsets = {
        i: {j: np.zeros(_model.dim) for j in graph.neighbors(i)} for i in range(graph.n_agents)
    }
    return references, offsets


def _run_admm(graph, horizon, x0, references, offsets, rho, weights, max_iter=MAX_ITER):
    solvers = _make_solvers(graph, horizon, dual=False, weights=weights, offsets=offsets)
    options = ADMMOptions(
        rho=rho, max_iterations=max_iter, eps_abs=TOL, eps_rel=TOL, adaptive_rho=False
    )
    return ConsensusADMM(graph, solvers, horizon, _model.dim, options).solve(
        x0=x0, references=references, offsets=offsets
    )


def _run_dd(graph, horizon, x0, references, offsets, step, weights, max_iter=MAX_ITER):
    solvers = _make_solvers(graph, horizon, dual=True, weights=weights, offsets=offsets)
    options = DualDecompositionOptions(
        step_size=step, max_iterations=max_iter, eps_abs=TOL, eps_rel=TOL
    )
    return DualDecomposition(graph, solvers, horizon, _model.dim, options).solve(
        x0=x0, references=references, offsets=offsets
    )


def _best_param(score_fn, params):
    """Return the parameter minimising the score (lower iterations = better)."""
    scores = {p: score_fn(p) for p in params}
    return min(scores, key=scores.get), scores


def _sweep_rho(graph, horizon, x0, references, offsets, weights, grid):
    """Coarse sweep for ADMM's penalty parameter, returning ``(best_rho, scores)``.

    The sweep lives in its own function so the closed-over values are function
    parameters rather than variables of the caller's ``for`` loop. Inlining the lambda at
    the call site captures the loop variables by reference (ruff B023), which is harmless
    only for as long as the lambda is consumed before the next iteration — a property the
    next person to edit this file should not have to verify.
    """
    return _best_param(
        lambda rho: _run_admm(
            graph, horizon, x0, references, offsets, rho, weights, SWEEP_MAX_ITER
        ).iterations,
        grid,
    )


def _sweep_step(graph, horizon, x0, references, offsets, weights, grid):
    """Coarse sweep for dual decomposition's step size; see :func:`_sweep_rho`."""
    return _best_param(
        lambda step: _run_dd(
            graph, horizon, x0, references, offsets, step, weights, SWEEP_MAX_ITER
        ).iterations,
        grid,
    )


def main() -> None:
    horizon = 10
    configs = [
        ("4 agents, cycle", CommunicationGraph.cycle(4), AgentCostWeights()),
        ("4 agents, complete", CommunicationGraph.complete(4), AgentCostWeights()),
        ("8 agents, cycle", CommunicationGraph.cycle(8), AgentCostWeights()),
        ("8 agents, path", CommunicationGraph.path(8), AgentCostWeights()),
        ("4 agents, r_input = 0", CommunicationGraph.cycle(4), AgentCostWeights(r_input=0.0)),
    ]

    rows = []
    for name, graph, weights in configs:
        references, offsets = _instance(graph, horizon)
        seeds = list(range(SEEDS))

        # Coarse sweeps on seed 0 to pick each method's best tuning.
        x0_0 = _initial_state(graph, 0)
        rho_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        step_grid = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]

        best_rho, _ = _sweep_rho(graph, horizon, x0_0, references, offsets, weights, rho_grid)
        best_step, _ = _sweep_step(graph, horizon, x0_0, references, offsets, weights, step_grid)

        admm_iters, admm_wall, admm_r = [], [], []
        dd_iters, dd_wall, dd_r = [], [], []
        for seed in seeds:
            x0 = _initial_state(graph, seed)
            a = _run_admm(graph, horizon, x0, references, offsets, best_rho, weights)
            d = _run_dd(graph, horizon, x0, references, offsets, best_step, weights)
            admm_iters.append(a.iterations)
            admm_wall.append(a.solve_time)
            admm_r.append(a.history.primal_residual[-1])
            dd_iters.append(d.iterations)
            dd_wall.append(d.solve_time)
            dd_r.append(d.history.primal_residual[-1])

        rows.append(
            {
                "problem": name,
                "admm_rho": best_rho,
                "dd_step": best_step,
                "admm_iters_median": float(np.median(admm_iters)),
                "dd_iters_median": float(np.median(dd_iters)),
                "admm_wall_median_ms": float(np.median(admm_wall)) * 1e3,
                "dd_wall_median_ms": float(np.median(dd_wall)) * 1e3,
                "admm_r_final_median": float(np.median(admm_r)),
                "dd_r_final_median": float(np.median(dd_r)),
            }
        )

    print("\n=== measured comparison (median over", SEEDS, "seeds) ===")
    print(json.dumps(rows, indent=2))
    print("\nMarkdown table:\n")
    print(
        "| Problem | ADMM rho | DD step | ADMM iters | DD iters | ADMM wall (ms) | DD wall (ms) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['problem']} | {r['admm_rho']:g} | {r['dd_step']:g} | "
            f"{r['admm_iters_median']:.0f} | {r['dd_iters_median']:.0f} | "
            f"{r['admm_wall_median_ms']:.0f} | {r['dd_wall_median_ms']:.0f} |"
        )


if __name__ == "__main__":
    main()
