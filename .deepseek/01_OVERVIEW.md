# §1 · What is being built

`N` double integrators in 2D, each running its own model predictive controller. The agents are
coupled only through formation costs on relative positions, and they agree on a joint plan by
running general-form consensus ADMM over a communication graph — one round of neighbour-to-neighbour
messages per iteration, no central solver anywhere.

```
                 agent i                                    agent j (a neighbour)
  ┌───────────────────────────────────┐               ┌──────────────────────────┐
  │ x-update  (local QP, no comms)    │               │ x-update                 │
  │   min f_i(U_i, y_i)               │               │                          │
  │     + (ρ/2)Σ‖y_i^j − z^j + λ_i^j‖² │               │                          │
  └───────────────┬───────────────────┘               └────────────┬─────────────┘
                  │  y_i^j + λ_i^j                                 │
                  ▼                    ── one message round ──►    ▼
  ┌───────────────────────────────────┐               ┌──────────────────────────┐
  │ z-update  z^i = mean over          │◄─────────────►│ z^j = mean over C_j      │
  │           contributors C_i         │   z^i, z^j    │                          │
  └───────────────┬───────────────────┘               └────────────┬─────────────┘
                  │ λ_i^j += y_i^j − z^j                           │
                  ▼                                                ▼
             u_i(0) → plant                                   u_j(0) → plant
```

Two implementations of the same algorithm:

| Path | Where | Role |
| --- | --- | --- |
| Python | `python/distributed_mpc_admm/` | the **reference**. Readable, CVXPY-backed, deliberately not fast. Every result in the notebooks comes from here. |
| C++ / ROS 2 | `cpp_admm/` | the **production kernel**. OSQP, preallocated, one OS process per agent, ROS 2 or ZeroMQ transport. Validated against the Python path to 1e-8. |

## §1.1 Why this repository exists

It is a portfolio repository for a robotics PhD application. Its job is to let a reviewer conclude,
in five minutes of reading, that the author can *distribute* a computation rather than simulate one
centrally and call it distributed — and that they know exactly where the convergence theory
underneath it stops applying.

That drives most of the engineering decisions in these documents:

- **The correctness claim rests on an external reference, not on self-consistency.** The
  distributed solution is checked against a single monolithic QP over all agents (A2). "My ADMM
  agrees with my ADMM" persuades nobody who has written one.
- **A second implementation checks the first.** The C++ kernel exists so that two independently
  written code paths must agree on the iterates to 1e-8 (A7).
- **The distributed claim is itself tested.** A3 is a structural test that no agent update ever
  sees data outside its closed neighbourhood. Without it, the central claim of the repository is
  an assertion in a README.
- **Stating limits is a strength here.** §1.6, the "what this does not prove" section of
  `docs/README_math.md`, and the README's limitations section are load-bearing. A reviewer who
  finds a limitation you did not list stops trusting the ones you did.

Downstream, the C++ kernel is the distributed solver inside the author's `transition-viable-swarm`.
Keep it free of ROS dependencies at the type level ([09_CPP_KERNEL.md §9.1](09_CPP_KERNEL.md)).

## §1.2 Deliverables beyond the code

These are part of "done", not extras:

| Deliverable | Produced by | Notes |
| --- | --- | --- |
| `media/4_agent_formation.gif` | `notebooks/03_formation_control.ipynb` | four agents to a square, communication edges overlaid |
| `media/topology_switch.gif` | `notebooks/04_switching_topology.ipynb` | split into two components, drift, merge — the split and merge instants annotated **on the frames** |
| `media/convergence_curves.png` | `notebooks/05_convergence_analysis.ipynb` | primal and dual residuals per topology, tolerance thresholds shown |
| `docs/COMPARISON_VS_DUAL_DECOMP.md` | a dual-decomposition baseline you implement | measured numbers, same seeds, same hardware ([§13.3](13_DOCS.md)) |
| README results table | `analysis/` + the test suite | every row names its hardware and git SHA |
| **One documented case where consensus ADMM loses** | [§12.4](12_ANALYSIS.md), [§13.3 §6](13_DOCS.md) | the disconnected-graph failure and the regimes where dual decomposition wins — deliberate, not an oversight |

## §1.3 Acceptance criteria

The whole project is done when all nine hold.

| # | Criterion | Verified by |
| --- | --- | --- |
| A1 | Clean-container build: pytest green on 3.12 and 3.13, ruff/black/mypy clean, `colcon build && colcon test` green in `ros:lyrical-ros-base` | `test.yml` ([§14.1](14_CI.md)) |
| A2 | Distributed equals centralised: `max abs(u_admm − u_central) ≤ 1e-3` at `eps_abs = eps_rel = 1e-6`, for `N ∈ {4, 8}` across complete / cycle / path / star | `test_matches_centralized_solution` ([§10.1](10_TESTS.md)) |
| A3 | No global information: every `LocalProblemData` an agent receives carries keys only inside its closed neighbourhood, and no residual or objective feeds an update | `test_no_global_information_used` ([§10.1](10_TESTS.md)) |
| A4 | Closed-loop formation: edge-RMS ≤ 0.05 m held over the final 20 % of a 100-step run, `N ∈ {4, 8}` | `test_agents_reach_formation` ([§10.2](10_TESTS.md)) |
| A5 | Iterations track connectivity: median iterations to 1e-4 satisfy complete ≤ cycle ≤ path over ≥ 30 seeds, and the fitted exponent of iterations against `λ₂` is reported | `test_converges_on_all_topologies` ([§10.1](10_TESTS.md)), [§12.5](12_ANALYSIS.md) |
| A6 | Warm starting: median iterations after control step 5 at most half the cold-start median | `test_warm_start_reduces_iterations` ([§10.1](10_TESTS.md)) |
| A7 | C++ and Python agree: kernel iterates elementwise to 1e-8 over 20 iterations (`alpha = 1`, fixed `rho`), converged `u₀` to 1e-6 | `PythonParityTest` ([§10.4](10_TESTS.md)) |
| A8 | Real-time: p95 per control step < 10 ms at `N = 4`, `T = 15`, 20 ADMM iterations, Release, **CPU named**; zero heap allocations inside `iterate()` | `NoHeapAllocationDuringIterate` + benchmark ([§10.4](10_TESTS.md)) |
| A9 | Degradation is measured, not claimed: loss and switching sweeps give bounded NaN-free residuals with monotone degradation in loss probability, and the disconnected case is documented as a failure with its mechanism | [§12.4](12_ANALYSIS.md), `test_packet_loss_degrades_gracefully` ([§10.1](10_TESTS.md)) |

**If a criterion cannot be met, change the criterion in this file with a written reason.**
Do not weaken a test in place — a threshold quietly relaxed to make CI green is a lie told to every
future reader.

A8 is the one most likely to be noisy: shared CI runners do not give reproducible timings. Make the
CI assertion generous and take the real p95 from a local Release run with the CPU named.

A9's second half is the one people forget. A degradation study that only shows the cases which
still work proves nothing. The disconnected-graph run *must* appear, and it must be described as a
failure with an explanation, not smoothed into a footnote.

## §1.4 Design decisions already made

These are settled. They are recorded here so they are not re-litigated mid-implementation; each is
justified in its own document.

| Decision | Value | Where |
| --- | --- | --- |
| Splitting | general-form consensus ADMM with per-agent local copies `y_i^j` | [06_ADMM.md §6.1](06_ADMM.md) |
| Who computes `z^j` | agent `j` itself, from its contributors | [06_ADMM.md §6.3](06_ADMM.md) |
| Dual form | **scaled** duals, `λ = u/ρ` | [16_CONVENTIONS.md §16.1](16_CONVENTIONS.md) |
| Formation targets | quadratic **costs**, never hard equalities | [07_FORMATION.md §7.1](07_FORMATION.md) |
| Formation encoding | relative offsets `d_ij = o_i − o_j`, not distances | [07_FORMATION.md §7.1](07_FORMATION.md) |
| Per-agent formation weight | `w_formation` per incident edge; the centralised reference uses `2·w_formation` per edge | [05_LOCAL_QP.md §5.3](05_LOCAL_QP.md) |
| Flattening | time-major, **C order**, everywhere | [16_CONVENTIONS.md §16.1](16_CONVENTIONS.md) |
| Horizon indexing | `X` covers `t = 1..T`; `x₀` is not a row of `X` | [16_CONVENTIONS.md §16.1](16_CONVENTIONS.md) |
| Block ordering | sorted closed neighbourhood, self in sorted position | [04_GRAPH.md §4.1](04_GRAPH.md) |
| Missing contribution in the z-average | **excluded**, divisor shrinks; never a stale substitute | [06_ADMM.md §6.3](06_ADMM.md) |
| Missing `z` in the broadcast | falls back to the last known value | [06_ADMM.md §6.6](06_ADMM.md) |
| Adaptive `rho` | off by default; when on, duals MUST be rescaled | [06_ADMM.md §6.5](06_ADMM.md) |
| Python solver back-end | CVXPY + OSQP, problem compiled once, parameters assigned | [05_LOCAL_QP.md §5.4](05_LOCAL_QP.md) |
| C++ solver back-end | OSQP directly, fixed sparsity pattern, value-only updates | [09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md) |
| C++ transport | abstract `ITransport`; ROS 2, ZeroMQ and in-process implementations | [09_CPP_KERNEL.md §9.1](09_CPP_KERNEL.md) |
| Node type | plain `rclcpp::Node` with a timer, one process per agent | [11_NODE.md §11.1](11_NODE.md) |
| ADMM QoS | best-effort, volatile, depth 1 | [11_NODE.md §11.1](11_NODE.md) |
| Licence | BSD-3-Clause (matches `LICENSE`, `package.xml` and the sibling repositories) | [02_ENVIRONMENT.md §2.6](02_ENVIRONMENT.md) |

## §1.5 What this skeleton adds beyond the original spec

[INFO.md](INFO.md) fixes the target structure. The skeleton implements all of it, plus the files
below, which exist because the specified files cannot be written without them. They are listed so
the deviation is explicit rather than discovered.

| Addition | Why |
| --- | --- |
| `docs/derivations/preamble.tex` | the three specified `.tex` files all `\input{preamble}`; without it none of them compiles, and three private copies of the macros is how three documents drift into three notations |
| `docs/derivations/references.bib` | all three specified `.tex` files call `\bibliography{references}` |
| `media/README.md` | provenance for generated artefacts; the GIFs themselves are build outputs, not source |
| `.gitignore` | keeps `build/`, `install/`, `log/`, `__pycache__/` and executed notebooks out of the tree |
| `cpp_admm/test/data/four_agent_reference.json` | A7 needs a fixture the C++ test can load; exported from notebook 02 ([§10.5](10_TESTS.md)) |

Two structural choices worth stating, both of which stay inside the specified tree:

- **`InProcessTransport`** is declared in the specified `admm_kernel.hpp` rather than in a new file.
  Without it the C++ kernel cannot be unit-tested without a network, which would make M7
  undebuggable.
- **The double-integrator model and prediction matrices live in `per_agent_solver.py`**, not in a
  new `dynamics.py`. `INFO.md` does not list a dynamics module and the model is only ever consumed
  by the local QP.

Nothing specified in `INFO.md` was dropped or renamed.

## §1.6 What this repository does not prove

Load-bearing. Repeat these, in these words, wherever a guarantee is discussed.

- **No recursive feasibility, no closed-loop stability guarantee.** There is no terminal set and no
  terminal cost. Each local problem is always feasible because the constraints are boxes and the
  neighbour copies are free variables — that is feasibility, not stability. Convergence of the
  closed loop is demonstrated numerically over the tested initial conditions and nothing more.
- **Convergence of the ADMM iteration is guaranteed only for a fixed, connected graph with
  synchronous updates and exact local solves.** Those are the assumptions of the standard proof.
  Every experiment in `notebooks/04_switching_topology.ipynb` deliberately breaks one of them, and
  the results there are *measurements*, not theorems. Say "we measure" and never "we show that it
  still converges".
- **The disconnected case does not converge to a common formation, and is not meant to.** Two
  components hold their own shapes and drift apart. This is reported as a failure with a mechanism;
  it is the motivating gap for the author's thesis, not a bug to be tuned away.
- **No collision avoidance of any kind.** Agents are coupled by formation cost only. Two agents
  whose formation targets cross will pass through each other.
- **Second-order integrator dynamics only.** No attitude, no drag, no actuator dynamics. A double
  integrator is not a quadrotor and the repository must not imply that it is.
- **State-estimation error is not modelled.** The measurement noise option perturbs what the solver
  sees; it is not an observer study.
- **The switching schedule in the ROS demo is published centrally.** It stands in for physical link
  availability. The *control* is fully distributed; the *schedule* is not. Say both.
