# §16 · Conventions and known traps

**Read §16.1 before M2. Re-read the whole page whenever something behaves inexplicably.** Most of
the expensive bugs in this repository are on this page, and the first section is the one that turns a
working optimiser into a confidently wrong one.

> **The failure mode of this repository is a solve that converges cleanly to the wrong problem.**
>
> Nothing here crashes when it is wrong. There is no barrier to violate, no obstacle to hit, no
> assertion that fires. A double-counted formation weight, an unscaled dual, a stale trajectory in
> the average, a missing Hessian cross term — each produces an iteration that decreases its
> residuals, reports `converged = True`, and returns the optimum of a problem you did not pose. The
> trajectory looks plausible. The GIF looks good.
>
> That is why A2 (comparison against a centralised solve) and A7 (comparison against a second
> implementation) exist, and why they are worth more than any number of internal consistency checks.

---

## §16.1 Conventions that decide correctness — memorise these

| Quantity | Convention |
| --- | --- |
| Flattening | **time-major, C order**: a `(T, dim)` block flattens to index `t·dim + d` |
| Horizon | `X = Phi·x₀ + Gamma·U` covers `t = 1..T`; **`x₀` is not a row of `X`** |
| Block ordering | **sorted closed neighbourhood**, self in sorted position — not first |
| Local copy | `y_i^j` = agent `i`'s copy of agent `j`'s **position** trajectory, `(T, dim)` |
| Consensus var | `z^j` is computed **by agent `j`**, from `contributors(j) = N̄_j` |
| Dual | **scaled**: `λ = u/ρ`. Changing `ρ` MUST rescale `λ` by the reciprocal |
| Formation offset | `d_ij = o_i − o_j`, so `d_ji = −d_ij`; the cost is `‖(y_i − y_j) − d_ij‖²` |
| Formation weight | `w_formation` per agent per incident edge ⇒ **`2·w_formation` per edge** in `Σ_i f_i` |
| z-update divisor | the number of **received** contributions, not `|C_j|` |
| Missing local copy | **excluded** from the z-average |
| Missing `z` broadcast | **falls back** to the last known value |
| Dynamics constraint | on `y_i^i` **only**; `y_i^j` for `j ≠ i` are free variables |
| Primal residual | `sqrt( Σ_i Σ_{j∈N̄_i} ‖y_i^j − z^j‖² )`, using the **un-relaxed** `y` |
| Dual residual | `ρ · sqrt( Σ_i Σ_{j∈N̄_i} ‖z^j − z^j_prev‖² )` |
| `n_dual` | `Σ_i |N̄_i| · T · dim` |
| `ρ` | larger = more weight on agreement; the U-curve optimum is problem-dependent |
| `α` (relaxation) | `∈ [1, 2)`; `1.0` disables. Changes the path, never the fixed point |
| OSQP Hessian | `½xᵀPx + qᵀx`, so a cost `w‖x‖²` contributes **`2w`** to `P`; `P` **upper-triangular CSC** |

**Write a test for each of these that inspects the assembled quantity directly.** Do not rely on
end-to-end trajectories to catch a convention error. An implementation with the wrong flattening
order still produces smooth, plausible motion.

### The four traps specific to this formulation

1. **Formation double-counting.** Edge `(i, j)` is in `f_i` *and* `f_j`. The centralised reference
   ([06_ADMM.md §6.7](06_ADMM.md)) must use `2·w_formation` per edge. This costs a day when missed,
   because A2 fails by a clean factor that reads as a convergence problem.

2. **Dual rescaling on a `ρ` change.** `λ` stores `u/ρ`. Adaptive `ρ` without the reciprocal
   rescaling silently changes the dual iterate. The symptom is "adaptive `ρ` does not help" — a
   conclusion plausible enough that nobody investigates.

3. **The two directions of "missing".** Excluded from the z-average; substituted from cache on the
   `z` broadcast. Getting them the same way round is the intuitive mistake, and it biases the
   consensus in a way no test will name for you.

4. **The Hessian cross terms.** `−2·w_formation·I` between the self block and each neighbour block
   ([09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md)). Omit them and the cost becomes
   `w(‖y_i‖² + ‖y_j‖²)` — a different, well-posed, convergent problem with the wrong answer.

## §16.2 Units, shapes and frames

SI throughout. Distances in metres, angles in radians, time in seconds at every internal interface.

| Object | Shape |
| --- | --- |
| agent state | `(4,)` — `[px, py, vx, vy]` |
| state log | `(K+1, N, 4)` — **includes `x₀`** |
| input log | `(K, N, 2)` — one shorter than the state log |
| input sequence | `(T, 2)` |
| trajectory block `y`, `z`, `λ` | `(T, 2)` — **position only** |
| stacked prediction | `(T·4,)`, C order |

The off-by-one between `states` and `inputs` is deliberate
([08_CLOSED_LOOP.md §8.4](08_CLOSED_LOOP.md)) and every consumer must respect it.

`h`-style dimensional reasoning does not apply here, but one thing does: **`w_formation` and
`q_position` are not interchangeable across problem scales.** Their *ratio* decides whether the
formation or the reference wins during a conflict, and it is scale-dependent through the offsets.
Document the ratio you tuned at, not just the values.

## §16.3 Time and discretisation

`dt` is a property of the model and of the generated prediction matrices, not a runtime knob.

`control_rate_hz` in the node MUST equal `1/dt` unless there is a stated reason
([11_NODE.md §11.2](11_NODE.md)). A mismatch means the plant advances by a different amount than the
prediction assumed, and every trajectory is then optimal for a system you are not controlling.

`T·dt` must exceed the formation settling time, or the horizon cannot see the manoeuvre it is
planning. This is the first thing to check when a formation converges slowly for no apparent reason.

## §16.4 Numerical policy

- **Every stochastic path takes an explicit `rng: np.random.Generator | int | None`.** Never call
  `np.random.*` module-level functions. A result that cannot be reproduced from a seed is not a
  result, and this repository's conclusions are mostly statistical.
- `λ₂` is clipped at 0 from below ([04_GRAPH.md §4.3](04_GRAPH.md)); the analytic zero eigenvalue
  arrives as `-1e-16` and propagates as a `nan` three functions later.
- Tolerances: `1e-12` for prediction-matrix identities, `1e-9` for a single QP's dynamics
  consistency, `1e-6` for ADMM convergence in tests, `1e-3` for the centralised comparison, `1e-8`
  for C++/Python parity. **Do not tighten one because it happens to pass, and never loosen one to
  make it pass.**
- `matrix_rank` always takes an explicit `tol` ([07_FORMATION.md §7.3](07_FORMATION.md)).
- Warm starts must never change a result, only the iteration count. If a warm start changes the
  answer beyond solver tolerance, the local problem is degenerate — investigate rather than accept
  it.
- `consensus_gap` and every residual norm are **diagnostics**. They must never influence an update
  (A3).

## §16.5 Iteration budget, not convergence target

In closed loop the ADMM iteration count is a **real-time budget**. `max_iterations = 50` in the node
and 10–20 warm-started iterations in practice; a warm-started 15 beats a cold 200
([08_CLOSED_LOOP.md §8.3](08_CLOSED_LOOP.md)).

Hitting the cap is therefore normal and is **not** a failure. What must be true is that
`converged = False` is reported honestly and that the step is counted in `SimulationLog.summary()`.
A run that capped out on 30 % of its steps and a run that converged everywhere can produce
indistinguishable trajectories; only the counter distinguishes them.

## §16.6 Logging and allocation

Nothing unthrottled in the ROS control loop — `RCLCPP_*_THROTTLE` with a 1–2 s period. A `printf` at
10 Hz across four processes is a latency bug, not a debugging aid.

**No heap allocation in `AdmmKernel::iterate()` after `configure()`**
([09_CPP_KERNEL.md §9.1](09_CPP_KERNEL.md)). A8 is a p95 criterion and an allocation that happens
1 % of the time is precisely what a tail measurement catches.
`NoHeapAllocationDuringIterate` enforces it.

## §16.7 C++ standard

The package builds at **C++20**, but the sources MUST stay **C++17-compatible**: no concepts, no
ranges, no `std::format`. Risk V3 ([§2.1](02_ENVIRONMENT.md)) is unresolved, and keeping the code at
17 is what makes the fallback a one-line change to `CMAKE_CXX_STANDARD`.

## §16.8 What must never appear in the same commit as a green report

- A tolerance that moved
- A test converted to `GTEST_SKIP` or `pytest.skip` without a written reason
- A regenerated parity fixture ([10_TESTS.md §10.5](10_TESTS.md))
- A number in a README or notebook that no longer has a cell producing it
- A `TODO(deepseek …)` deleted without the implementation

## §16.9 The mistakes, ranked by what they cost

1. **Formation edge double-counting** (§16.1 trap 1) — presents as an ADMM convergence bug.
2. **Missing dual rescaling in adaptive `ρ`** (trap 2) — converges at fixed `ρ`, stalls with
   adaptation.
3. **Missing Hessian cross terms in C++** (trap 4) — a different problem that still converges.
4. **Flatten-order mismatch** — Fortran vs C order between `Gamma` and CVXPY. A plausible-looking
   wrong trajectory.
5. **Horizon off-by-one** — `X` starting at `t = 0` instead of `t = 1`.
6. **Rebuilding the CVXPY problem inside `solve`, or losing DPP compliance**
   ([05_LOCAL_QP.md §5.4](05_LOCAL_QP.md)) — ~100× slowdown, no wrong answers, so nothing fails
   except your patience and your performance conclusions.
7. **`poll()` sleeping instead of spinning** ([11_NODE.md §11.3](11_NODE.md)) — the ROS demo
   silently degrades to fully asynchronous while appearing to work.
8. **A stale value substituted into the z-average** (trap 3).
9. **Non-upper-triangular `P` handed to OSQP** (risk V11) — quiet misbehaviour.
10. **Graph caches not invalidated on `add_edge` / `remove_edge`**
    ([04_GRAPH.md §4.1](04_GRAPH.md)) — every spectral quantity silently describes the old topology.
11. **`np.random` module functions instead of a seeded `Generator`** (§16.4).
12. **Zeroing duals on warm start** ([08_CLOSED_LOOP.md §8.3](08_CLOSED_LOOP.md)) — A6 fails.

---

## The failures you will actually hit

| Symptom | First thing to check |
| --- | --- |
| **ADMM converges, but not to the centralised solution** | formation double-count (§16.1 trap 1), then the Hessian/objective term by term |
| Converges for fixed `ρ`, stalls with `adaptive_rho=True` | dual rescaling (trap 2) |
| Residual plateaus at a floor well above tolerance | stale data in the z-average (trap 3), or a `LossyChannel` left enabled from a previous cell |
| Trajectory is smooth but the formation is subtly wrong | flatten order, or the horizon off-by-one |
| A 4-agent closed-loop run takes minutes | the CVXPY problem is being rebuilt, or DPP compliance was lost (risk V8) |
| Iterations explode on the path graph but not the cycle | expected — `λ₂(path) ≈ 0.586`, `λ₂(cycle) = 2`. Check against [§12.5](12_ANALYSIS.md) before calling it a bug |
| Formation converges in shape but drifts bodily | no leader, or a follower unreachable from the leader set ([07_FORMATION.md §7.4](07_FORMATION.md)) |
| Agents pass through each other | expected. There is no collision avoidance ([01_OVERVIEW.md §1.6](01_OVERVIEW.md)) |
| `nan` after a topology switch | z-average divided by zero received contributions ([06_ADMM.md §6.3](06_ADMM.md)) |
| Spectral numbers describe the wrong graph | cache not invalidated (§16.9 item 10) |
| C++ and Python disagree | block ordering, flatten order, or the `2w` factor in the OSQP Hessian — in that order |
| C++ solve time spikes intermittently | allocation in `iterate()` (§16.6), or `updateRho` being called every iteration when `ρ` did not change |
| ROS demo forms up but every staleness counter reads zero | `poll()` is sleeping instead of spinning ([11_NODE.md §11.3](11_NODE.md)) |
| One agent never moves in the ROS demo | its process died at construction — check for a thrown config exception ([11_NODE.md §11.4](11_NODE.md)) |
| GIF shows edges that should have vanished | `blit=True` in the animation ([08_CLOSED_LOOP.md §8.6](08_CLOSED_LOOP.md)) |

## A note on debugging order

When something is wrong, the cost of checking goes: **conventions in §16.1 (minutes) → the
centralised comparison (minutes) → the C++/Python parity check (minutes) → per-iteration residual and
`consensus_gap` dumps (minutes) → the running ROS graph (hours)**. Work in that order.

The tests in [10_TESTS.md](10_TESTS.md) exist precisely so the cheap checks are available before you
need them. Reaching for `ros2 topic echo` first is the expensive path, and it is the one everyone
takes.
