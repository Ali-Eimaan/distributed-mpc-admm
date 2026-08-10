# §9 · The C++ kernel

**Governs:** `cpp_admm/include/cpp_admm/admm_kernel.hpp` · `per_agent_qp.hpp`,
`cpp_admm/src/admm_kernel.cpp` · `per_agent_qp.cpp`
**Milestone:** M7
**Done when:** **A7** and **A8**.

The Python path is the reference; this is the one that runs on a vehicle. It exists to prove two
things: that the algorithm survives being written twice (A7), and that it fits a real-time budget
(A8).

---

## §9.1 What the kernel is, and what it must not know about

`AdmmKernel` MUST NOT include a ROS header or name a ROS type. It is destined for the author's
`transition-viable-swarm`, and a ROS type on its interface drags the whole middleware in.

Communication goes through `ITransport`:

```cpp
virtual bool publish(MessageKind, const NeighborMessage &) = 0;
virtual std::size_t poll(MessageKind, std::chrono::microseconds, std::vector<NeighborMessage> &) = 0;
virtual void flush() = 0;
```

Three implementations:

| Implementation | Where | For |
| --- | --- | --- |
| `InProcessTransport` | `admm_kernel.cpp` (§9.8) | unit tests, deterministic loss and delay, no network |
| `ZeroMqTransport` | `admm_kernel.cpp` (§9.9) | standalone benchmark with no ROS graph |
| `RosTransport` | `consensus_node.cpp` ([§11.3](11_NODE.md)) | the demo |

`publish` returning `false` is a **dropped packet, not an error**. A full queue is a normal outcome
and the kernel is required to tolerate it (§9.5).

### Real-time contract

- **No heap allocation inside `iterate()`.** Everything is sized in `configure()`.
  `NoHeapAllocationDuringIterate` ([§10.4](10_TESTS.md)) enforces it — write `iterate()` assuming it
  will be checked, because it will.
- **No unbounded blocking.** `poll` takes an explicit timeout and a missed message is a handled
  outcome, never a reason to stall the control loop.
- `iterate()` is callable from a real-time thread; `configure()` and `setNeighbors()` are not, and
  every call site of `setNeighbors()` should say so.

## §9.2 `AgentConfig`

`closed_neighborhood()` returns `neighbors ∪ {agent_id}`, **sorted ascending**. Derive it here and
nowhere else — this ordering defines the QP block layout (§9.3) and must match
`CommunicationGraph.closed_neighborhood` in Python ([04_GRAPH.md §4.1](04_GRAPH.md)) exactly, or A7
fails in a way that looks numerical.

`validate()` throws `std::invalid_argument` on: `agent_id` outside `[0, n_agents)`; `agent_id`
present in `neighbors`; duplicate neighbours; `horizon ≤ 0`; `dim ∉ {2, 3}`; `dt ≤ 0`; any negative
weight; an `offsets` key that is not a neighbour; an offset whose size is not `dim`.

Fail at construction, loudly. A defaulted neighbour list produces a controller that runs and
converges to the wrong formation ([00_RULES.md](00_RULES.md), rule 6).

## §9.3 `PerAgentQp`

### Variable layout — the contract

```
theta = [ U | y^{c_0} | y^{c_1} | … | y^{c_{M−1}} ]
```

`M = |closed neighbourhood|`, blocks in ascending agent-id order, each `horizon·dim` long,
**time-major** within a block (index `t·dim + d`). Total `n = (1 + M)·horizon·dim`.

Constraint rows, in this order:

| Rows | Constraint | Count |
| --- | --- | --- |
| 1 | `y^self − Gamma_p·U = Phi_p·x₀` | `horizon·dim` |
| 2 | `−u_max ≤ U ≤ u_max` | `horizon·dim` |
| 3 | `−v_max ≤ Cv(Phi·x₀ + Gamma·U) ≤ v_max` | `horizon·dim` |

Every caller goes through `inputOffset()` / `copyOffset(j)` / `extractInputs` / `extractCopy`. Do
not recompute an offset inline anywhere — `BlockLayoutMatchesHeaderContract`
([§10.4](10_TESTS.md)) tests the accessors, not your arithmetic.

### Why the consensus penalty adds no constraint rows

`(ρ/2)‖y^j − z^j + λ^j‖²` is quadratic in `y` with Hessian `ρI` and linear term `−ρ(z^j − λ^j)`. A
change in `ρ` touches the diagonal of `P`; a change in `(z, λ)` touches `q` only. Both are
**value-only updates on an unchanged sparsity pattern**. That is the entire reason this class
exists, and it is what makes A8 achievable.

### The Hessian

OSQP minimises `½·xᵀPx + qᵀx`, so a cost `w‖x‖²` contributes `2w` to `P`.

| Block | Contribution |
| --- | --- |
| `U` | `2·(r_input·I + r_rate·DᵀD + q_velocity·Gamma_vᵀGamma_v)` |
| `y^self` | `2·(q_position·I + terminal + Σ_j w_formation·I) + ρ·I` |
| `y^j`, `j ≠ self` | `2·w_formation·I + ρ·I` |
| cross `y^self` / `y^j` | `−2·w_formation·I` |

**Do not omit the cross terms.** Without them the formation cost is `w(‖y_i‖² + ‖y_j‖²)` instead of
`w‖y_i − y_j − d‖²` — a different problem that still solves, still converges, and returns a
plausible trajectory to the wrong formation ([§16.9](16_CONVENTIONS.md), item 3).

`Gamma_vᵀGamma_v` is dense. That is expected, and it is why `q_velocity` defaults small.

Build `P` **upper-triangular in CSC** (risk V11, [§2.1](02_ENVIRONMENT.md)). OSQP requires it and
misbehaves quietly when handed the full symmetric matrix.
`SetupSucceedsAndHessianIsPsd` checks both the triangularity and `λ_min(P) ≥ 0`.

### Value-only updates

Record `rho_diag_indices` at setup: the positions in `P.valuePtr()` of the `ρI` diagonal entries.
`updateRho` overwrites those and calls `osqp_update_data_mat` with the **same index list**.

A wrong index list corrupts the Hessian into a still-solvable but wrong problem.
`UpdateRhoTouchesOnlyDiagonalEntries` is the guard, and it is one of the three tests in this
repository that catch a silent-wrongness bug.

**Skip `updateRho` entirely when `ρ` is unchanged.** A `P` update triggers a refactorisation costing
roughly as much as a solve, and with `adaptive_rho` off it would run every iteration for nothing.

Cache the static part of `q` so `updateConsensus` is one axpy per block.

Verify the OSQP API names against your installed `osqp.h` before writing any of this (risk V10) and
put the verified names in a comment.

## §9.4 Status mapping

`QpSolution::ok()` returns true for `kSolved` **and** `kSolvedInaccurate`. The outer ADMM iteration
tolerates inexact inner solves; that is a documented property of the method, not a shortcut. Every
other status is not ok.

`toString(QpStatus)` returns a `static constexpr` string per enumerator. It is called from log paths;
do not build a `std::string`.

**Read the status enumeration out of your installed OSQP headers and put the verified mapping in a
comment** (risk V10). An infeasible solve reported as success is the one failure here that is worse
than a crash.

## §9.5 `AdmmKernel`

`Impl` holds, all sized in `configure()` and never resized afterwards:

```
AgentConfig config; ADMMOptions options; ITransport * transport;
std::unique_ptr<PerAgentQp> qp;
std::vector<int> closed_nbhd;                       // block ordering
std::unordered_map<int, Eigen::VectorXd> y, y_hat, lam, z_received;
Eigen::VectorXd z_self, z_self_prev, inputs, x0, reference;
std::unordered_map<int, int64_t> last_seen_iteration;
std::vector<NeighborMessage> rx_buffer;             // reserved once, reused every poll
ADMMStats stats; int64_t control_step; int iteration;
```

`iterate()` runs exactly the phase order of [06_ADMM.md §6.1](06_ADMM.md):
`xUpdate → relax → exchangeLocalCopies → zUpdate → broadcastConsensus → dualUpdate →
computeResiduals → updateRho`.

`solve()`:

```cpp
stats.reset();
for (int k = 0; k < options.max_iterations; ++k) {
  if (iterate()) { stats.converged = true; break; }
  if (options.max_staleness > 0 && stats.max_staleness_seen > options.max_staleness) break;
}
```

The staleness exit matters: without it, an agent whose link died spins to `max_iterations` on frozen
data every control step, burning the budget and producing an input derived from a neighbour
prediction that is seconds old.

`xUpdate()` on a non-ok QP status **holds the previous `y` and records it**. Do not propagate NaNs
into the consensus step; one agent's failed solve would otherwise poison every neighbour's average
in the same iteration.

`computeResiduals()` computes **this agent's local contribution only**. The kernel must decide when
to stop using what a real agent actually has. Publish the local residual in diagnostics and let an
offline monitor sum them ([11_NODE.md §11.4](11_NODE.md)).

`exchangeLocalCopies()` drops any message whose `control_step` is not current. `zUpdate()` excludes
missing contributions and shrinks the divisor; `broadcastConsensus()` falls back to the cached
last-known `z^j` and bumps `stats.max_staleness_seen`. Same asymmetry as
[06_ADMM.md §6.3 / §6.6](06_ADMM.md) — keep the two directions straight.

`shiftWarmStart()` shifts `y`, `z` **and `lam`** by one `dim`-sized step, duplicating the tail
([08_CLOSED_LOOP.md §8.3](08_CLOSED_LOOP.md)).

`setNeighbors()` preserves `y`/`lam` for surviving neighbours, zero-initialises new ones, drops the
rest, and re-setups OSQP. It reallocates.

## §9.6 `ADMMStats`

`reset()` zeroes every field **except `rho`**, which keeps its adapted value across control steps.
Resetting `rho` each step throws away the adaptation and makes the closed-loop iteration counts
look worse than they are.

Fields worth having and easy to forget: `messages_missing`, `max_staleness_seen`, and the split of
`solve_time_ms` into `qp_time_ms` and `comm_time_ms`. The split is what tells you whether the
kernel is solver-bound or network-bound, and it is the number [§12.2](12_ANALYSIS.md) quotes.

## §9.7 `NeighborMessage`

`payload` is a flattened `horizon·dim` position block in the same time-major C order as everywhere
else ([§16.1](16_CONVENTIONS.md)).

`byte_size()` is header fields plus `payload.size() * sizeof(double)`. It MUST match the analytic
model in [04_GRAPH.md §4.6](04_GRAPH.md) or the bandwidth comparison in
[§12.7](12_ANALYSIS.md) is not a check of anything.

`MessageKind` distinguishes `kLocalCopy` (`y_i^j`, contributor → subject) from `kConsensus` (`z^j`,
subject → contributors). Keep it explicit: a receiver that mistakes a local copy for an agreed
consensus value produces an iteration that looks like ADMM and is not.

## §9.8 `InProcessTransport`

Deterministic loopback. `connect(peers)` populates a shared registry mapping agent id to `Impl*`;
`publish` pushes into the destination's inbox (mutex-guarded — a future multi-threaded test will
need it), applying **the same loss and delay model as `LossyChannel`**
([04_GRAPH.md §4.5](04_GRAPH.md)) so a C++ result and a Python result on the same seed are
comparable.

`poll` ignores the timeout — delivery is synchronous — and drains the inbox for the requested kind.

## §9.9 `ZeroMqTransport`

PIMPL so `<zmq.hpp>` stays out of the header. One `PUB` socket bound to `bind_endpoint`, one `SUB`
socket connected to each neighbour, topics `"admm/<kind>/<subject>"` **precomputed at construction**
— building them per publish allocates in the hot path.

Set `ZMQ_CONFLATE` on the `SUB` socket. Only the newest iterate is ever useful, and an unbounded
queue converts a backlog into unbounded staleness, which is much worse than a drop.

Built without libzmq: the constructor throws `std::runtime_error` naming the missing dependency
([03_BUILD_SYSTEM.md §3.3](03_BUILD_SYSTEM.md)). **Never a silent no-op transport.**

## §9.10 The parity contract

The C++ and Python implementations MUST agree on: the update equations, the phase order, the
residual definitions, the block ordering, the flattening order, and the adaptive-`ρ` rule.

`PythonParityTest` ([§10.4](10_TESTS.md), fixture in [§10.5](10_TESTS.md)) pins the first 20
iterates elementwise to 1e-8 with `alpha = 1.0` and `adaptive_rho` off, and the converged `u₀` to
1e-6.

**If parity breaks, one of the two implementations changed.** Find out which. Regenerating the
fixture to make a red test go green is not an option, and it is explicitly out of bounds
([§10.5](10_TESTS.md)).
