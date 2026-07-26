# Implementation guide

Target: **deepseek-v4-flash**, implementing the skeleton in this repository.

Every skeleton file contains `TODO [GUIDE x.y]` markers. Each marker points at the
correspondingly numbered subsection below. Read the subsection before writing the body.

---

## 0. Ground rules

**0.1 — Do not change public signatures.** The skeleton's function names, argument names,
argument order, and return types are the contract. Tests, notebooks, the C++ kernel, and
the derivations are all written against them. If a signature is genuinely wrong, change it
*and* update every caller *and* note it at the top of your response — do not change it
silently.

**0.2 — Do not delete the docstrings.** They are the specification. Add to them if you
learn something while implementing; never replace them with a one-liner.

**0.3 — Delete a `TODO [GUIDE x.y]` comment only when that item is fully done**, including
its tests. A leftover TODO is a correct signal; a deleted TODO over an unfinished
implementation is a lie that costs hours later.

**0.4 — The Python reference and the C++ kernel implement the same math.** Not "the same
approach" — the same equations, the same iteration order, the same residual definitions.
`cpp_admm/test/test_admm_kernel.cpp::PythonParityTest` compares them numerically. If you
change a formula in one, change it in both in the same commit.

**0.5 — No global information in an agent update.** The whole claim of this repository is
that agent `i` computes using only data from `closed_neighborhood(i)`. Residual norms,
objective sums, and progress bars are logging only. Never let one feed back into an update.
`test_no_global_information_used` enforces this; do not weaken it.

**0.6 — Prefer failing loudly at setup over degrading silently at runtime.** A bad
neighbor list, a formation edge with no communication link, a follower unreachable from any
leader: all of these produce a controller that *runs* and converges to the wrong thing.
Raise at construction.

**0.7 — When something cannot be implemented as specified, stop and say so.** Do not
substitute a simpler algorithm and leave the docstring describing the original. If the
CVXPY DPP path does not work as described in 4.3, say that and propose the alternative
rather than silently rebuilding the problem every solve.

**0.8 — Style.** Python: `black -l 100`, `ruff`, full type hints (`mypy
--disallow-untyped-defs` passes). C++: 17, `-Wall -Wextra -Wpedantic` clean, `ament_uncrustify`
clean. No `using namespace` at file scope.

---

## 1. Build order

Implement in this order. Each phase has an acceptance gate; do not start the next phase
until the gate passes. The dependency graph is real — building phase 3 before phase 2
means debugging the ADMM loop and the QP simultaneously, which is the single most
expensive mistake available here.

| Phase | Files | Gate |
| --- | --- | --- |
| 1 | `communication_graph.py` | Graph tests in `test_formation_consensus.py` pass; `laplacian`/`algebraic_connectivity` match hand-computed values for path and cycle on 4 nodes |
| 2 | `per_agent_solver.py` (model + dataclasses) | `test_prediction_matrices_match_rollout`, `test_position_prediction_is_row_subset` pass |
| 3 | `formation_constraints.py` | All geometry and rigidity tests pass |
| 4 | `per_agent_solver.py` (`CvxpyAgentSolver`) | A single-agent solve returns a dynamically consistent, limit-respecting trajectory |
| 5 | `consensus_admm.py` (`ConsensusADMM`) | `test_matches_centralized_solution` passes — **the gate that matters** |
| 6 | `consensus_admm.py` (`DistributedMPC`) | `test_agents_reach_formation` passes |
| 7 | `plotting.py` | Notebooks 02 and 03 run end to end |
| 8 | Notebooks 01–05 | `pytest --nbmake notebooks/` passes |
| 9 | `cpp_admm` (`per_agent_qp`, `admm_kernel`) | `PythonParityTest` passes |
| 10 | `cpp_admm` (`consensus_node`, launch) | `ros2 launch cpp_admm 4_agent_admm.launch.py` runs 4 processes to a formation |
| 11 | `analysis/`, `docs/`, media, README | LaTeX compiles; media regenerate from committed code |

---

## 2. Conventions

### 2.1 — Package exports (`python/distributed_mpc_admm/__init__.py`)

Replace the TODO with plain top-level imports of every name in `__all__`. Keep
`__all__` and the imports in sync; a name in one and not the other is a lint failure
waiting to happen.

Guard nothing behind try/except. If CVXPY is missing the package should fail to import
with a clear `ImportError`, not half-work.

### 2.2 — Array shapes

| Object | Shape | Notes |
| --- | --- | --- |
| agent state | `(4,)` | `[px, py, vx, vy]` |
| state log | `(K+1, N, 4)` | includes the initial condition |
| input sequence | `(T, 2)` | one row per horizon step |
| trajectory block `y`, `z`, `lam` | `(T, 2)` | position only |
| stacked prediction `X` | `(T*4,)` | flattened **C order** |

### 2.3 — Flattening order: time-major, C order, everywhere

A `(T, 2)` block flattens to index `t*2 + d`. This is `numpy`'s default (`.ravel()`,
`.reshape(-1)`), it is what `Phi`/`Gamma` are built for, and it is what the C++ side uses.

**CVXPY trap.** `cp.vec` historically defaults to Fortran (column-major) order. Do not
rely on it. Declare the CVXPY variables **flat** — `cp.Variable(T * dim)` — and reshape
only for readability with an explicit `order="C"`. This removes the ambiguity entirely
rather than depending on a version-specific default.

Write `test_prediction_matrices_match_rollout` first and make it fail before `Gamma` is
built. It catches both the flattening order and the horizon off-by-one, which are the two
bugs that will otherwise surface as "ADMM converges to something slightly wrong".

### 2.4 — Horizon indexing

`X = Phi @ x0 + Gamma @ U` covers `t = 1 .. T`. The initial state `x0` is **not** a row of
`X`. Every formula in this guide and in the derivations uses that convention.

### 2.5 — Randomness

Every stochastic path takes an explicit `rng: np.random.Generator | int | None`. Never call
`np.random.*` module-level functions. Any test or notebook result that cannot be reproduced
from a seed is not a result.

---

## 3. `communication_graph.py`

### 3.1 — `CommunicationGraph.__init__`

Internal state:

```python
self._n = n_agents
self._edges: frozenset[tuple[int, int]]        # canonical (min, max)
self._weights: dict[tuple[int, int], float]    # canonical keys
self._adj_cache: np.ndarray | None = None
self._lap_cache: dict[bool, np.ndarray] = {}
```

Validation, all raising `ValueError`: `n_agents < 1`; any id outside `[0, n_agents)`; `i == j`.
Duplicate and reversed-duplicate edges collapse silently — that is normal input, not an error.

`add_edge` / `remove_edge` must invalidate both caches. Forgetting this gives a graph whose
`laplacian()` describes a topology it no longer has, and every downstream number is then
quietly wrong.

`__eq__` compares `(n_agents, edges, weights)`. `__hash__` should be left undefined
(mutable object).

`neighbors` returns a **sorted tuple**, not a set or list. The order is load-bearing: it
defines the block ordering of the per-agent decision vector in both Python and C++.

`closed_neighborhood(i)` is `tuple(sorted(neighbors(i) + (i,)))` — the self index sits in
sorted position, not at the front. Same in C++.

`contributors(j)` returns `closed_neighborhood(j)` for undirected graphs. Keep it a
separate method; the z-update reads from it and conflating the two makes a future directed
variant a rewrite rather than an override.

Factories: `complete`, `cycle` (`i -- i+1 mod N`; for `N == 2` this is a single edge, not a
doubled one — handle it), `path`, `star`, `random_connected` (resample whole graphs, do not
patch a disconnected one — patching biases the degree distribution and invalidates the
robustness study).

### 3.2 — Spectral quantities

```python
L = D - A                      # normalized=False
L_norm = I - D^{-1/2} A D^{-1/2}   # normalized=True; isolated nodes -> row of zeros
```

Use `scipy.linalg.eigh` (symmetric), not `numpy.linalg.eig`. `algebraic_connectivity`
returns the second-smallest eigenvalue, clipped at 0 from below — the smallest is
analytically 0 and floating point routinely delivers `-1e-16`, which then propagates as a
`nan` through a `sqrt` three functions later.

`is_connected` should test `algebraic_connectivity() > 1e-10`, not run a separate BFS.
One definition of connectivity, one code path.

### 3.3 — `TimeVaryingGraph`

Normalise both construction modes into one callable in `__init__`:

```python
if callable(schedule):
    self._fn = schedule
else:
    graphs = tuple(schedule)
    # validate identical n_agents across all of them
    if mode == "hold":
        self._fn = lambda k: graphs[min(k, len(graphs) - 1)]
    elif mode == "cycle":
        self._fn = lambda k: graphs[k % len(graphs)]
```

Cache `at(k)` results in a dict — a callable schedule may be expensive and `at` is called
in the control loop and again in every plotting routine.

`switching(graphs, dwell_time, mode)` maps step `k` to graph index `k // dwell_time`, then
applies `mode`.

`union_over(k0, k1)` returns a new graph with the union of the edge sets. `is_jointly_connected`
is `union_over(...).is_connected()`.

`switch_steps` returns the `k` in the window where `at(k) != at(k-1)`, using
`CommunicationGraph.__eq__`.

### 3.4 — `LossyChannel`

State:

```python
self._mailbox: dict[tuple[int, int], Message]           # (receiver, subject) -> freshest arrived
self._inflight: list[tuple[int, Message]]               # (arrival_iteration, message)
self._stats = ChannelStats()
```

`send(message, iteration)`:
1. `stats.sent += 1`, `stats.bytes_sent += payload.nbytes`
2. draw loss: `if self._rng.random() < loss_prob:` → `stats.dropped += 1`, return `False`
3. draw delay: `d = self._rng.integers(0, max_delay + 1)`
4. push `(iteration + d, message)` onto `_inflight`, return `True`

`advance(iteration)` moves everything with `arrival <= iteration` into `_mailbox`, keeping
the one with the **largest `admm_iteration`** when several arrive for the same
`(receiver, subject)` — arrival order and production order are not the same thing under a
random delay, and taking the last arrival silently reorders time.

`receive` returns the mailbox entry or `None`, and records
`iteration - message.admm_iteration` in `stats.staleness_histogram`.

`set_graph` does **not** clear the mailboxes. An agent keeps the last thing it heard from a
node that has since disconnected. That is both realistic and the behaviour the
split/merge experiment depends on.

### 3.5 — `communication_load`

Per ADMM iteration, agent `i` sends `|N_i|` local copies plus `|N_i|` consensus broadcasts,
each `horizon * dim` floats:

```
packets_per_iteration = 2 * sum_i |N_i| = 4 * |E|
bytes = packets_per_iteration * admm_iterations * horizon * dim * float_bytes
```

Return `{"packets", "bytes", "bytes_per_agent", "packets_per_iteration"}`. The analysis
notebook cross-checks this closed form against `LossyChannel.stats` from a real run with
`loss_prob = 0` — if they disagree, the model is wrong, not the measurement.

---

## 4. `per_agent_solver.py` and `formation_constraints.py`

### 4.1 — `DoubleIntegrator`

```
A = [[I2, dt*I2],
     [ 0,    I2]]

B = [[0.5*dt^2*I2],
     [   dt*I2   ]]
```

Prediction over `t = 1..T`, with `x_t = A^t x0 + sum_{s=0}^{t-1} A^{t-1-s} B u_s`:

* `Phi` block row `t` (0-based row `r`, time `t = r+1`) is `A^(r+1)`
* `Gamma` block `(r, s)` is `A^(r-s) @ B` for `s <= r`, else zero

Build the `A` powers **iteratively** (`P = I; for r in range(T): P = A @ P; ...`). Computing
`np.linalg.matrix_power` per block is `O(T^2)` matrix products for no benefit.

`position_prediction_matrices` selects rows `[t*4, t*4+1]` for each `t` — implement it as an
index array applied to the full matrices, not as a separate derivation. Two derivations of
the same quantity will eventually disagree.

Cache on `(dt, dim, horizon)`. `DoubleIntegrator` is a frozen dataclass, so use a
module-level `functools.lru_cache` on a free function that the methods delegate to, rather
than trying to mutate the instance.

### 4.2 — The local problem

Agent `i`'s decision variables: `U_i` (`T x 2`) and `y_i^j` (`T x 2`) for every
`j` in `closed_neighborhood(i)`.

Objective:

```
f_i =  sum_{t=1..T} [ q_position * ||y_i^i[t] - ref_i[t]||^2
                    + q_velocity * ||v_i[t]||^2
                    + r_input    * ||U_i[t]||^2 ]
     + r_rate * sum_{t=1..T} ||U_i[t] - U_i[t-1]||^2          (U_i[0] := u_prev, or U_i[1])
     + p_terminal * ||y_i^i[T] - ref_i[T]||^2
     + w_formation * sum_{j in N_i} sum_{t=1..T} ||(y_i^i[t] - y_i^j[t]) - d_ij||^2

penalty = (rho/2) * sum_{j in Ncl(i)} || y_i^j - z^j + lam_i^j ||_F^2
```

Constraints:

```
y_i^i          = Phi_p @ x0 + Gamma_p @ U_i        (flattened, C order)
|U_i|         <= u_max                             (elementwise)
|V_i|         <= v_max     where V_i = Phi_v x0 + Gamma_v U_i
p_min <= y_i^i <= p_max                            (if set)
```

The copies `y_i^j` for `j != i` carry **no** dynamics constraint. The consensus penalty is
the only thing pinning them down. That is precisely what makes the problem separable — if
you find yourself wanting to constrain them, the splitting has been misunderstood.

> **Double-counting of formation edges — read this before writing the centralized
> reference.** Edge `(i, j)` appears in `f_i` *and* in `f_j`, each with weight
> `w_formation`. The sum `sum_i f_i` therefore weights each edge by `2 * w_formation`. The
> centralized QP in `solve_centralized` must use `2 * w_formation` per edge, or
> `test_matches_centralized_solution` fails by a factor that looks like a convergence
> problem and is not. Keep the per-agent weight as `w_formation` (it is the tunable a user
> reasons about) and put the factor of 2 in the reference implementation, with a comment
> pointing here.

### 4.3 — `CvxpyAgentSolver`

Build the problem **once** in `__init__`:

```python
T, d = horizon, model.dim
self._U = cp.Variable(T * d, name="U")
self._y = {j: cp.Variable(T * d, name=f"y_{j}") for j in neighborhood}

self._x0_p  = cp.Parameter(model.n_states)
self._ref_p = cp.Parameter(T * d)
self._rho_p = cp.Parameter(nonneg=True)
self._z_p   = {j: cp.Parameter(T * d) for j in neighborhood}
self._lam_p = {j: cp.Parameter(T * d) for j in neighborhood}
```

`solve()` assigns `.value` on the parameters and calls
`self._problem.solve(solver=cp.OSQP, warm_start=True)`. It **never** rebuilds. Rebuilding
inside `solve` makes a 4-agent closed-loop run roughly two orders of magnitude slower and
is the most common way to get this class wrong.

**DPP compliance.** `rho_p * cp.sum_squares(y - z_p + lam_p)` is a parameter times a
parameter-dependent expression, which is not DPP. CVXPY will re-canonicalise every solve
and you lose the entire benefit. Two ways out, in order of preference:

1. Make the parameter `sqrt_rho` and write `cp.sum_squares(sqrt_rho_p * (y - z_p + lam_p))`
   — still not DPP, `sqrt_rho_p` multiplies a parameter.
2. **Recommended:** expand the penalty so parameters appear only affinely:

   ```
   (rho/2)*||y - z + lam||^2  ==  (rho/2)*||y||^2 - rho*<y, z - lam> + const(y)
   ```

   Declare a single parameter `w_p = rho * (z - lam)` per neighbor (assigned numerically in
   `_assign_parameters`) and a scalar `rho_p`. Then the objective contains
   `0.5 * rho_p * cp.sum_squares(y)` — parameter times a parameter-free expression, which
   *is* DPP — and `- y @ w_p`, which is affine in the parameter. The dropped constant does
   not affect the argmin; add it back in `_extract_solution` only if `local_objective` is
   being compared against the centralized value.

   Verify with `problem.is_dcp(dpp=True)` in `__init__` and raise if it is `False`. Do not
   let this degrade silently.

`_assign_parameters` must raise `KeyError` with a useful message if `data.z` or `data.lam`
is missing a neighbor. Silently defaulting to zero produces a run that converges to the
wrong formation.

`warm_start(solution)` writes `solution.inputs.ravel()` into `self._U.value` and each copy
into `self._y[j].value`; CVXPY's `warm_start=True` then picks them up.

Handle a non-optimal status by raising `RuntimeError` in the Python reference. Python is
the correctness reference — it should be loud. The C++ kernel is the one that degrades
gracefully.

### 4.4 — `build_reference_trajectory`

```python
idx = np.clip(np.arange(start_step + 1, start_step + horizon + 1), 0, len(waypoints) - 1)
return waypoints[idx]
```

Note `start_step + 1`: the horizon covers `t = 1..T`, matching 2.4.

### 4.5 — `FormationSpec`

Offsets are stored **mean-centred**; every factory must subtract the mean before returning.
`test_offsets_are_mean_centred` checks it.

`anchor_offsets()` applies scale then rotation:

```python
R = np.array([[cos, -sin], [sin, cos]])
return self.scale * (self.offsets @ R.T)
```

`relative_offset(i, j)` is `anchor_offsets()[i] - anchor_offsets()[j]` — computed from the
transformed offsets, so `scale` and `rotation` propagate. This is antisymmetric by
construction; do not add a special case for `i == j`.

`edge_offsets(agent)` returns `{j: relative_offset(agent, j) for j in graph.neighbors(agent)}`.

Factories:
* `regular_polygon(n, radius)` — angles `2*pi*k/n`; already mean-centred, but subtract
  anyway so the invariant holds by construction rather than by luck.
* `line(n, spacing, heading)` — positions `(k - (n-1)/2) * spacing` along `heading`.
* `v_shape(n, spacing, half_angle)` — agent 0 at the apex, then alternate arms.
* `grid(rows, cols, spacing)` — default graph is the 4-neighbour lattice, not complete.
* `rendezvous(n, graph)` — all zeros.

### 4.6 — Rigidity

Rigidity matrix `R(p)`, one row per edge `(i, j)`, `dim*N` columns:

```
row[(i,j)][i*dim : (i+1)*dim] =  (p_i - p_j)
row[(i,j)][j*dim : (j+1)*dim] = -(p_i - p_j)
```

`is_infinitesimally_rigid`: `matrix_rank(R, tol) == dim*N - dim*(dim+1)/2` (`= 2N - 3` in
2D). Use `np.linalg.matrix_rank` with an explicit `tol`; the default is scale-dependent and
a formation scaled by 10 will change its answer.

`rigidity_eigenvalue`: eigenvalues of `R.T @ R`, sorted ascending, return index
`dim*(dim+1)/2` (i.e. the first one after the trivial ones). Clip at 0.

Note in the docstring that this repo's *offset* encoding fixes orientation and so converges
even for a flexible framework; rigidity is reported as a diagnostic and to make the
comparison against distance-based encodings meaningful. Do not overstate it.

### 4.7 — `LeaderFollowerSpec.validate_against`

BFS from the leader set over `graph`. Any unreached agent raises `ValueError` naming the
agent. An unreachable follower has nothing anchoring its absolute position, so the
formation converges in shape and drifts as a rigid body — a failure that looks like a
tuning problem in a plot and is not.

### 4.8 — Error metrics

`formation_error(positions, spec, anchor_reference)`:
* `per_edge[(i,j)] = ||(p_i - p_j) - d_ij||` for each edge in `spec.graph`
* `edge_rms = sqrt(mean(per_edge^2))`, `edge_max = max(per_edge)`
* `centroid_error = ||mean(positions, axis=0) - anchor_reference||`, or `0.0` when
  `anchor_reference is None`

`settling_step(errors, tolerance, hold)`: smallest `k` such that
`np.all(errors[k:k+hold] < tolerance)` **and** the condition holds for the whole tail.
Return `None` otherwise. A transient dip below tolerance is not settling.

---

## 5. `consensus_admm.py`

### 5.1 — The iteration

```
for k in 0 .. max_iterations-1:

    # 1. x-update, one QP per agent, no communication
    for i in agents:
        (U_i, y_i) = solvers[i].solve(LocalProblemData(x0=x0[i], rho=rho,
                                                       z=z_received[i], lam=lam[i], ...))

    # 2. over-relaxation
    y_hat[i][j] = alpha * y[i][j] + (1 - alpha) * z[j]

    # 3. z-update, executed at subject j over its contributors
    for j in agents:
        C = graph.contributors(j)
        z_new[j] = mean over i in C of (y_hat[i][j] + lam[i][j])

    # 4. broadcast z, obtaining each agent's received view
    z_received = broadcast(z_new)

    # 5. dual update
    lam[i][j] += y_hat[i][j] - z_received[i][j]

    # 6. residuals, tolerances, optional rho update
```

Order matters. Do not move the relaxation after the z-update, and do not use `z_new`
in step 5 for agents that did not receive it.

### 5.2 — `ConsensusADMM.__init__`

```python
self._graph = graph
self._solvers = dict(solvers)
self._T, self._dim = horizon, dim
self._opts = options or ADMMOptions()
self._channel = channel
self._y: dict[int, dict[int, np.ndarray]] = {}       # (T, dim) blocks
self._lam: dict[int, dict[int, np.ndarray]] = {}
self._z: dict[int, np.ndarray] = {}
self._z_last_known: dict[int, dict[int, np.ndarray]] = {}   # per-agent cache under loss
```

Validate that `solvers` covers `range(graph.n_agents)` exactly. Zero-initialise every block.

### 5.3 — The z-update, and why the divisor is what it is

```python
z_new = {}
for j in range(N):
    contributions = [y_hat[i][j] + lam[i][j] for i in contributors(j) if received(i, j)]
    z_new[j] = np.mean(contributions, axis=0)
```

**The divisor is the number of received contributions, not `|C_j|`.** A missing
contribution is excluded from the average; it is never replaced by a stale value. Averaging
a stale term with fresh ones biases `z` in a way that is far harder to characterise than a
term that is simply absent — and the characterisation is the point of the whole exercise.

The `z`-broadcast in step 4 is the opposite: a missing `z^j` *does* fall back to
`_z_last_known[i][j]`, because agent `i` has to put *something* in its next QP.

**Invariant worth asserting.** In the synchronous, lossless case, summing the dual update
over `i in C_j` gives `sum_i lam[i][j] == 0` exactly, from iteration 1 onward. Consequently
`z[j]` reduces to `mean_i y_hat[i][j]`. Add this as a debug assertion behind
`options.verbose` — if it fails, the dual update and the z-update disagree, and no amount
of tuning will fix it.

### 5.4 — Residuals and stopping

```
n_dual   = sum_i |Ncl(i)| * T * dim

r = sqrt( sum_i sum_{j in Ncl(i)} ||y[i][j] - z[j]||_F^2 )
s = rho * sqrt( sum_i sum_{j in Ncl(i)} ||z[j] - z_prev[j]||_F^2 )

eps_pri  = sqrt(n_dual)*eps_abs + eps_rel * max( ||y||_F , ||z_stacked||_F )
eps_dual = sqrt(n_dual)*eps_abs + eps_rel * rho * ||lam||_F
```

where `||z_stacked||` stacks `z[j]` once per `(i, j)` pair, not once per `j` — it has to
live in the same space as `y` for the comparison to mean anything.

Stop when `r <= eps_pri` **and** `s <= eps_dual`. Use the un-relaxed `y` for `r`.

`consensus_gap = max over (i, j) of max|y[i][j] - z[j]|`. Log it: the Frobenius residual
can look small while one agent is badly off, and this is the number that exposes it.

### 5.5 — Adaptive rho

```python
if primal > mu * dual:
    factor = tau
elif dual > mu * primal:
    factor = 1.0 / tau
else:
    factor = 1.0

new_rho = float(np.clip(rho * factor, rho_min, rho_max))
if new_rho != rho:
    actual = new_rho / rho
    for i in lam:
        for j in lam[i]:
            lam[i][j] /= actual          # scaled duals absorb rho
```

**The rescaling is mandatory.** `lam` here is `u / rho`; changing `rho` without dividing
`lam` by the same factor silently changes the dual iterate. The symptom is a run that
converges for fixed `rho` and stalls with adaptation on.
`test_converges_for_range_of_rho` catches it.

Default `adaptive_rho=False`. Turn it off before quoting any convergence-rate number — the
linear-rate result assumes fixed `rho`.

### 5.6 — `DistributedMPC`

Cache solvers keyed by `(agent_id, neighborhood)`:

```python
self._solver_cache: dict[tuple[int, tuple[int, ...]], PerAgentSolver] = {}
```

`_rebuild_solvers(graph)` rebuilds only agents whose closed neighborhood changed. Building a
`CvxpyAgentSolver` is expensive (problem compilation); rebuilding all `N` at every switch
dominates the runtime of the switching notebook.

`run(x0, n_steps)`:

```
x = x0.copy()
for k in range(n_steps):
    graph = schedule.at(k)                       # rebuild solvers if changed
    refs = self._references_at(k)
    result = admm.solve(x, refs, offsets, initial_guess=prev.shifted() if warm_start else None)
    u = result.first_inputs()
    x = model.simulate_step(x, u) + process_noise
    log everything, including graph and history
```

Apply the input to the **true** state, not to the predicted one. Add process noise after
the dynamics update, measurement noise only to what is passed into the next solve. Conflating
the two makes the noise study meaningless.

`ADMMResult.shifted()` drops row 0 of every `(T, dim)` block and duplicates the last row.
Shift `lam` the same way — zeroing the duals throws away most of the warm-start benefit and
`test_warm_start_reduces_iterations` will fail.

`SimulationLog.save` uses `np.savez_compressed`. `graphs` and `histories` do not survive
`npz` directly: serialise `graphs` as a `(K, N, N)` adjacency stack and `histories` as
stacked residual arrays plus an index. Do not use `pickle` — these files are committed and
regenerated by CI.

---

## 6. `cpp_admm`

### 6.1 — Build

`CMakeLists.txt` and `package.xml` are written and should configure as-is. Verify with:

```bash
colcon build --packages-select cpp_admm --cmake-args -DCMAKE_BUILD_TYPE=Release
```

If `osqp_vendor` does not resolve via rosdep, build OSQP from source
(`-DCMAKE_INSTALL_PREFIX=/usr/local`) and note it in `docs/README_math.md` section 10 and in
the CI workflow. Do not vendor a copy of OSQP into this repo.

### 6.2 — `NeighborMessage::byte_size`

`sizeof(sender) + sizeof(subject) + sizeof(admm_iteration) + sizeof(control_step) +
payload.size() * sizeof(double)`. Used only for bandwidth accounting; it must match the
analytic model in 3.5 or the comparison in the notebook is not a check of anything.

### 6.3 — `InProcessTransport`

Deterministic loopback for tests. `connect(peers)` populates a shared registry mapping agent
id to `Impl*`. `publish` pushes directly into the destination's inbox deque (guarded by a
mutex, since a future multi-threaded test will need it), applying the same loss and delay
model as `LossyChannel` in 3.4 — same semantics, so a C++ result and a Python result on the
same seed are comparable.

`poll` ignores the timeout (delivery is synchronous) and drains the inbox for the requested
`MessageKind`.

### 6.4 — `ZeroMqTransport`

PIMPL so `<zmq.hpp>` stays out of the header. One `PUB` socket bound to `bind_endpoint`, one
`SUB` socket connected to each neighbor endpoint. Topic strings `"admm/<kind>/<subject>"`,
precomputed at construction — building them per publish allocates in the hot path.

Set `ZMQ_CONFLATE` on the `SUB` socket. Only the newest iterate is ever useful; an unbounded
queue converts a backlog into unbounded staleness, which is much worse than a drop.

When built without libzmq (`CPP_ADMM_WITH_ZMQ` undefined), the constructor throws
`std::runtime_error` with a message naming the missing dependency. It must not silently
become a no-op transport.

### 6.5 — `AdmmKernel::Impl`

Everything sized in `configure()`, nothing resized afterwards. Reserve `rx_buffer` to
`closed_nbhd.size()` and reuse it across polls.

`iterate()` runs exactly the phase order in 5.1. `NoHeapAllocationDuringIterate` is a real
test — write `iterate()` assuming it will be checked, because it will.

`solve()`:

```
stats.reset();
for (int k = 0; k < options.max_iterations; ++k) {
    if (iterate()) { stats.converged = true; break; }
    if (options.max_staleness > 0 && stats.max_staleness_seen > options.max_staleness) break;
}
```

`shiftWarmStart()` shifts `y`, `z`, and `lam` by one `dim`-sized step, duplicating the tail.

`computeResiduals()` computes this agent's **local** contribution only. The kernel must be
able to decide when to stop using what a real agent actually has. Publish the local residual
in diagnostics and let an offline monitor sum them.

`setNeighbors()` reallocates and re-setups OSQP. Preserve `y`/`lam` for surviving neighbors,
zero-init new ones, drop the rest. Never call it from a real-time thread; the header says so
and every call site should too.

### 6.6 — `toString(QpStatus)`

Trivial, but return a `static constexpr` string per enumerator — do not build a
`std::string`, this is called from log paths.

`QpSolution::ok()` returns true for `kSolved` and `kSolvedInaccurate`. The outer ADMM
iteration tolerates inexact inner solves; that is a documented property of the method, not a
shortcut.

### 6.7 — `PerAgentQp::Impl`

Variable layout (also in the header — keep them in sync):

```
theta = [ U | y^{c_0} | y^{c_1} | ... | y^{c_{M-1}} ]
```

`M = |closed neighborhood|`, blocks in ascending agent-id order, each `horizon*dim` long,
time-major within a block. `n = (1 + M) * horizon * dim`.

Constraint rows, in order: dynamics equality (`horizon*dim`), input box (`horizon*dim`),
velocity box (`horizon*dim`).

Hessian blocks (note the factor of 2 — OSQP minimises `0.5 x'Px + q'x`, so a cost
`w||x||^2` contributes `2w` to `P`):

| Block | Contribution |
| --- | --- |
| `U` | `2*(r_input*I + r_rate*D'D + q_velocity * Gamma_v' Gamma_v)` |
| `y^self` | `2*(q_position*I + terminal + sum_j w_formation * I) + rho*I` |
| `y^j`, `j != self` | `2*w_formation*I + rho*I` |
| cross `y^self`/`y^j` | `-2*w_formation*I` |

**Do not omit the cross terms.** Without them the formation cost is
`w(||y_i||^2 + ||y_j||^2)` instead of `w||y_i - y_j - d||^2` — a different problem that
still converges, to the wrong answer.

`Gamma_v' Gamma_v` is dense. That is expected and is why `q_velocity` defaults small.

Build `P` upper-triangular in CSC. OSQP requires it and misbehaves quietly if handed the
full symmetric matrix — `SetupSucceedsAndHessianIsPsd` checks it.

Record `rho_diag_indices`: the positions in `P.valuePtr()` of the `rho*I` diagonal entries.
`updateRho` overwrites those and calls `osqp_update_data_mat` with the same index list. A
wrong index list corrupts the Hessian into a still-solvable but wrong problem —
`UpdateRhoTouchesOnlyDiagonalEntries` is the guard.

Skip `updateRho` entirely when `rho` is unchanged; it triggers a refactorisation costing
about as much as a solve.

Cache the static part of `q` so `updateConsensus` is one axpy per block.

### 6.8 / 6.9 — `RosTransport` and `ConsensusNode`

The one non-obvious thing: **`poll()` must pump the executor, not sleep.** ROS 2
subscription callbacks only run while the executor is spinning. A `poll` that sleeps
guarantees an empty inbox, and the kernel then degrades to a fully asynchronous run while
appearing to work. Call `rclcpp::spin_some(node)` in a loop until the timeout expires or
every expected subject has arrived.

QoS for ADMM traffic: best-effort, volatile, depth 1. Reliable QoS is wrong here — a
retransmitted stale iterate is worse than a dropped one, because the kernel handles a miss
explicitly but cannot tell that a late arrival is late.

`controlStep` distinguishes two failure modes:
* **Not converged** — apply the input anyway, log throttled at WARN. An unconverged ADMM
  iterate is a suboptimal *feasible* input.
* **QP failed outright** — `enterSafeState`. Zero acceleration, throttled warning, recover
  only on fresh state, and reset the warm start on recovery since the stored iterate is
  stale by an unknown amount.

`onGraphUpdate` logs old and new neighborhoods at INFO. These events are what the thesis
analyses; they must be reconstructable from a bag file alone.

`main()` catches `std::exception` around construction, logs, and exits non-zero. A launch
file starting four agents must make a bad config obvious rather than leaving one agent
silently dead.

### 6.10 / 6.11 — Launch files

`build_topology` must produce edge lists **identical** to `CommunicationGraph` in Python —
same node numbering, same cycle orientation. The C++ and Python demos are compared against
each other and a differently-numbered cycle makes that comparison meaningless.

Factor `build_topology` / `build_offsets` into a shared `cpp_admm/launch/launch_utils.py`
imported by both launch files. Two divergent copies of the topology definition is the one
bug this pair of files cannot afford.

`schedule_edges` must be deterministic given `seed` — the split/merge GIF has to be
reproducible.

Note in the README that the switching schedule is published centrally while the control
stays distributed. It stands in for physical link availability. State it rather than letting
a reader assume otherwise and discover it in the code.

---

## 7. Tests

### 7.1 — C++ fixtures (`test_admm_kernel.cpp`)

`makeConfig(agent_id)` builds the 4-agent cycle, rendezvous, `horizon = 10`, `dt = 0.1`
configuration. A helper wires four kernels through `InProcessTransport` with no loss and
returns owning handles.

`PythonParityTest` loads `test/data/four_agent_reference.json`, exported from
`notebooks/02_4_agent_consensus.ipynb`: `x0`, weights, `rho`, and the first 20 iterates of
`y`, `z`, `lambda`. Compare elementwise to `1e-8` with `alpha = 1.0` and `adaptive_rho`
off.

**Regenerating that fixture to make a red test go green is not an option.** If parity
breaks, one of the two implementations changed; find out which and fix it.

### 7.2 — Python tests

Both test files are fully specified in their docstrings — implement exactly what each
docstring describes. Three notes:

* `test_residuals_decrease` must **not** assert per-iteration monotonicity. With
  over-relaxation or adaptive `rho`, the primal residual legitimately rises on individual
  steps. Assert that the minimum over successive 10-iteration windows is non-increasing.
* `test_packet_loss_degrades_gracefully` must not assert convergence. The point of that
  experiment is that the guarantee is lost. Assert boundedness, no NaN, and monotone
  degradation with `loss_prob` across seeds.
* `test_no_global_information_used` wraps each solver's `solve` and asserts the
  `LocalProblemData` it receives has `z`/`lam` keys only within that agent's closed
  neighborhood. Cheap to write, and it is the property the entire repository claims.

Keep the full suite (excluding `@pytest.mark.slow`) under 60 s. If it creeps past that,
shrink horizons, not coverage.

---

## 8. Notebooks

Each notebook skeleton has markdown cells stating what each section must show and a code
cell with the TODO. Rules:

* **Every notebook must run top to bottom on a clean kernel**, in CI, with no manual steps.
* Commit them **with outputs**, so GitHub renders the figures. Notebook outputs are the
  demo for a reader who will not clone the repository.
* Seed everything. A number in a notebook that cannot be reproduced is not a result.
* If notebooks 04 and 05 exceed the CI runner budget, gate sweep sizes on an `NB_CI`
  environment variable. Do **not** drop them from CI — an unexecuted notebook is a broken
  notebook nobody knows about.

**8.1** (`01_admm_intuition`) — scalar and 2-agent ADMM written *inline*, not imported. The
reader must see all six lines.
**8.2** (`02_4_agent_consensus`) — the centralized cross-check lives here. Also export
`test/data/four_agent_reference.json` for the C++ parity test from this notebook.
**8.3** (`03_formation_control`) — produces `media/4_agent_formation.gif`.
**8.4** (`04_switching_topology`) — produces `media/topology_switch.gif`. The final markdown
cell maps each experiment to the assumption it violates in `convergence_proof.tex`; that
cell is the bridge to the thesis and is the most valuable text in the repository.
**8.5** (`05_convergence_analysis`) — 30 seeds per configuration, fixed `rho`, tolerances
`1e-6`. State the protocol in the first cell and do not deviate; comparability across cells
is half the value. Produces `media/convergence_curves.png`.
**8.6–8.8** (`analysis/`) — consume saved `.npz` logs rather than re-simulating, so the
figures regenerate in seconds. In 8.7, cross-check the analytic bandwidth model against
`LossyChannel.stats`. In 8.8, plot outcomes against `lambda_2` rather than against the
failure probability, and if the collapse is *not* sharper in that coordinate, report that —
it would mean the connectivity story is weaker than claimed.

---

## 9. Docs

**9.1** `README_math.md` — fill every section. The notation table is the contract; no symbol
may appear in a `.tex` file without an entry there first.

**9.2** `COMPARISON_VS_DUAL_DECOMP.md` — must contain **measured** numbers from a dual
decomposition baseline implemented against the same problem instances, same seeds, same
hardware. Reuse `PerAgentSolver` with `rho = 0` and a dual-ascent outer loop. Section 6
("when dual decomposition still wins") is not optional: a comparison concluding that the
method the author implemented wins in every case reads as advocacy and will be discounted
by exactly the readers this repo targets.

**9.3–9.5** The three `.tex` documents. Section-by-section TODOs are in the files.
`convergence_proof.tex` section 7 is the payload — it must be precise about which assumption
each experiment breaks, and it must label the "structured failure" claim as a
measurement-supported conjecture, not a theorem.

**9.6** `preamble.tex` — one macro per symbol in the notation table. Macros live here only;
a macro redefined in an individual document is how three files drift into three notations.

**9.7** `references.bib` — verify every entry against the actual publication. A wrong page
range in a repository whose purpose is to demonstrate scholarly care is expensive.

---

## 10. CI

**10.1** Notebooks run via `nbmake` on push, not on every PR (they are slow and are not a
correctness gate).
**10.2** If `rosdep` cannot resolve `osqp_vendor`, build OSQP from source in the workflow
and remove the `|| true` — a silently skipped dependency install surfaces as a confusing
link error three steps later.
**10.3** The LaTeX job fails the build on a compile error. A repository whose headline
deliverable is a set of derivations must not ship documents that do not compile.

---

## 11. Media and README

**11.1** All three media files are generated by `python -m distributed_mpc_admm.plotting`.
Keep each GIF under ~5 MB (GitHub will not autoplay above that). Use
`plotting.apply_style("readme")` so figures are legible in dark mode — a white-background
PNG in a dark README reads as a rendering bug. Annotate the split and merge instants *on the
frames* of the switching GIF, not only in the caption.

**11.2** README: hero GIF first, then three sentences on what this is, then the results
table with hardware and solver versions stated. The **"What this does not do"** section stays
above the fold. A reader who finds a limitation themselves discounts everything else; a
reader who is told it up front trusts the rest.

---

## 12. Pitfall index

Ordered by how much time each one costs when missed.

1. **Formation edge double-counting** (4.2) — centralized reference needs `2*w_formation`
   per edge. Presents as an ADMM convergence bug. Costs a day.
2. **Missing dual rescaling in adaptive rho** (5.5) — converges with fixed `rho`, stalls
   with adaptation.
3. **Missing cross terms in the C++ Hessian** (6.7) — a different problem that still
   converges, to the wrong answer.
4. **Flatten-order mismatch** (2.3) — Fortran vs C order between `Gamma` and CVXPY.
   Produces a plausible-looking wrong trajectory.
5. **Horizon off-by-one** (2.4) — `X` starting at `t=0` instead of `t=1`.
6. **Rebuilding the CVXPY problem inside `solve`** (4.3) — 100x slowdown, no wrong answers,
   so nothing fails except your patience.
7. **`poll()` sleeping instead of spinning** (6.8) — ROS demo silently degrades to fully
   asynchronous.
8. **Stale value substituted into the z-average** (5.3) — biases `z`; the correct behaviour
   is to shrink the divisor.
9. **Non-upper-triangular `P` handed to OSQP** (6.7) — quiet misbehaviour.
10. **Graph caches not invalidated on `add_edge`/`remove_edge`** (3.1) — every spectral
    quantity silently describes the old topology.
11. **`np.random` module functions instead of a seeded `Generator`** (2.5) — irreproducible
    results, discovered only when a reviewer asks.
12. **Zeroing duals on warm start** (5.6) — most of the warm-start benefit disappears.

---

## 13. Definition of done

- [ ] `pytest -m "not slow"` green on 3.10, 3.11, 3.12
- [ ] `ruff check .`, `black --check .`, `mypy distributed_mpc_admm` clean
- [ ] `pytest --nbmake notebooks/` green; all five notebooks committed with outputs
- [ ] `colcon build && colcon test` green, including `PythonParityTest`
- [ ] `ros2 launch cpp_admm 4_agent_admm.launch.py` drives 4 processes to a formation
- [ ] `ros2 launch cpp_admm time_varying_graph.launch.py schedule:=split_merge` reproduces
      the split/merge behaviour
- [ ] All three `.tex` documents compile; PDFs upload as a CI artifact
- [ ] `media/` regenerates from committed code via one command
- [ ] `docs/README_math.md` notation table covers every symbol used anywhere
- [ ] `COMPARISON_VS_DUAL_DECOMP.md` table populated with measured numbers
- [ ] README results table filled, with hardware and solver versions
- [ ] Zero `TODO [GUIDE` markers remain
