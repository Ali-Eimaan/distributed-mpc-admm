# §5 · The agent model and the local QP

**Governs:** `python/distributed_mpc_admm/per_agent_solver.py`
**Milestones:** M2 (§5.1–§5.2) and M4 (§5.3–§5.5)
**Done when:** M2 — `test_prediction_matrices_match_rollout` passes to 1e-12. M4 — a single-agent
solve returns a dynamically consistent, limit-respecting trajectory and
`problem.is_dcp(dpp=True)` is `True`.

This module owns the ADMM `x`-update: given the consensus trajectories `z^j` and the scaled duals
`λ_i^j` handed down by [06_ADMM.md](06_ADMM.md), agent `i` solves a small strictly convex QP over
its own inputs and its local copies of the neighbour trajectories.

---

## §5.1 `DoubleIntegrator`

Continuous `ṗ = v`, `v̇ = u`; exact zero-order-hold discretisation at sample time `dt`:

```
A = [[I₂, dt·I₂],        B = [[½dt²·I₂],
     [ 0,    I₂]]             [  dt·I₂ ]]
```

State ordering `[p₁..p_d, v₁..v_d]`, so `n_states = 2·dim`, `n_inputs = dim`.

Condensed prediction over `t = 1..T`, from `x_t = Aᵗ x₀ + Σ_{s=0}^{t−1} A^{t−1−s} B u_s`:

- `Phi` block row `r` (0-based; time `t = r+1`) is `A^(r+1)`
- `Gamma` block `(r, s)` is `A^(r−s) · B` for `s ≤ r`, else zero

Build the `A` powers **iteratively** (`P = I; for r in range(T): P = A @ P; …`). Calling
`np.linalg.matrix_power` per block is `O(T²)` matrix products for no benefit and it obscures the
recursion that makes the indexing checkable.

`position_prediction_matrices` MUST be an **index selection applied to the full matrices**, not a
separate derivation. Rows `[t·2d, t·2d+1, …, t·2d+d−1]` for each `t`. Two derivations of the same
quantity will eventually disagree, and the one that disagrees will be the one in the constraint.

Same for `velocity_prediction_matrices`, which feeds the velocity box.

Cache on `(dt, dim, horizon)`. `DoubleIntegrator` is a frozen dataclass, so put the cache on a
module-level `functools.lru_cache`d free function that the methods delegate to — do not try to
mutate the instance.

## §5.2 Flattening and horizon conventions

Restated here because everything below depends on them; the authoritative statement is
[§16.1](16_CONVENTIONS.md).

- A `(T, dim)` block flattens **time-major, C order**: index `t·dim + d`. This is NumPy's default
  (`.ravel()`, `.reshape(-1)`) and it is what `Gamma` is built for and what the C++ side uses.
- `X = Phi @ x0 + Gamma @ U` covers `t = 1..T`. **`x₀` is not a row of `X`.**

> **CVXPY trap.** `cp.vec` has historically defaulted to Fortran (column-major) order, and the
> availability of an `order=` argument varies by version. Do not depend on either. Declare the
> CVXPY variables **flat** — `cp.Variable(T * dim)` — and reshape only for readability, with an
> explicit `order="C"`. This removes the ambiguity rather than depending on a version-specific
> default.

Write `test_prediction_matrices_match_rollout` **first** and watch it fail before `Gamma` exists. It
catches both the flattening order and the horizon off-by-one, which are the two bugs that otherwise
surface much later as "ADMM converges to something slightly wrong".

## §5.3 The local problem

Agent `i`'s decision variables: `U_i` of shape `(T, dim)`, and `y_i^j` of shape `(T, dim)` for every
`j` in `closed_neighborhood(i)`.

**Objective**

```
f_i =  Σ_{t=1..T} [ q_position · ‖y_i^i[t] − ref_i[t]‖²
                  + q_velocity · ‖v_i[t]‖²
                  + r_input    · ‖U_i[t]‖² ]
     + r_rate      · Σ_{t=1..T} ‖U_i[t] − U_i[t−1]‖²        (U_i[0] := u_prev, else U_i[1])
     + p_terminal  · ‖y_i^i[T] − ref_i[T]‖²
     + w_formation · Σ_{j ∈ N_i} Σ_{t=1..T} ‖(y_i^i[t] − y_i^j[t]) − d_ij‖²

penalty = (ρ/2) · Σ_{j ∈ N̄_i} ‖ y_i^j − z^j + λ_i^j ‖_F²
```

**Constraints**

```
y_i^i            = Phi_p·x₀ + Gamma_p·U_i        (flattened, C order)
|U_i|           ≤ u_max                          (elementwise, ∞-norm not 2-norm)
|V_i|           ≤ v_max        where V_i = Phi_v·x₀ + Gamma_v·U_i
p_min ≤ y_i^i   ≤ p_max                          (if set)
```

**Only `y_i^i` carries a dynamics constraint.** The copies `y_i^j` for `j ≠ i` are free variables;
the consensus penalty is the only thing pinning them down. That is precisely what makes the problem
separable across agents. If you find yourself wanting to constrain them, the splitting has been
misunderstood — go back to [06_ADMM.md §6.1](06_ADMM.md).

### The formation double-count — read before writing the centralised reference

Edge `(i, j)` appears in `f_i` **and** in `f_j`, each with weight `w_formation`. The sum `Σ_i f_i`
therefore weights each edge by `2·w_formation`.

The centralised QP in `solve_centralized` ([§6.7](06_ADMM.md)) MUST use `2·w_formation` per edge, or
A2 fails by a factor that looks exactly like a convergence problem and is not.

Keep the per-agent weight as `w_formation` — it is the tunable a user reasons about — and put the
factor of two in the reference implementation, with a comment pointing here. This is the single
most expensive mistake available in this repository ([§16.9](16_CONVENTIONS.md), item 1).

## §5.4 `CvxpyAgentSolver`

**Build the problem once, in `__init__`.** Everything that changes between solves is a
`cp.Parameter`:

```python
T, d = horizon, model.dim
self._U = cp.Variable(T * d, name="U")
self._y = {j: cp.Variable(T * d, name=f"y_{j}") for j in neighborhood}

self._x0_p  = cp.Parameter(model.n_states)
self._ref_p = cp.Parameter(T * d)
self._rho_p = cp.Parameter(nonneg=True)
self._w_p   = {j: cp.Parameter(T * d) for j in neighborhood}   # see below
```

`solve()` assigns `.value` and calls `self._problem.solve(solver=cp.OSQP, warm_start=True)`. It
**never** rebuilds. Rebuilding inside `solve` makes a 4-agent closed-loop run roughly two orders of
magnitude slower and is the most common way to get this class wrong.

### DPP compliance — the part that actually bites

The natural expression

```python
self._rho_p * cp.sum_squares(y - z_p + lam_p)
```

is a parameter multiplying a parameter-dependent expression, which is **not DPP**. CVXPY accepts it
and re-canonicalises the whole problem on every solve, so you get correct answers at roughly the
speed of rebuilding — and no error tells you.

Expand the penalty so parameters appear only affinely:

```
(ρ/2)·‖y − z + λ‖²  =  (ρ/2)·‖y‖²  −  ρ·⟨y, z − λ⟩  +  const(y)
```

Declare one parameter `w_p[j]` per neighbour, assigned numerically as `rho * (z[j] - lam[j])`, plus
the scalar `rho_p`. The objective then contains `0.5 * rho_p * cp.sum_squares(y)` — a parameter
times a parameter-free expression, which *is* DPP — and `- y @ w_p[j]`, which is affine in the
parameter.

The dropped constant `(ρ/2)‖z − λ‖²` does not affect the argmin. Add it back in
`_extract_solution` only if `local_objective` is being compared against a centralised value.

**Assert `self._problem.is_dcp(dpp=True)` in `__init__` and raise if it is `False`** (risk V8,
[§2.1](02_ENVIRONMENT.md)). Do not let this degrade silently; a silent 100× slowdown becomes a
false conclusion about ADMM's cost in [§12.2](12_ANALYSIS.md).

### Other requirements

- `_assign_parameters` MUST raise `KeyError` with a useful message if `data.z` or `data.lam` is
  missing a neighbour. Silently defaulting to zero produces a run that converges to the wrong
  formation.
- `warm_start(solution)` writes `solution.inputs.ravel()` into `self._U.value` and each copy into
  `self._y[j].value`; CVXPY's `warm_start=True` then picks them up.
- A non-optimal solver status MUST raise `RuntimeError`. **Python is the correctness reference and
  should be loud.** The C++ kernel is the one that degrades gracefully
  ([09_CPP_KERNEL.md §9.5](09_CPP_KERNEL.md)); do not copy that behaviour here.
- `solve()` MUST NOT mutate `data`. The ADMM loop reuses the object.

## §5.5 `build_reference_trajectory`

```python
idx = np.clip(np.arange(start_step + 1, start_step + horizon + 1), 0, len(waypoints) - 1)
return waypoints[idx]
```

Note `start_step + 1`: the horizon covers `t = 1..T` (§5.2). Clamping past the end (hold-last) keeps
the terminal cost well defined when the horizon runs off the end of the mission.

## §5.6 Tests owned by this section

In `python/tests/test_admm_convergence.py` ([§10.1](10_TESTS.md)):

- `test_prediction_matrices_match_rollout` — `Phi x₀ + Gamma U` equals an explicit forward loop,
  to 1e-12, for random `U`
- `test_position_prediction_is_row_subset` — `Phi_p`/`Gamma_p` are exactly the position rows
- `test_input_limits_respected` — from an `x₀` aggressive enough to saturate
- a dynamics-consistency check on a single solve: `y_i^i == Phi_p x₀ + Gamma_p U` to 1e-9
- a DPP check: `problem.is_dcp(dpp=True)`

Add a timing smoke test only if you can make it non-flaky. A wall-clock assertion in the Python
suite on shared CI is noise; the real performance criterion is A8, and it lives on the C++ side.
