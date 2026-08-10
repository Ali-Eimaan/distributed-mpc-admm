# §6 · The consensus ADMM iteration

**Governs:** `python/distributed_mpc_admm/consensus_admm.py` — `ADMMOptions`, `ADMMHistory`,
`ADMMResult`, `ConsensusADMM`
**Milestone:** M5
**Done when:** **A2** and **A3**.

This is the centre of the repository. Everything before it is inputs and everything after it is
presentation.

---

## §6.1 The iteration

General-form consensus ADMM in scaled dual form (Boyd et al. 2011, §7.2 and §3.4.1). For ADMM
iteration `k`:

```
1.  x-update      (one QP per agent, fully parallel, no communication)
      (U_i, y_i) ← argmin  f_i(U_i, y_i) + (ρ/2) Σ_{j∈N̄_i} ‖ y_i^j − z^j_k + λ_i^j_k ‖_F²

2.  relaxation    (α ∈ [1, 2); α = 1 disables it)
      ŷ_i^j ← α·y_i^j + (1−α)·z^j_k

3.  z-update      (computed at agent j, from its contributors)
      z^j_{k+1} ← (1/|C_j|) Σ_{i∈C_j} ( ŷ_i^j + λ_i^j_k )

4.  broadcast     agent j sends z^j to its neighbours

5.  dual update
      λ_i^j_{k+1} ← λ_i^j_k + ŷ_i^j − z^j_{k+1}

6.  residuals, tolerances, optional ρ update
```

**Order matters.** Do not move the relaxation after the z-update, and do not use the *new* `z` in
step 5 for an agent that did not receive it.

Step 3 is the only place the method is distributed: `C_j = contributors(j) = N̄_j`, so the average
needs one round of neighbour messages and nothing global. If you ever find yourself computing `z`
from a full stack of all agents' variables, you have written the centralised algorithm.

## §6.2 `ConsensusADMM` state

```python
self._graph: CommunicationGraph
self._solvers: dict[int, PerAgentSolver]
self._T, self._dim: int
self._opts: ADMMOptions
self._channel: LossyChannel | None
self._y:   dict[int, dict[int, np.ndarray]]   # (T, dim) blocks
self._lam: dict[int, dict[int, np.ndarray]]
self._z:   dict[int, np.ndarray]
self._z_last_known: dict[int, dict[int, np.ndarray]]   # per-agent cache under loss
```

Validate in `__init__` that `solvers` covers `range(graph.n_agents)` exactly, and that each solver
was constructed with the neighbourhood `graph` implies. Zero-initialise every block.

`set_graph` MUST raise when the supplied solvers no longer match the new neighbourhoods. The QP
variable count depends on `|N̄_i|`; solving the old problem against the new graph is silent and
wrong.

**The class simulates a distributed algorithm in one process.** No update may read data a real agent
could not have. Global quantities — residual norms, the objective sum, the consensus gap — are
computed for logging only and MUST never feed back into an update (A3, rule 1 in
[00_RULES.md](00_RULES.md)).

## §6.3 The z-update

```python
z_new = {}
for j in range(N):
    contributions = [y_hat[i][j] + lam[i][j] for i in graph.contributors(j) if received(i, j)]
    z_new[j] = np.mean(contributions, axis=0)
```

**The divisor is the number of received contributions, not `|C_j|`.**

A missing contribution is **excluded** from the average; it is never replaced by a stale value.
Averaging a stale term with fresh ones biases `z` in a way that is far harder to characterise than
a term that is simply absent — and characterising the degradation is the point of
[§12.4](12_ANALYSIS.md). This is the opposite of the broadcast rule in §6.6; keep the two straight.

If *no* contribution arrives for subject `j` in an iteration, hold `z^j` at its previous value and
count it in `history.messages_dropped`. Do not divide by zero and do not silently zero the
trajectory.

### An invariant worth asserting

Summing the dual update over `i ∈ C_j` gives, in the synchronous lossless case,

```
Σ_{i∈C_j} λ_i^j = 0   exactly, from iteration 1 onward
```

and therefore `z^j` reduces to `mean_i ŷ_i^j`. Add this as a debug assertion behind
`options.verbose`. If it fails, the dual update and the z-update disagree about which set they range
over, and no amount of tuning will fix it. It does **not** hold under packet loss — that is expected
and is itself worth measuring.

## §6.4 Residuals, tolerances, stopping

```
n_dual   = Σ_i |N̄_i| · T · dim

r = sqrt( Σ_i Σ_{j∈N̄_i} ‖ y_i^j − z^j ‖_F² )                    (primal)
s = ρ · sqrt( Σ_i Σ_{j∈N̄_i} ‖ z^j − z^j_prev ‖_F² )              (dual)

eps_pri  = sqrt(n_dual)·eps_abs + eps_rel · max( ‖y‖_F , ‖z_stacked‖_F )
eps_dual = sqrt(n_dual)·eps_abs + eps_rel · ρ · ‖λ‖_F
```

`z_stacked` stacks `z^j` **once per `(i, j)` pair**, not once per `j` — it has to live in the same
space as `y` for the comparison to mean anything.

Use the **un-relaxed** `y` for `r`.

Stop when `r ≤ eps_pri` **and** `s ≤ eps_dual`. Hitting `max_iterations` sets `converged = False`
and returns the last iterate; it does not raise ([00_RULES.md](00_RULES.md), rule "never report an
unconverged solve as converged").

Also log `consensus_gap = max over (i,j) of max|y_i^j − z^j|`. The Frobenius residual can look small
while one agent is badly off, and this is the number that exposes it. Diagnostics only — it must
never influence the stopping test.

## §6.5 Adaptive `rho`

```python
if primal > mu * dual:      factor = tau
elif dual > mu * primal:    factor = 1.0 / tau
else:                       factor = 1.0

new_rho = float(np.clip(rho * factor, rho_min, rho_max))
if new_rho != rho:
    actual = new_rho / rho
    for i in lam:
        for j in lam[i]:
            lam[i][j] /= actual          # scaled duals absorb rho
```

**The rescaling is mandatory.** `lam` stores `u/ρ`; changing `ρ` without dividing `lam` by the same
factor silently changes the dual iterate. The symptom is a run that converges for fixed `ρ` and
stalls with adaptation on, and it presents as "adaptive rho does not help", which is a plausible
enough conclusion that nobody investigates. `test_converges_for_range_of_rho`
([§10.1](10_TESTS.md)) catches it.

Note the clip: when `rho * factor` lands outside `[rho_min, rho_max]`, `actual` is the **realised**
ratio, not `factor`. Computing `actual` from the clipped value is what keeps the rescaling exact at
the bounds.

**Default `adaptive_rho = False`.** Turn it off before quoting any convergence-rate number — the
linear-rate result assumes fixed `ρ`, and a rate fitted across a `ρ` change is meaningless
([§12.5](12_ANALYSIS.md)).

## §6.6 Asynchrony and loss

When a `LossyChannel` is supplied:

- **Local copies (step 3 input):** a missing contribution is excluded, divisor shrinks (§6.3).
- **Consensus broadcast (step 4 output):** a missing `z^j` **does** fall back to
  `_z_last_known[i][j]`, because agent `i` must put *something* into its next QP. Record the
  staleness.

With no channel, `_broadcast_consensus` returns `{i: z for all i}`. Sharing the array reference is
fine; downstream code MUST NOT mutate it.

Under loss the iteration is no longer textbook ADMM and the convergence guarantee does not apply.
Say that in the docstring, in `docs/README_math.md`, and in the notebook — not "it still converges"
([01_OVERVIEW.md §1.6](01_OVERVIEW.md)).

## §6.7 The centralised reference

`solve_centralized` lives in `python/tests/test_admm_convergence.py` and is what makes A2 evidence
rather than self-consistency.

Build **one monolithic CVXPY problem** over all agents: per-agent tracking, velocity and effort
costs, the formation edge costs at `2·w_formation` per edge ([§5.3](05_LOCAL_QP.md)), per-agent
dynamics and box constraints, coupling imposed **directly** on `(p_i − p_j)`. No local copies, no
duals, no ADMM.

Compare **inputs**, not just objective values. Equal cost with different inputs means the problem is
not strictly convex and the weights need fixing — with `r_input > 0` and `q_velocity > 0` it is,
and that is why those defaults are nonzero.

## §6.8 Tests owned by this section

[§10.1](10_TESTS.md) specifies them; the ones that gate this milestone:

| Test | Gates |
| --- | --- |
| `test_matches_centralized_solution` | **A2** — the one that must never be skipped or loosened |
| `test_no_global_information_used` | **A3** |
| `test_single_agent_admm_equals_local_qp` | the degenerate case; residuals at solver noise after one iteration |
| `test_residuals_decrease` | see [§10.3](10_TESTS.md) — window minima, **not** per-iteration monotonicity |
| `test_converges_for_range_of_rho` | the dual-rescaling bug in §6.5 |
| `test_over_relaxation_preserves_optimum` | `α` changes the path, not the fixed point |
| `test_solution_invariant_to_agent_relabelling` | no accidental dependence on iteration order in the z-update |
| `test_disconnected_graph_does_not_reach_consensus` | terminates cleanly, no NaN, `converged = False` |
| `test_max_iterations_reported_not_raised` | degrades, does not crash |
