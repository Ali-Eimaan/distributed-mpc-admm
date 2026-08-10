# §10 · Tests

**Governs:** `python/tests/test_admm_convergence.py`, `python/tests/test_formation_consensus.py`,
`cpp_admm/test/test_admm_kernel.cpp`, `cpp_admm/test/data/four_agent_reference.json`
**Milestone:** with each milestone, not after them.

Both Python test files are fully specified by the docstrings already in them. Implement exactly what
each docstring describes. This document adds the things a docstring cannot carry: what **not** to
assert, and why.

---

## §10.1 `test_admm_convergence.py`

Tests the optimiser. Property-based rather than golden-number, so it survives re-tuning.

| Test | Gates | Note |
| --- | --- | --- |
| `test_prediction_matrices_match_rollout` | M2 | write this first; it fails before `Gamma` exists |
| `test_position_prediction_is_row_subset` | M2 | |
| `test_single_agent_admm_equals_local_qp` | M5 | the degenerate case |
| `test_matches_centralized_solution` | **A2** | the gate that matters |
| `test_residuals_decrease` | M5 | see §10.3 |
| `test_converges_on_all_topologies` | **A5** | assert the *ordering*, not absolute counts |
| `test_disconnected_graph_does_not_reach_consensus` | M5 | terminates cleanly, no NaN |
| `test_converges_for_range_of_rho` | M5 | catches the missing dual rescaling |
| `test_over_relaxation_preserves_optimum` | M5 | |
| `test_adaptive_rho_reaches_same_optimum` | M5 | |
| `test_warm_start_reduces_iterations` | **A6** | perturb `x₀` between solves |
| `test_solution_invariant_to_agent_relabelling` | M5 | |
| `test_input_limits_respected` | M4 | from a saturating `x₀` |
| `test_max_iterations_reported_not_raised` | M5 | degrades, does not crash |
| `test_packet_loss_degrades_gracefully` | **A9** | see §10.3 |
| `test_scales_to_eight_agents` | M5, `@slow` | |
| `test_no_global_information_used` | **A3** | see below |

### `test_matches_centralized_solution` — the one that must never be skipped

Settings: `eps_abs = eps_rel = 1e-6`, `max_iterations = 2000`, `adaptive_rho = False`.
Tolerance `1e-3` on the stacked input vector.

Compare **inputs**, not objective values ([06_ADMM.md §6.7](06_ADMM.md)).

If it fails by a suspiciously clean factor, check the formation double-count in
[05_LOCAL_QP.md §5.3](05_LOCAL_QP.md) before touching anything else.

### `test_no_global_information_used`

Wrap each solver's `solve` with `monkeypatch` and assert the `LocalProblemData` it receives has `z`
and `lam` keys **only** within that agent's closed neighbourhood.

Cheap to write, and it is the property the entire repository claims. Do not weaken it and do not add
a convenience argument to `LocalProblemData` that would make it easy to violate.

## §10.2 `test_formation_consensus.py`

Tests that the optimiser is pointed at the right problem: geometry consistency
([07_FORMATION.md §7.7](07_FORMATION.md)), graph properties
([04_GRAPH.md §4.7](04_GRAPH.md)), and closed-loop behaviour
([08_CLOSED_LOOP.md §8.7](08_CLOSED_LOOP.md)).

The two closed-loop tests that carry the most weight:

- `test_agents_reach_formation` (**A4**) — assert on the *held* error over the last 20 % of the run.
- `test_switch_to_disconnected_graph_drifts` — the **expected-failure documentation test**. It pins
  the behaviour the thesis exists to explain. Assert that the two components hold their own shapes
  (each component's internal edge-RMS stays small) **and** that the inter-component distance grows.
  A test that only asserts "it fails" would pass for the wrong reasons.

## §10.3 What not to assert

Three tests will tempt you into an assertion that is wrong. Getting these right is most of the value
of this document.

**`test_residuals_decrease` — do NOT assert per-iteration monotonicity.** With over-relaxation or
adaptive `ρ`, the primal residual legitimately rises on individual steps. Assert that the **minimum
over successive 10-iteration windows is non-increasing**, and that the final value is below
tolerance. A strict-monotonicity assertion here forces `alpha = 1`, which then hides an
over-relaxation bug.

**`test_packet_loss_degrades_gracefully` — do NOT assert convergence.** The entire point of that
experiment is that the guarantee is lost ([01_OVERVIEW.md §1.6](01_OVERVIEW.md)). Assert:
boundedness, no NaN, and monotone degradation of the final formation error with `loss_prob` across
seeds. Asserting convergence under loss would either fail honestly or force you to weaken the loss
until it proves nothing.

**`test_formation_holds_while_tracking` — do NOT assert zero centroid lag.** This is a
finite-horizon tracker with no integral action. Bounded lag is the property it has; zero lag is a
property it does not have, and asserting it would eventually be "fixed" by adding an integrator
nobody asked for.

## §10.4 `test_admm_kernel.cpp`

`makeConfig(agent_id)` builds the 4-agent cycle, rendezvous, `horizon = 10`, `dt = 0.1`
configuration. A helper wires four kernels through `InProcessTransport` with no loss and returns
owning handles.

| Group | Tests | Gates |
| --- | --- | --- |
| Dynamics | matrices vs closed form; prediction vs rollout; position rows are a subset | M7 |
| QP | setup + PSD + upper-triangular `P`; block layout via the accessors; dynamics constraint satisfied; input limits; `updateConsensus` changes only `q`; **`updateRho` touches only `rho_diag_indices`**; warm start reduces inner iterations | M7 |
| Kernel | `configure` rejects invalid configs; single agent converges immediately; four agents reach consensus; **`NoHeapAllocationDuringIterate`**; `shiftWarmStart` preserves the tail for `lam` too; `setNeighbors` preserves surviving blocks; stale data triggers early exit; packet loss produces no NaN | M7, **A8** |
| Parity | `MatchesReferenceIterates`, `MatchesReferenceOptimum` | **A7** |

`NoHeapAllocationDuringIterate`: override global `operator new` with a counting hook, run
`configure()`, reset the counter, run 10 `iterate()` calls, assert the count is zero. Without this
test the real-time claim in the header is a comment.

`UpdateRhoTouchesOnlyDiagonalEntries`: snapshot `P.valuePtr()`, call `updateRho(2*rho)`, assert only
`rho_diag_indices` differ and by exactly the expected delta.

The A8 timing measurement is a **separate local Release run**, not a gtest assertion. Record the p95
with the CPU named in `FIX_REPORT.md` ([§14.5](14_CI.md) explains why not in CI).

## §10.5 The parity fixture

`cpp_admm/test/data/four_agent_reference.json`, exported from
`notebooks/02_4_agent_consensus.ipynb` ([§12.2](12_ANALYSIS.md)). Contents:

```json
{
  "config":  { "n_agents": 4, "edges": [[0,1],[1,2],[2,3],[3,0]], "horizon": 10, "dt": 0.1, "dim": 2 },
  "weights": { "q_position": …, "q_velocity": …, "r_input": …, "r_rate": 0.0,
               "p_terminal": …, "w_formation": … },
  "limits":  { "u_max": …, "v_max": … },
  "options": { "rho": 1.0, "alpha": 1.0, "adaptive_rho": false },
  "x0":      [[…], [], [], []],
  "offsets": { "0": {"1": [dx, dy], "3": [dx, dy]}, … },
  "iterates": [ { "k": 0, "y": …, "z": …, "lam": … }, … 20 entries ],
  "optimum": { "inputs": … }
}
```

Serialise every trajectory block already flattened in the repository's C order
([§16.1](16_CONVENTIONS.md)) so the C++ side does no reshaping and cannot introduce an ordering bug
in the test itself.

> **Regenerating this fixture to make a red test go green is not an option.** If parity breaks, one
> of the two implementations changed. Find out which, fix it, and only then regenerate — and say in
> the commit message that you did.

## §10.6 Budget and markers

Keep the full Python suite, excluding `@pytest.mark.slow`, **under 60 seconds**. If it creeps past
that, shrink horizons and agent counts — not coverage.

`@pytest.mark.slow` is for anything over ~5 s. CI runs `-m "not slow"` on pull requests and the full
suite on push ([§14.1](14_CI.md)).

Every stochastic test takes an explicit seed. A flaky test in this repository is not a nuisance; it
is indistinguishable from a real intermittent bug in a distributed algorithm, and it will be
dismissed as flakiness the one time it is real.
