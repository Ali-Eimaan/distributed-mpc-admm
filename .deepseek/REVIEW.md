# REVIEW.md — review protocol and findings log

**Status: no review has been run.** The repository is a skeleton; nothing has been implemented,
built, or tested. This file is the protocol for the first review sweep and the log it writes into.

Run a review sweep at the end of every milestone in [15_ROADMAP.md](15_ROADMAP.md), and a full sweep
before any release tag (§18).

---

## How to read an issue id

`R<round>-<number>` — e.g. `R1-8` is the eighth issue of round one. **Ids are permanent.** When a
later round revisits an issue it gets a new id that references the old one. Do not renumber.

## Agent assignment

| Agent | Use for |
| --- | --- |
| `deepseek-v4-flash` | Mechanical, well-specified fixes: a known factor-of-two, a renamed constant, a missing include, a doc edit. The fix is stated exactly and the verification is a single test. |
| `deepseek-v4-pro` | Anything needing derivation, design judgement, or a change that ripples across files: re-deriving the DPP formulation, restructuring the block layout, resolving a spec ambiguity. |

**Rule for both:** every fix lands with the test that proves it, in the same change (rule 5 in
[00_RULES.md](00_RULES.md)). If a fix cannot be verified by a test, say so in the commit message and
explain how it was verified instead.

## Severity

| Severity | Means | Examples in this repository |
| --- | --- | --- |
| **Critical** | Produces a converged, confidently wrong answer, or breaks the build | formation edge double-counted; missing Hessian cross terms; `λ` not rescaled on a `ρ` change; a stale value substituted into the z-average; global information reaching an agent update |
| **Major** | Wrong results, or violates a spec decision | flatten-order mismatch; horizon off-by-one; block ordering divergent between C++ and Python; duals zeroed on warm start; graph caches not invalidated; a test asserting per-iteration residual monotonicity |
| **Minor** | Robustness, clarity, hygiene | missing throttle on a loop warning; an unnamed agent in an error message; a stale comment; an unseeded RNG in a non-statistical test |

A "converges cleanly and returns the wrong optimum" bug is always **Critical**, however small the
diff. That is the class this repository exists to get right
([16_CONVENTIONS.md](16_CONVENTIONS.md)).

---

## §R.1 What to check, per sweep

Work down this list. It is ordered by cost-to-discover-later, not by where the code lives.

### Conventions — always first

- [ ] Every row of [16_CONVENTIONS.md §16.1](16_CONVENTIONS.md) checked against the **assembled
      quantity** — the actual `q` vector, the actual `P` block, the actual `z` divisor — not against
      trajectory behaviour
- [ ] Flattening is time-major C order in **both** implementations, and the CVXPY variables are flat
- [ ] `X` covers `t = 1..T`; `x₀` is not a row of `X`
- [ ] `closed_neighborhood` is sorted with self in sorted position, in Python **and** C++
- [ ] The centralised reference uses `2·w_formation` per edge
- [ ] Hessian cross terms `−2·w_formation·I` present in the C++ `P`
- [ ] `λ` rescaled by the **realised** (post-clip) ratio whenever `ρ` changes
- [ ] z-update divisor is the number of received contributions
- [ ] Missing local copy → excluded; missing `z` → cached fallback. Both directions, not one rule

### The four failure modes named in §0

- [ ] No global quantity feeds any update — residuals, objective and gap are logging only (A3)
- [ ] No path reports `converged = True` on an unconverged iterate
- [ ] No claim of recursive feasibility, stability, or convergence-under-switching in any comment,
      docstring, notebook or derivation ([01_OVERVIEW.md §1.6](01_OVERVIEW.md))
- [ ] Python and C++ agree, and `PythonParityTest` exists and runs

### The local QP

- [ ] The CVXPY problem is built **once**; `solve()` only assigns parameters
- [ ] `is_dcp(dpp=True)` asserted in `__init__` and raising on failure (risk V8)
- [ ] Only `y_i^i` carries a dynamics constraint
- [ ] A missing `z` or `lam` key raises rather than defaulting to zero
- [ ] A non-optimal solver status raises in Python (the reference should be loud)
- [ ] `solve()` does not mutate its `LocalProblemData`

### OSQP integration

- [ ] **The OSQP API names have been read out of the installed headers** and the verified names are
      in a comment (risks V10, V11)
- [ ] `P` is upper-triangular CSC, and a test asserts it
- [ ] `rho_diag_indices` verified by `UpdateRhoTouchesOnlyDiagonalEntries`
- [ ] `updateRho` skipped when `ρ` is unchanged
- [ ] The `2w` factor applied consistently across every `P` block
- [ ] Block offsets always go through the accessors, never recomputed inline

### Graphs, switching, loss

- [ ] `add_edge` / `remove_edge` invalidate both caches, and a test proves it
- [ ] `λ₂` clipped at zero from below
- [ ] `is_connected` defined through `λ₂`, not a second BFS
- [ ] `random_connected` resamples rather than patching
- [ ] `LossyChannel.advance` keeps the largest `admm_iteration`, not the last arrival
- [ ] `set_graph` does not clear mailboxes
- [ ] Zero received contributions holds `z` rather than dividing by zero

### Real-time and numerical

- [ ] No heap allocation in `AdmmKernel::iterate()`, and `NoHeapAllocationDuringIterate` proves it
- [ ] Tolerances match the budget in [16_CONVENTIONS.md §16.4](16_CONVENTIONS.md) — none tightened to
      "make it pass", none loosened to make a red test green
- [ ] `consensus_gap` influences diagnostics only
- [ ] Every stochastic path takes an explicit seeded `Generator`
- [ ] Timing numbers come from a Release build with the CPU named
- [ ] Nothing unthrottled in the ROS control loop

### Honesty

- [ ] No number in the README, docs, notebooks or comments that is not reproducible from a committed
      cell or test
- [ ] No citation field invented; every entry checked against the publication
      ([13_DOCS.md §13.4](13_DOCS.md))
- [ ] Every `UNVERIFIED` either resolved or still labelled
- [ ] `COMPARISON_VS_DUAL_DECOMP.md` has measured numbers, and its "when dual decomposition wins"
      section is populated rather than empty
- [ ] The limitations section lists everything in [01_OVERVIEW.md §1.6](01_OVERVIEW.md)
- [ ] The **disconnected-graph failure is reported as a failure**, with its mechanism — not smoothed
      into a footnote (A9)
- [ ] No acceptance criterion weakened in place
- [ ] A7 verified by CI, or the README does not imply it is ([14_CI.md §14.5](14_CI.md))
- [ ] The parity fixture was generated by the committed notebook, not adjusted by hand

### Hygiene

- [ ] No `TODO(deepseek …)` left on implemented code
- [ ] No `raise NotImplementedError` or `throw std::logic_error("not implemented")` remaining
- [ ] Every source file carries the SPDX header ([02_ENVIRONMENT.md §2.6](02_ENVIRONMENT.md))
- [ ] Every new C++ test registered in `CMakeLists.txt`
      ([03_BUILD_SYSTEM.md §3.7](03_BUILD_SYSTEM.md))
- [ ] `-Wconversion` clean, not silenced
- [ ] `ruff`, `black --check`, `mypy` clean

---

## §R.2 Status summary

Update this table at the end of each sweep.

| Severity | Open | Fixed this round |
| --- | --- | --- |
| Critical | — | — |
| Major | — | — |
| Minor | — | — |

*(No sweep has been run. Replace the dashes with counts, and add the findings below, when R1
happens.)*

---

## §R.3 Findings

Use one subsection per issue, in this shape:

```markdown
### R1-1 · [Critical] Centralised reference omits the factor of two on formation edges

**File:** `python/tests/test_admm_convergence.py:88`
**Spec:** [05_LOCAL_QP.md §5.3](05_LOCAL_QP.md)
**Agent:** deepseek-v4-flash

**What is wrong:** <one sentence>
**How it fails:** <the concrete configuration where the two objectives differ, and by how much>
**Fix:** <the exact change>
**Verified by:** <the test, by name, added in the same change>
**Status:** open | fixed in <commit>
```

The **How it fails** line is not optional. A finding without a concrete failure scenario is an
opinion, and opinions do not get fixed in the right order.

---

## §R.4 Reviewing without a build

If the repository has never been run at the time of a sweep — which will be true for R1 — say so at
the top of the round and be explicit that findings come from reading the code and re-deriving the
mathematics, not from a failing test. Then:

- Do **not** report performance findings. You cannot see them.
- Do **not** report "this would not compile" unless you are certain; report it as a question.
- Do **not** report a CVXPY or OSQP API mismatch as a defect without having checked the installed
  version — the documents themselves flag those names as UNVERIFIED
  ([02_ENVIRONMENT.md §2.1](02_ENVIRONMENT.md)).
- **Do** report every convention, formula, shape and layout finding. Those are exactly what a
  reading review catches and a passing test suite can miss — and in this repository they are also
  the ones that produce a confident wrong answer rather than a crash.

A review that overstates its evidence is worse than no review, for the same reason a green summary
over a broken build is (rule 9, [00_RULES.md](00_RULES.md)).
