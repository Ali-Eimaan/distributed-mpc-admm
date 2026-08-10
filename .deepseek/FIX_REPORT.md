# FIX_REPORT.md — milestone completion report

**Status: nothing implemented.** The repository is a skeleton; no milestone has been completed. This
file is the template. Fill one section per milestone completed, newest at the top, and keep the old
ones — the history of what was fixed and how it was verified is the record that makes the
repository's claims checkable.

Write this at the end of every milestone in [15_ROADMAP.md](15_ROADMAP.md), before starting the next
one.

---

## How to fill it in

Three rules, all from [00_RULES.md](00_RULES.md):

1. **Report the real result.** If a test fails, show the output. If you skipped something, say which
   and why. A green table over a broken build is the most expensive thing you can produce.
2. **Every number is measured.** Solve times come from a Release build, with the CPU named. Iteration
   counts come from a stated seed and a stated tolerance. A number you did not measure does not go
   in the table.
3. **Name what is still open.** A milestone can be complete with known open issues; it cannot be
   complete with hidden ones.

---

## Template — copy this block per milestone

```markdown
## M<n> · <milestone name>

**Date:** YYYY-MM-DD
**Commit:** <sha or "working tree">
**Spec:** <the .deepseek document(s) implemented>
**Build:** <Python version> · Release · <compiler and version> · <CPU, if any timing is reported>
**Solvers:** CVXPY <version>, OSQP <version / tag — required from M4 onward>

### Test results

| # | Test | Type | Time | Result |
|---|------|------|------|--------|
| 1 | test_prediction_matrices_match_rollout | pytest | — | PASS / FAIL / SKIP |

Total: <n> passed, <n> failed, <n> skipped.

### Acceptance criteria touched

| Criterion | Before | After | Evidence |
|---|---|---|---|
| A2 | not met | met | `test_matches_centralized_solution`, max residual = <value> over <n> configurations |

### What was implemented

<Per file: what the body now does, and anything a reader would not predict from the signature.>

### Deviations from the spec

<Every place the implementation differs from the .deepseek document, and why. If a document was
wrong, say which section and confirm you fixed it in the same commit (rule: "When a document is
wrong", §0).>

### UNVERIFIED resolved

| Risk | Was | Is |
|---|---|---|
| V10 | OSQP C API names assumed | read from <header>, names are <...> |

### Still open

<Known issues, with severity and where they are logged in REVIEW.md.>

### Numbers for downstream documents

<Anything the README, COMPARISON_VS_DUAL_DECOMP.md or a notebook will quote, so it is recorded once
and cited rather than re-measured inconsistently.>
```

---

## Milestone-specific notes

Things that must appear in particular reports, because they are easy to complete without.

| Milestone | Must record |
| --- | --- |
| M1 | the measured `λ₂` for `path(4)`, `cycle(4)`, `complete(4)` against the analytic values — the baseline every later connectivity claim rests on |
| M2 | the prediction-vs-rollout residual, so later integration questions have a baseline |
| M3 | which built-in formations are infinitesimally rigid, with the rank and the tolerance used |
| M4 | **whether `is_dcp(dpp=True)` is actually True**, the CVXPY version, and the per-solve wall time — the DPP question is invisible otherwise (risk V8) |
| M5 | **the max input residual against the centralised solve** (A2) for every tested configuration, and the iteration counts per topology |
| M6 | the settling step and final edge-RMS (A4); the cold-vs-warm iteration ratio (A6); the fitted exponent of iterations against `λ₂` (A5) |
| M7 | **the OSQP version and API variant** (risk V10); the parity residual (A7); the p95 per-step time **with the CPU named** (A8); the allocation count from `NoHeapAllocationDuringIterate` |
| M8 | whether `4_agent_admm.launch.py` ran with no arguments from a clean clone, and in how many **separate processes** |
| M9 | whether A7 is verified *by CI* or only locally, and the total workflow wall time |
| M10 | the dual-decomposition baseline's numbers under the same protocol; which experiments in notebook 04 broke which assumption; and the seeds used |

## What a milestone report must never do

- Quote a timing from a Debug build or from CI ([14_CI.md §14.5](14_CI.md))
- Report a criterion as met when the test that verifies it was skipped
- Record a tolerance that was moved without saying so and why
  ([16_CONVENTIONS.md §16.8](16_CONVENTIONS.md))
- Claim A7 on a regenerated fixture without saying the fixture was regenerated and why
  ([10_TESTS.md §10.5](10_TESTS.md))
