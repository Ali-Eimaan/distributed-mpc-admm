# §0 · Working rules

**Read this before touching any file.** These rules are ordered by priority; when two conflict,
the lower number wins.

---

1. **One file per change.** Implement a file completely, run its tests, then move on. Do not open
   six files and leave all of them half-finished.

2. **Do not change public signatures.** The function names, argument names, argument order and
   return types in `python/distributed_mpc_admm/`, the QP block layout in
   [09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md), and the node parameter keys in
   [11_NODE.md §11.2](11_NODE.md) are the contract between subsystems. The tests, the notebooks,
   the C++ kernel and the parity check are all written against them. If a signature is genuinely
   wrong, **stop and say so** instead of quietly editing it.

3. **Delete the `TODO(deepseek …)` comment when you implement it.** A leftover TODO on implemented
   code is a lie that costs the next reader ten minutes. Keep the surrounding explanatory comment —
   it encodes the mathematics and is not the TODO.

4. **Never invent a number.** Not a measured solve time, not an iteration count presented as
   typical, not a figure from a paper you did not open, not a citation field. If you do not have a
   source, leave the placeholder and mark it `UNVERIFIED`. This repository's value is that a
   reviewer can trust its numbers; one invented figure costs more credibility than the missing
   result would have.

5. **Every claim needs a check.** If you implement something whose correctness is not obvious —
   a z-update divisor, an adaptive-`rho` rescaling, a Hessian cross term, a warm-start shift —
   add the test that proves it **in the same change**. See [10_TESTS.md](10_TESTS.md).

6. **Silent wrongness is the failure mode of this repository.** Nothing here crashes when it is
   wrong. A double-counted formation weight, an unscaled dual, a stale trajectory in the average:
   each produces a run that converges, reports success, and solves the wrong problem.
   [16_CONVENTIONS.md §16.1](16_CONVENTIONS.md) fixes every convention once. Follow it exactly,
   and test the assembled quantities directly — not just the end-to-end trajectory.

7. **Build order matters.** Follow the milestones in [15_ROADMAP.md](15_ROADMAP.md). The
   dependency chain is real: the ADMM loop cannot be debugged before the local QP is known good,
   and the C++ kernel cannot be validated before the Python reference exists.

8. **If a step is blocked** — a missing dependency, an unresolved version question, a decision
   these documents do not settle — implement everything that is *not* blocked, then report exactly
   what is blocked and why. Do not stub around it silently and do not guess.

9. **Report honestly.** If a test fails, say so and show the output. If you skipped something, say
   that. A green summary over a broken build is the most expensive thing you can produce here,
   because it will be believed.

---

## What "implemented" means

A file is not done when it imports or compiles. It is done when:

- every `TODO(deepseek …)` in it is implemented and the marker deleted, or converted into a
  specific written issue with a reason
- Python: `ruff`, `black --check` and `mypy --disallow-untyped-defs` are clean; no
  `raise NotImplementedError` remains
- C++: builds with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` clean (the warnings are on
  deliberately — fix them, do not silence them); no `throw std::logic_error("not implemented")`
  remains
- its tests pass, in a **Release** build for anything C++
- it carries the SPDX header ([02_ENVIRONMENT.md §2.6](02_ENVIRONMENT.md))
- any behaviour a reader would not predict from the signature is documented in a comment

The full definition is §17 in [15_ROADMAP.md](15_ROADMAP.md).

## Four rules specific to a distributed optimiser

**Never let global information into an agent update.** This is the claim the whole repository
makes. Residual norms, objective sums, iteration counts and progress output are *logging*. If any
of them ever feeds back into an `x`-update, a `z`-update or a dual update, the repository is
simulating a centralised algorithm and describing it as a distributed one. `test_no_global_information_used`
([10_TESTS.md §10.1](10_TESTS.md)) enforces this; do not weaken it, and do not add a convenience
argument that makes it easy to violate.

**Never weaken a convergence tolerance to make a test pass.** If `test_matches_centralized_solution`
fails at `1e-3`, the answer is never `1e-2`. Either the splitting is wrong, or the two objectives
differ — and the most likely cause is the formation-weight double-count in
[05_LOCAL_QP.md §5.3](05_LOCAL_QP.md), not the tolerance.

**Never report an unconverged solve as converged.** `ADMMResult.converged` reflects the actual
residual test. Hitting the iteration cap is a normal, expected outcome in closed loop and the
correct response is to return the last iterate with `converged = False` — an MPC loop must degrade,
not crash. What must never happen is the flag saying otherwise.

**Never let the Python and C++ implementations drift.** The update equations, the iteration order,
the residual definitions and the QP block layout exist in exactly two places, and
`PythonParityTest` ([10_TESTS.md §10.4](10_TESTS.md)) pins them to 1e-8. When they diverge, every
notebook number and every README claim becomes wrong by an amount nobody can bound, and nothing
crashes.

## When a document is wrong

These documents were written before the code existed. Some of it will turn out to be wrong —
particularly the version pins in [02_ENVIRONMENT.md](02_ENVIRONMENT.md), the CVXPY DPP guidance in
[05_LOCAL_QP.md §5.4](05_LOCAL_QP.md), and the OSQP C API names in
[09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md), which change between releases.

When you find an error: **fix the document in the same commit as the code.** A specification that
has silently drifted from the implementation is worse than no specification, because the next
reader will trust it.

Do not weaken an acceptance criterion or a test threshold in place. If a criterion cannot be met,
change it explicitly in [01_OVERVIEW.md §1.3](01_OVERVIEW.md) with a written reason.
