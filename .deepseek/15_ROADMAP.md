# §15 · Implementation order · §17 · Definition of done · §18 · Release

---

## §15 · Milestones

Each milestone is independently verifiable. **Do not start one before the previous one passes.**

| # | Milestone | Files | Spec | Done when |
| --- | --- | --- | --- | --- |
| M1 | Graph layer | `communication_graph.py` | [04](04_GRAPH.md) | `λ₂` matches the analytic values for `path(4)`, `cycle(4)`, `complete(4)`; channel round-trip at zero loss |
| M2 | Model and prediction | `per_agent_solver.py` §5.1–§5.2 | [05](05_LOCAL_QP.md) | `Phi x₀ + Gamma U` matches an explicit rollout to 1e-12 |
| M3 | Formation geometry | `formation_constraints.py` | [07](07_FORMATION.md) | geometry and rigidity tests green |
| M4 | Local QP | `CvxpyAgentSolver` | [05 §5.3–§5.5](05_LOCAL_QP.md) | a single-agent solve is dynamically consistent and limit-respecting; `is_dcp(dpp=True)` |
| M5 | ADMM core | `ConsensusADMM` | [06](06_ADMM.md) | **A2, A3** |
| M6 | Closed loop and figures | `DistributedMPC`, `SimulationLog`, `plotting.py` | [08](08_CLOSED_LOOP.md) | **A4, A5, A6** |
| M7 | C++ kernel | `per_agent_qp.*`, `admm_kernel.*` | [09](09_CPP_KERNEL.md) | **A7, A8** |
| M8 | Node and launch | `consensus_node.*`, `launch/` | [11](11_NODE.md) | `4_agent_admm.launch.py` drives four processes to a formation with no arguments |
| M9 | CI | `.github/workflows/test.yml` | [14](14_CI.md) | **A1**, and A7 verified *by CI* |
| M10 | Notebooks, analysis, docs, media | `notebooks/`, `analysis/`, `docs/`, `media/`, root `README.md` | [12](12_ANALYSIS.md), [13](13_DOCS.md) | **A9**; notebooks execute; README numbers reproducible |

### Why this order

**M1–M6 need no ROS and no C++ at all.** The Python package installs and tests with `pip` on any
machine, including the Windows host ([02_ENVIRONMENT.md §2.3](02_ENVIRONMENT.md)). That is not an
accident — it is what makes the first six milestones debuggable in a unit test instead of inside a
running ROS graph. Getting the block ordering wrong is a five-minute fix at M5 and a two-hour one at
M8.

**M4 before M5.** An ADMM loop wrapped around a local QP that has never solved anything correctly
gives you two unvalidated layers and no way to tell which is wrong. Make the single-agent solve
provably right first; then the ADMM loop has a known-good component.

**M5 before M6.** A closed loop around a wrong open-loop solve produces trajectories that look
plausible and are not. A2 is the gate here for exactly that reason: it is the only check in the
repository that compares against something *outside* the ADMM implementation.

**M5 before M7** is the non-obvious one. The temptation is to start the C++ early, because it is the
interesting engineering. Resist it: the parity test (A7) is meaningless until the Python side is
known correct, and discovering a block-ordering bug after the C++ is built on top means unwinding
both. A fast kernel computing the wrong answer is worse than no kernel.

**M7 before M8**, more strongly. A four-process demo of a wrong controller is worse than no demo,
because it is persuasive.

**M9 after M8** rather than early: CI that runs before there is anything to run produces noise, and
noisy CI gets ignored.

### Parallelism

M9 and M10 are independent and can follow M8 in either order. Everything else is a strict chain.

Within M10, the notebooks ([12](12_ANALYSIS.md)) and the derivations
([13 §13.2](13_DOCS.md)) can proceed in parallel, but the derivations' worked numbers must be
cross-checked against the notebooks' simulations before either is called done. The `ρ` optimum from
the rate bound and the empirical optimum from [§12.5](12_ANALYSIS.md) are the specific pair to
check.

### The three milestones people underestimate

**M4.** The DPP formulation in [05 §5.4](05_LOCAL_QP.md) is fiddly and its failure mode is a silent
100× slowdown, not an error. Budget time to get `is_dcp(dpp=True)` actually true rather than
assumed.

**M7.** Getting OSQP's fixed-sparsity value-only updates right — the `rho_diag_indices` list, the
upper-triangular CSC Hessian, the API names for your installed version — is a week of careful work,
not an afternoon. Three of the version risks (V9, V10, V11) live here.

**M10.** Filling `COMPARISON_VS_DUAL_DECOMP.md` honestly means implementing a second algorithm and
running it under the same protocol. That is the milestone that makes the repository worth having; it
is also the one most likely to be rushed, because the code already works by then.

---

## §17 · Definition of done

### Per file

- every `TODO(deepseek …)` implemented and the marker **deleted**, or converted into a specific
  written issue with a reason
- Python: `ruff` and `black --check` clean, `mypy --disallow-untyped-defs` clean, no
  `raise NotImplementedError` remaining
- C++: builds with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` clean, no
  `throw std::logic_error("not implemented")` remaining
- its tests pass, in a **Release** build for anything C++
- carries the SPDX header ([§2.6](02_ENVIRONMENT.md)) — notebooks excepted
- any behaviour a reader would not predict from the signature is documented in a comment

### Per repository

- all nine acceptance criteria in [01_OVERVIEW.md §1.3](01_OVERVIEW.md) hold
- the README's numbers are reproducible from `analysis/` or the test suite, on the hardware named
  next to them
- the README's limitations section is complete and specific ([13_DOCS.md §13.6](13_DOCS.md))
- no `UNVERIFIED` marker remains in [02_ENVIRONMENT.md §2.1](02_ENVIRONMENT.md), `CMakeLists.txt` or
  the workflow without an accompanying note explaining why it could not be resolved — **V10 and V11
  (the OSQP API and Hessian format) must be resolved, not annotated**
- `COMPARISON_VS_DUAL_DECOMP.md`'s results table has measured numbers, and its section 6 is
  populated
- every citation in `references.bib` checked against the actual publication
- A7 is verified **by CI**, not merely by a local run ([14_CI.md §14.5](14_CI.md))
- every notebook executes top to bottom on a clean kernel and is committed with outputs

### Progress check

```bash
grep -rn "TODO(deepseek" --exclude-dir=.git --exclude-dir=.deepseek . | wc -l
```

```bash
grep -rn "UNVERIFIED" --exclude-dir=.git --exclude-dir=.deepseek .
```

Both should trend to zero. **The second reaching zero matters more than the first** — an
unimplemented function is visible to everyone, an unverified assumption is visible to nobody until
it costs a day. Here the sharpest examples are V10 and V11: an OSQP API call that silently does
nothing, or a Hessian handed over in the wrong triangular format, produces a solver that runs, reports
success, and optimises a problem you did not pose.

---

## §18 · Release criteria

The project is pre-release at `0.1.0` (`cpp_admm/package.xml`, `python/pyproject.toml`). Do **not**
tag `1.0.0` until the gate below holds. Keeping `0.1.0` while anything is open is the honest number.

**Gate for `1.0.0`:** all nine acceptance criteria green, in CI, on the hardware named in the README.

### Release checklist (run before every tag)

1. A1–A9 green in CI on the README's named hardware.
2. All three media files exist and are reproducible from the notebooks by one command
   ([12_ANALYSIS.md §12.9](12_ANALYSIS.md)).
3. `LICENSE`, `package.xml` and `pyproject.toml` all say BSD-3-Clause
   ([§2.6](02_ENVIRONMENT.md)); every source file carries the SPDX header.
4. `package.xml` and `pyproject.toml` versions match.
5. Every version pin in [02_ENVIRONMENT.md](02_ENVIRONMENT.md) reflects what was actually built, not
   what was assumed — **including the OSQP tag**.
6. The derivations compile and their worked numbers match the code
   ([13_DOCS.md §13.2](13_DOCS.md)).
7. `COMPARISON_VS_DUAL_DECOMP.md` has no empty table cells and its "when dual decomposition wins"
   section is populated.
8. The README quick-start runs verbatim from a clean clone.
9. The parity fixture in the repository was generated by the committed notebook, not by hand
   ([10_TESTS.md §10.5](10_TESTS.md)).

If a criterion cannot be met, change it in [01_OVERVIEW.md §1.3](01_OVERVIEW.md) with a written
reason — do not weaken a test in place.
