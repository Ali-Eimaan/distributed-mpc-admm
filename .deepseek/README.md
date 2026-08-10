# `.deepseek/` — implementation specification

**You are the implementing model.** This directory is your complete instruction set for turning
the `distributed-mpc-admm` skeleton into working code. Everything you need is here; nothing
outside this directory instructs you.

**Scope:** every file in the repository containing a `TODO(deepseek …)` marker.
**Authority:** where a document here and a code comment disagree, the document wins — say so in
the commit message rather than silently diverging.

> **Provenance.** This directory was adapted from the `mpc-cbf` specification of the same author,
> which in turn came from `cbf-safety-filter`. The working rules, review protocol and toolchain
> notes carry over; every subsystem document was rewritten for this repository. If you find a
> sentence that still talks about barrier functions, acados, RPI sets or tube tightening, it is a
> leftover — report it.

---

## Read in this order

| Read first | Document | Covers |
| --- | --- | --- |
| 1 | [00_RULES.md](00_RULES.md) | How to work. Non-negotiable. Read before touching a file. |
| 2 | [01_OVERVIEW.md](01_OVERVIEW.md) | What is being built, why it exists, acceptance criteria A1–A9 |
| 3 | [02_ENVIRONMENT.md](02_ENVIRONMENT.md) | Ubuntu 26.04 / ROS 2 Lyrical Luth / CVXPY / OSQP, pinned versions, **version risk register** |
| 4 | [15_ROADMAP.md](15_ROADMAP.md) | Milestones M1–M10 in dependency order, definition of done |
| 5 | [16_CONVENTIONS.md](16_CONVENTIONS.md) | **Layout and sign conventions**, units, numerical policy, traps. Re-read when confused. |

§16.1 is the single most important page in this directory. Nothing in this repository crashes when
it is wrong. A formation weight counted twice, a dual left unscaled after a `rho` change, a stale
trajectory averaged into the consensus step — each produces a solver that converges cleanly,
reports success, and returns the optimum of a problem you did not pose. Read it before you write
your first update equation, not after your first inexplicable plot.

Then work through the milestones, opening the subsystem document for each:

| Document | Implements | Milestone |
| --- | --- | --- |
| [03_BUILD_SYSTEM.md](03_BUILD_SYSTEM.md) | `python/pyproject.toml`, `cpp_admm/CMakeLists.txt`, `package.xml` | throughout |
| [04_GRAPH.md](04_GRAPH.md) | `communication_graph.py` — topologies, switching, the lossy channel | M1 |
| [05_LOCAL_QP.md](05_LOCAL_QP.md) | `per_agent_solver.py` — model, prediction matrices, the local QP | M2, M4 |
| [06_ADMM.md](06_ADMM.md) | `consensus_admm.py` — `ConsensusADMM` | M5 |
| [07_FORMATION.md](07_FORMATION.md) | `formation_constraints.py` — geometry, rigidity, leader-follower | M3 |
| [08_CLOSED_LOOP.md](08_CLOSED_LOOP.md) | `DistributedMPC`, `SimulationLog`, `plotting.py` | M6 |
| [09_CPP_KERNEL.md](09_CPP_KERNEL.md) | `admm_kernel.*`, `per_agent_qp.*`, the transports | M7 |
| [10_TESTS.md](10_TESTS.md) | `python/tests/`, `cpp_admm/test/` | with each milestone |
| [11_NODE.md](11_NODE.md) | `consensus_node.*`, `launch/` | M8 |
| [14_CI.md](14_CI.md) | `.github/workflows/test.yml` | M9 |
| [12_ANALYSIS.md](12_ANALYSIS.md) | `python/notebooks/`, `analysis/`, `media/` | M10 |
| [13_DOCS.md](13_DOCS.md) | `docs/` derivations, `README_math.md`, root `README.md` | M10 |

[FILE_MAP.md](FILE_MAP.md) — every skeleton file in the repository mapped to the document that
specifies it. Use it when you have a file and need its spec.

[REVIEW.md](REVIEW.md) and [FIX_REPORT.md](FIX_REPORT.md) are the review protocol and the
reporting template. Read REVIEW.md before you declare a milestone done.

[INFO.md](INFO.md) is the author's original portfolio specification for this repository —
**read-only**. It fixes the target file structure; [01_OVERVIEW.md §1.5](01_OVERVIEW.md) records
every file the skeleton adds beyond it, and why.

---

## Section numbering

Each document owns a section number that matches its filename prefix: `04_GRAPH.md` contains §4,
`09_CPP_KERNEL.md` contains §9, and so on. A cross-reference written `§6.3` therefore always means
"section 6.3, which lives in `06_ADMM.md`". This holds everywhere, including in the
`TODO(deepseek §6.3)` comments inside the skeleton source files.

`15_ROADMAP.md` carries §15 (implementation order), §17 (definition of done) and §18 (release
gate).

## Conventions used in these documents

- **MUST** — required for correctness. Deviating is a bug.
- **SHOULD** — strong default. Deviate only with a stated reason in a code comment.
- **UNVERIFIED** — a value or assumption that has not been confirmed. Verify before relying on
  it, and update the document with what you found.

## Before you start

Run this to see the work remaining:

```bash
grep -rn "TODO(deepseek" --exclude-dir=.git --exclude-dir=.deepseek .
```

Every one of those markers is specified somewhere in this directory. If you find one that is
not, that is a gap in the spec — report it rather than guessing.

## What the repository is not

This is a **distributed optimisation demonstrator**: N double integrators, coupled only through
formation costs, each solving its own MPC, agreeing through consensus ADMM. It is not a planner,
not a perception stack, and not a collision-avoidance system. If you find yourself writing obstacle
handling, barrier constraints or a global path planner, you have left the scope — that work is the
author's `mpc-cbf`, which is the per-agent *safety* problem, and `transition-viable-swarm`, which
composes the two.

Three things this repository is *also* not, which matter more than they sound:

- **It is not a proof of recursive feasibility or closed-loop stability.** There is no terminal
  set and no terminal cost. Per-step feasibility comes from the box constraints; stability is
  demonstrated numerically, not guaranteed. See [01_OVERVIEW.md §1.6](01_OVERVIEW.md). Any
  comment, docstring, notebook or derivation claiming otherwise is a bug.
- **It is not a claim that consensus ADMM converges under switching topologies or packet loss.**
  It is the opposite: the repository *measures* where the standard guarantee stops applying. The
  wording matters, and [13_DOCS.md §13.2](13_DOCS.md) is unusually strict about it.
- **It is not a re-implementation of a QP solver.** CVXPY/OSQP in Python and OSQP in C++ are
  dependencies. Your job is the splitting, the consensus iteration, the diagnostics and the
  evidence — not the QP.
