# IMPLEMENTATION_GUIDE.md

**The implementation specification is this directory. Start at [README.md](README.md).**

This file used to sit at the repository root and hold the whole guide. It was folded into the
numbered documents here so there is exactly one specification rather than two that can drift apart —
a spec that has silently diverged from the code is worse than no spec, because the next reader will
trust it.

Do not re-expand this file. If you need to add specification content, add it to the document that
owns that section number.

---

## Where things went

The root guide's sections map onto this directory as follows. Nothing was dropped; several sections
grew.

| Root guide had | Now lives in |
| --- | --- |
| §0 Ground rules | [00_RULES.md](00_RULES.md) |
| §1 Build order | [15_ROADMAP.md §15](15_ROADMAP.md) |
| §2 Conventions (exports, shapes, flattening, horizon, randomness) | [16_CONVENTIONS.md §16.1–§16.4](16_CONVENTIONS.md), [03_BUILD_SYSTEM.md §3.5](03_BUILD_SYSTEM.md) |
| §3 `communication_graph.py` | [04_GRAPH.md](04_GRAPH.md) |
| §4 `per_agent_solver.py` and `formation_constraints.py` | [05_LOCAL_QP.md](05_LOCAL_QP.md), [07_FORMATION.md](07_FORMATION.md) — split, because they are different milestones |
| §5 `consensus_admm.py` | [06_ADMM.md](06_ADMM.md) (open loop), [08_CLOSED_LOOP.md](08_CLOSED_LOOP.md) (closed loop, logging, figures) |
| §6 `cpp_admm` | [09_CPP_KERNEL.md](09_CPP_KERNEL.md) (kernel and QP), [11_NODE.md](11_NODE.md) (node and launch), [03_BUILD_SYSTEM.md](03_BUILD_SYSTEM.md) (CMake, package.xml) |
| §7 Tests | [10_TESTS.md](10_TESTS.md) |
| §8 Notebooks | [12_ANALYSIS.md](12_ANALYSIS.md) |
| §9 Docs | [13_DOCS.md](13_DOCS.md) |
| §10 CI | [14_CI.md](14_CI.md) |
| §11 Media and README | [12_ANALYSIS.md §12.9](12_ANALYSIS.md), [13_DOCS.md §13.5–§13.7](13_DOCS.md) |
| §12 Pitfall index | [16_CONVENTIONS.md §16.9](16_CONVENTIONS.md) |
| §13 Definition of done | [15_ROADMAP.md §17](15_ROADMAP.md) |

What the root guide did **not** have, and this directory adds: acceptance criteria A1–A9
([01_OVERVIEW.md §1.3](01_OVERVIEW.md)), a version risk register
([02_ENVIRONMENT.md §2.1](02_ENVIRONMENT.md)), milestones with dependency reasoning
([15_ROADMAP.md §15](15_ROADMAP.md)), a release gate ([§18](15_ROADMAP.md)), a review protocol
([REVIEW.md](REVIEW.md)), a milestone reporting template ([FIX_REPORT.md](FIX_REPORT.md)), and a
file-to-spec map ([FILE_MAP.md](FILE_MAP.md)).

## Marker renumbering

The root guide's markers were written `TODO [GUIDE x.y]`. They are now `TODO(deepseek §x.y)` and
renumbered to the document that owns each section. If you are working from an older checkout, the
mapping is:

| Old | New | Old | New |
| --- | --- | --- | --- |
| `[GUIDE 2.1]` | `§3.5` | `[GUIDE 6.7]` | `§9.3` |
| `[GUIDE 3.1]` | `§4.1` | `[GUIDE 6.8]` | `§11.3` |
| `[GUIDE 3.3]` | `§4.4` | `[GUIDE 6.9]` | `§11.4` |
| `[GUIDE 3.4]` | `§4.5` | `[GUIDE 6.10]` | `§11.6` |
| `[GUIDE 4.3]` | `§5.4` | `[GUIDE 6.11]` | `§11.7` |
| `[GUIDE 5.2]` | `§6.2` | `[GUIDE 7.1]` | `§10.4` |
| `[GUIDE 5.6]` | `§8.2` | `[GUIDE 8.1–8.8]` | `§12.1–§12.8` |
| `[GUIDE 6.1]` | `§3.2` | `[GUIDE 9.1–9.7]` | `§13.1–§13.4` |
| `[GUIDE 6.2]` | `§9.7` | `[GUIDE 10.1–10.3]` | `§14.4`, `§14.3`, `§14.6` |
| `[GUIDE 6.3–6.5]` | `§9.8`, `§9.9`, `§9.5` | `[GUIDE 11.1]`, `[GUIDE 11.2]` | `§12.9`, `§13.5` |
| `[GUIDE 6.6]` | `§9.4` | | |

## Section numbering

A document's number matches its filename prefix, so `§9.3` always means "section 9.3, in
`09_CPP_KERNEL.md`". The `TODO(deepseek §9.3)` markers in the source files carry the same numbers.

The command that lists the remaining work is in [README.md](README.md) — it is kept there, and only
there, so that running it does not match the documentation describing it.
