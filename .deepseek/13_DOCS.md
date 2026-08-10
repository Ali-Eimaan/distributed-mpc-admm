# §13 · Documentation

**Governs:** `docs/README_math.md`, `docs/derivations/*.tex`, `docs/derivations/preamble.tex`,
`docs/derivations/references.bib`, `docs/COMPARISON_VS_DUAL_DECOMP.md`, root `README.md`
**Milestone:** M10
**Done when:** all three derivations compile in CI, the comparison table has measured numbers, and
the README's numbers are reproducible.

For a portfolio repository the documentation is not overhead — it is a substantial fraction of what
is being evaluated. An advisor who reads one file in this repository reads
`consensus_admm_derivation.tex`.

---

## §13.1 `docs/README_math.md`

The **single source of truth for notation**. If a symbol here disagrees with the code, the code is
wrong.

The notation table is a contract: **no symbol may appear in a `.tex` file without an entry here
first.** That rule is what stops three documents drifting into three notations, which is the default
outcome otherwise.

Sections to fill, all outlined in the file: notation; the agent model with the ZOH discretisation and
the `t = 1..T` convention **stated explicitly** (§16.1 — this is the most common source of
off-by-one bugs in the repository, so write it down here and cite this section from the code);
the coupled MPC problem with its size as a function of `(N, T)`; why it splits; the consensus ADMM
iteration; residuals and tolerances with the `n_dual` count spelled out; adaptive `ρ` with the
mandatory rescaling; convergence with its assumptions; a tuning guide; and implementation notes on
the shared block layout.

The convergence section MUST list the assumptions and then state plainly which ones the
switching-topology and packet-loss experiments break. **This repository demonstrates a gap; it does
not close it** ([01_OVERVIEW.md §1.6](01_OVERVIEW.md)).

## §13.2 The derivations

Three documents plus a shared preamble and bibliography. All five carry `TODO(deepseek §13.2)` or
`§13.4`.

**`augmented_lagrangian.tex`** — the object the other two build on. Ordinary Lagrangian, dual ascent
and its two failure modes; the augmented Lagrangian and why the quadratic term buys a differentiable
dual with a `1/ρ`-Lipschitz gradient (a full paragraph — this is *why the method works* and it is
usually reduced to a citation); the method of multipliers, including the derivation that the dual
step size *is* `ρ`; why it does not split; and the scaled-dual substitution, flagging that the code
stores the scaled dual so a `ρ` change must rescale `λ`.

**`consensus_admm_derivation.tex`** — the centrepiece. The coupled problem; the observation that the
only inseparable terms are the edge costs; local copies and the equivalence lemma (one paragraph —
it is immediate, but it is the step that licenses everything else); the three updates derived by
holding the others fixed, with the z-update derived to its closed form and the explicit sentence
that the average ranges only over `N̄_j`; the scaled-form algorithm block, kept
**verbatim-comparable** with the code; over-relaxation; residuals; cost accounting; the receding
horizon; and a worked two-agent, `T = 2` example with every matrix written out numerically and three
iterations done by hand.

That worked example is tedious to write and disproportionately convincing to read. Do not skip it.

**`convergence_proof.tex`** — assumptions A1–A5 numbered and referred to by number; the Lyapunov
function motivated before it is used; monotone decrease derived; residual and objective convergence
with the `O(1/k)` ergodic rate; the linear rate under strong convexity carrying the `λ₂` dependence
through; specialisation to this problem with the rate bound **evaluated numerically for the 4-agent
square** and compared against the empirical optimum from [§12.5](12_ANALYSIS.md).

Then §7, which is the reason the document exists:

- **A3 (fixed graph).** When `N̄_i` changes, the decision variable changes *dimension*. The Lyapunov
  function is defined on a space that no longer exists, so monotone decrease does not weaken — it
  becomes ill-typed. A merge event needs a set-valued reset map. State precisely that; it is the
  AHTD object in the thesis proposal.
- **A4 (synchrony).** Under loss, agents update on different `z`, so there is no single `z^k` for the
  Lyapunov function to reference. Cite the partially-asynchronous ADMM results that do exist, state
  their bounded-delay assumption, and note that they do not cover a simultaneously changing graph.
- **A5 (exact solves).** Inexact ADMM has summable-error results; a fixed iteration budget does not
  satisfy them. Reference the measured residual floor from [§12.4](12_ANALYSIS.md).

Close with what the experiments show — bounded degradation for moderate loss and dwell times,
structured failure at disconnection — and **label it a measurement-supported conjecture, not a
theorem**. A reviewer who catches one overclaim discounts every other claim in the repository.

Finish with three or four sharply stated open questions. Those are what an advisor actually reads
this document for.

**`preamble.tex`** — one macro per symbol in the §13.1 notation table. Macros live here only; a macro
redefined in an individual document is exactly how the three drift apart.

## §13.3 `docs/COMPARISON_VS_DUAL_DECOMP.md`

**This document must contain measured numbers from this repository, not textbook claims.** A
comparison with hand-waved numbers is worse than none.

Implement the dual-decomposition baseline against the **same** problem instances used in
[§12.5](12_ANALYSIS.md) — same seeds, same tolerances, same hardware — reusing `PerAgentSolver` with
`ρ = 0` and a dual-ascent outer loop. Anything less makes the comparison unfalsifiable.

Measure: iterations to `1e-4` across topologies and `N`; sensitivity to the tuning parameter (`ρ`
versus step size) with **both U-curves on one axis** — the width of the good region is the real
result; behaviour when `f_i` loses strict convexity (`q_velocity = r_input = 0`); wall time per
iteration (dual decomposition's iterations are cheaper — account for that honestly rather than
comparing iteration counts alone); and behaviour under packet loss.

**Section 6, "when dual decomposition still wins", is not optional.** It has cheaper iterations, no
`ρ` to tune when the Lipschitz constant is genuinely known, a smaller memory footprint, and it
parallelises across constraints rather than agents. Name the regimes where those matter. A comparison
concluding that the method the author implemented wins in every case reads as advocacy and will be
discounted by exactly the readers this repository is aimed at.

## §13.4 `references.bib`

**Verify every entry against the actual publication.** Do not copy a citation you have not opened,
and do not fill a page range from memory (rule 4, [00_RULES.md](00_RULES.md)). A wrong page range in
a repository whose purpose is to demonstrate scholarly care is expensive.

Beyond the two entries already present (Boyd et al. 2011; Stellato et al. 2020), the bibliography
needs: a distributed-MPC survey for framing; the linear-rate result for ADMM under strong convexity;
asynchronous / partially-asynchronous ADMM with its bounded-delay assumption; consensus over
switching topologies (joint connectivity, dwell time); graph rigidity theory for
[07_FORMATION.md §7.3](07_FORMATION.md); and a hybrid-systems reference for the reset-map discussion
in §7 of the convergence proof.

## §13.5 Root `README.md`

Order matters. A reviewer gives this thirty seconds before deciding whether to keep reading.

1. **Hero GIF first** (`media/4_agent_formation.gif`). The first thing a reader sees must be the
   system working, not a paragraph.
2. Three sentences: the problem, the method, what is in the repository.
3. The **results table**, filled from [§12.5](12_ANALYSIS.md) and `analysis/`. Columns:
   configuration, ADMM iterations to `1e-4`, closed-loop settling (s), final formation error (cm),
   per-step wall time (ms). Rows for 4 and 8 agents across cycle/complete/path. **Every row names its
   hardware, solver versions and git SHA underneath.** A timing without them is not a claim.
4. The switching GIF plus two sentences on the split/merge event, pointing at
   `convergence_proof.tex` §7.
5. Repository layout, quick start (a snippet that runs **verbatim from a clean clone** — copy it out
   of notebook 02 and actually execute it), and the method in three lines of display math naming the
   only quantity that crosses the network: one `(T × 2)` block per neighbour per iteration.

Badges only once the corresponding job is green. A red badge on the first screen is worse than no
badge.

## §13.6 The limitations section

**Above the fold, not in a footnote.** It lists everything in
[01_OVERVIEW.md §1.6](01_OVERVIEW.md): no recursive feasibility or stability guarantee; no
collision avoidance; double-integrator dynamics only; the convergence guarantee holds only for a
fixed connected graph with synchronous updates and exact solves; the disconnected case fails and is
meant to; and the ROS demo's switching schedule is published centrally while the control is
distributed.

A reader who finds one of these themselves discounts everything else in the repository. A reader who
is told them up front trusts the rest. This section is a credibility asset, not a disclaimer.

## §13.7 The citation hook

One paragraph, factual. The C++ kernel is the distributed solver inside the author's
`transition-viable-swarm`. The gap identified in `convergence_proof.tex` §7 — that the guarantee
assumes a fixed graph and synchronous updates — is what motivates the AHTD object in the thesis
proposal.

Keep it honest: **this repository demonstrates the gap and measures its consequences; it does not
close it.**
