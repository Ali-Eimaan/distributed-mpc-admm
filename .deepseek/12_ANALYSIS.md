# §12 · Notebooks, analysis, media

**Governs:** `python/notebooks/01`–`05`, `analysis/*.ipynb`, `media/`
**Milestone:** M10
**Done when:** **A9**; every notebook executes in CI; `media/` regenerates from committed code.

Each notebook skeleton already has markdown cells stating what each section must show and a code
cell carrying the `TODO(deepseek §12.x)`.

**Rules for all of them:**

- Every notebook MUST run top to bottom on a clean kernel, in CI, with no manual steps.
- Commit them **with outputs**. Notebook outputs are the demo for a reader who will not clone the
  repository.
- Seed everything ([§16.4](16_CONVENTIONS.md)). A number in a notebook that cannot be reproduced
  from a seed is not a result.
- If 04 and 05 exceed the CI runner budget, gate the sweep sizes on an `NB_CI` environment variable.
  **Do not drop them from CI** — an unexecuted notebook is a broken notebook nobody knows about.

---

## §12.1 `01_admm_intuition.ipynb`

Scalar and two-agent ADMM, written **inline** — do not import `ConsensusADMM`. The point is that the
reader sees all six lines and can predict what `ρ` does before running anything.

Sections: the splitting idea on `f(x)=½(x−3)²`, `g(z)=|z|`; two-agent scalar consensus; a `ρ` sweep
over three decades showing primal falling fast at large `ρ` and dual lagging (and the reverse); the
`α` step; then the jump to trajectories, stated symbolically with block shapes annotated and **no new
numerics**.

## §12.2 `02_4_agent_consensus.ipynb`

Four agents, cycle graph, rendezvous. The credibility notebook.

Must contain, in this order: setup and initial configuration; one open-loop ADMM solve with the
residual plot; **the centralised cross-check** ([06_ADMM.md §6.7](06_ADMM.md)) printing
`max|u_admm − u_central|`; the closed loop; and a breakdown of where the time goes — mean local-QP
time per agent against the z-update time.

That last cell is the argument for why the C++ kernel targets the QP rather than the transport, and
it is the number [09_CPP_KERNEL.md §9.6](09_CPP_KERNEL.md) asks the kernel to reproduce.

**This notebook also exports `cpp_admm/test/data/four_agent_reference.json`**
([§10.5](10_TESTS.md)). Add a cell that writes it, and note in the markdown that regenerating it is
a deliberate act.

## §12.3 `03_formation_control.ipynb`

Rigid square; the same square on a cycle versus a complete comm graph with `λ₂` printed for each;
leader-follower on a circle or lemniscate with the steady-state lag quantified; the same on a path
graph showing `is_infinitesimally_rigid` returning `False` and the corresponding shear; a morph from
line to V; and eight agents in a two-row grid.

Produces **`media/4_agent_formation.gif`**.

## §12.4 `04_switching_topology.ipynb`

Where the textbook guarantee stops applying. Every experiment is chosen because it violates one
specific assumption, and the job is to **measure**, not to claim it still works
([01_OVERVIEW.md §1.6](01_OVERVIEW.md)).

| Experiment | Violates | Reports |
| --- | --- | --- |
| baseline, fixed cycle | — | the reference everything else is measured against |
| scheduled cycle ↔ path | fixed graph | transient size at each switch |
| dwell-time sweep (2, 5, 10, 20, 40) | fixed graph | the threshold below which transients stop decaying |
| **split and merge** | fixed graph, connectivity | drift while split; residual formation error at the merge instant |
| packet loss (0 … 0.4, 20 seeds) | synchrony | median + IQR of final formation error |
| bounded delay (0, 1, 2, 5) | synchrony | the **residual floor** stale duals produce |

Produces **`media/topology_switch.gif`**, with the split and merge instants annotated *on the frames*
([§12.9](12_ANALYSIS.md)).

**The final markdown cell is the most valuable text in the repository.** A table mapping each
experiment to the assumption it breaks in `convergence_proof.tex` §7, and what was observed. That
cell is the bridge to the thesis proposal; write it last, from the measurements, not from
expectation.

## §12.5 `05_convergence_analysis.ipynb`

Quantitative, with error bars. **State the protocol in the first cell and do not deviate** — 30 seeds
per configuration, fixed `ρ` (adaptation off, [06_ADMM.md §6.5](06_ADMM.md)),
`eps_abs = eps_rel = 1e-6`. Half the value of this notebook is that the numbers across its cells are
comparable.

Sections: empirical linear rate fitted to `log r_k` and compared against the theoretical bound;
`ρ` sweep U-curve with the empirical optimum located; iterations against `λ₂` across
complete/star/cycle/path/random with a fitted power law (**A5**); size sweep `N = 2..8` with
iterations and wall time on **separate panels** — they scale differently and conflating them hides
the story; horizon sweep `T = 5..40` (iterations nearly flat, per-iteration cost not); adaptive `ρ`
against best-fixed *and* median-fixed `ρ`, reporting both comparisons honestly.

Produces **`media/convergence_curves.png`**.

## §12.6 `analysis/iterations_to_consensus.ipynb`

Consumes saved `SimulationLog` `.npz` files under `analysis/data/` rather than re-simulating, so the
figures regenerate in seconds.

Iterations to `1e-2` / `1e-4` / `1e-6` per topology; cold versus warm start per control step; and a
real-time budget overlay — horizontal lines at 10/20/50 Hz on the wall-time plot, so the achievable
rate can be read off.

The headline: moderate accuracy is cheap and high accuracy is not, and in MPC a loose solve is
refined by the next control step anyway. That asymmetry is the standard argument for ADMM here;
state it with the measurement rather than as received wisdom.

## §12.7 `analysis/communication_load_study.ipynb`

**First cell: check the analytic model against reality.** `communication_load(...)`
([04_GRAPH.md §4.6](04_GRAPH.md)) versus `LossyChannel.stats` from a real run at `loss_prob = 0`,
asserted. If they disagree, the model is wrong, not the measurement.

Then: scaling with `N` across topologies, annotating `O(N²)` against `O(N)`; payload compression
(float32, and every-other-horizon-step with interpolation) plotted as accuracy cost against bytes;
and a bandwidth envelope with realistic mesh-radio budgets overlaid, giving the largest feasible
`(N, T, iteration budget)` triple.

## §12.8 `analysis/topology_robustness.ipynb`

Random edge failure (200 trials per `q`); the same outcomes plotted against the **realised `λ₂`**
rather than against `q`; adversarial min-cut removal for a worst case; intermittent connectivity
built with `TimeVaryingGraph.is_jointly_connected` ([04_GRAPH.md §4.4](04_GRAPH.md)); and recovery
time after reconnection against how long the graph was split.

> The `λ₂` plot is a real test of the repository's story, not a decoration. If the collapse is **not**
> sharper in that coordinate than against `q`, the connectivity narrative is weaker than claimed —
> and that must be reported, not quietly replaced with the `q` plot.

## §12.9 `media/`

Everything here is generated, never hand-edited:

```bash
python -m distributed_mpc_admm.plotting
```

| File | From |
| --- | --- |
| `4_agent_formation.gif` | §12.3 |
| `topology_switch.gif` | §12.4 |
| `convergence_curves.png` | §12.5 |

Constraints ([08_CLOSED_LOOP.md §8.6](08_CLOSED_LOOP.md)): under ~5 MB per GIF, `apply_style("readme")`
so they are legible in dark mode, and the switching GIF must make the **event** legible — annotate
the split and merge instants on the frames, not only in the caption. A reader watching a
topology-switch GIF who cannot see when the topology switched has learned nothing.
