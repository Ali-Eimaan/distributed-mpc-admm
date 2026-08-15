# Remaining Tasks — Handoff for Claude Opus 5

This document lists everything left to finish milestone **M10** of the `distributed-mpc-admm`
project. Nothing below has been modified by the previous session except where explicitly noted
as "done". Read this top-to-bottom before touching files.

Goal of M10: remove every `TODO(deepseek …)` marker outside `.deepseek/` and make the repo
self-consistent (README, docs, media, notebooks all executed with correct outputs).

---

## 0. Environment (verified, do not re-verify unless something breaks)

- Virtualenv: `/home/eiman/Documents/tools/xenvs/mpc-admm/bin/python` (Python 3.14.4)
- Jupyter: `/home/eiman/Documents/tools/xenvs/mpc-admm/bin/jupyter`
- Key versions: cvxpy 1.9.2, osqp 1.1.3, numpy 2.5.2, scipy 1.18.0, networkx 3.6.1,
  matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1, nbmake 1.5.5
- Hardware: Intel Core i7-7600U @ 2.80GHz, `nproc` = 4
- `NB_CI=1` gates sweep sizes in notebooks; full protocols run without it.
- LaTeX (`latexmk`/`pdflatex`) is **NOT installed**. The `.tex` docs are written but uncompiled.
  Do not attempt to compile them unless the user asks.

Repo root: `/home/eiman/Documents/ephd/ros-ws/src/distributed-mpc-admm`

---

## 1. CRITICAL — Re-execute notebook 05 and verify the correct numbers

**The single most important remaining issue.** Notebook `python/notebooks/05_convergence_analysis.ipynb`
was patched (offsets passed to the solver **constructor**, see §1.1) but its saved outputs are
**stale** — they still show the *uncoupled* problem's numbers. This invalidates the README
results table, the `analysis/iterations_to_consensus.ipynb` `.npz` input, and `FIX_REPORT.md`.

### 1.1 The bug (do not reintroduce it)

`python/distributed_mpc_admm/per_agent_solver.py` — `CvxpyAgentSolver.__init__` compiles the
formation cost from the **constructor** parameter `self._offsets`:

```python
for j, offset in self._offsets.items():
    objective += weights.w_formation * cp.sum_squares(y_self - self._y[j] - d_full)
```

Passing offsets only to `.solve()` (as `data.offsets`) silently drops the formation coupling.
The fix is: pass `offsets={...}` **to the constructor** of the agent solver, never only at
solve time. Notebook 05 and two `analysis/` notebooks have already been patched this way —
verify the patch is still present and do not revert it.

### 1.2 The correct numbers (verified by direct reproduction)

Measured: `cycle(4)`, seed 0, `rho=1.0`, `TOL=1e-6`:

- WITHOUT constructor offsets: **62 iterations** (uncoupled — this is the WRONG stale output)
- WITH constructor offsets: **733 iterations, converged=True** (correct rendezvous)

Therefore the stale cells in notebook 05 are wrong:

| Cell text | Stale value | Expected (correct) |
|---|---|---|
| `iterations = 62` | 62 | ~733 (re-run to confirm exact) |
| `empirical linear rate = -0.1991` | -0.1991 | changes — recompute |
| `beta = 0.282` | 0.282 | changes — recompute |
| `empirical optimum rho = 0.01 (58.5 iterations)` | 0.01 / 58.5 | changes — recompute |

### 1.3 How to re-execute

```bash
cd /home/eiman/Documents/ephd/ros-ws/src/distributed-mpc-admm
timeout 3600 /home/eiman/Documents/tools/xenvs/mpc-admm/bin/jupyter nbconvert \
  --to notebook --execute --inplace python/notebooks/05_convergence_analysis.ipynb
```

Then verify the output actually changed (do NOT trust exit code 0 — the previous session got
exit 0 while outputs stayed stale):

```bash
/home/eiman/Documents/tools/xenvs/mpc-admm/bin/python - <<'PY'
import json
nb = json.load(open('python/notebooks/05_convergence_analysis.ipynb'))
for c in nb['cells']:
    if 'iterations =' in ''.join(c['source']):
        for o in c.get('outputs', []):
            if 'text' in o:
                print(''.join(o['text']))
PY
```

If it still shows `iterations = 62`, stop and investigate rather than proceeding — the
remaining tasks below depend on this being correct.

---

## 2. Re-execute `analysis/iterations_to_consensus.ipynb`

This notebook consumes the `.npz` written by notebook 05 (now stale). After §1 is verified,
re-run:

```bash
cd /home/eiman/Documents/ephd/ros-ws/src/distributed-mpc-admm
timeout 3600 /home/eiman/Documents/tools/xenvs/mpc-admm/bin/jupyter nbconvert \
  --to notebook --execute --inplace analysis/iterations_to_consensus.ipynb
```

---

## 3. Write the root `README.md` (12 TODO markers)

File: `README.md` at repo root. Fill every `TODO(deepseek §13.5)` comment with real content.
The markers are at these locations (line numbers from the current file):

1. **line 5 — badges**: add CI/version/licence badges. Do NOT add a badge before the
   corresponding job passes. If unsure of exact badge URLs, prefer omitting vs. fabricating.
2. **line 9 — hero GIF**: `media/4_agent_formation.gif`. If this asset does not exist, generate
   it (see §5) or, if generation is out of scope, leave a clear pointer rather than a broken link.
3. **line 14 — "What this is"**: three sentences (problem → method → what's here).
4. **line 25 — results table**: columns `configuration | ADMM iterations to 1e-4 | closed-loop
   settling (s) | final formation error (cm) | per-step wall time (ms)`, rows for 4/8 agents
   across cycle/complete/path. **Use the corrected notebook-05 numbers** (733, not 62).
   Quote hardware + solver versions underneath.
5. **line 31 — `media/convergence_curves.png`**: embed it (generate if missing, §5).
6. **line 35 — switching topology**: `media/topology_switch.gif` + 2 sentences on the split/merge
   event, + pointer to `docs/derivations/convergence_proof.tex` §7 (why the standard guarantee
   does not cover switching).
7. **line 61 — 10-line runnable snippet**: build a 4-agent square formation and plot it; copy from
   notebook 02 and actually execute it before committing.
8. **line 72 — ROS 2 prereqs**: ROS 2 Lyrical Luth + OSQP + Eigen, and how to get `osqp_vendor`
   if `rosdep` cannot resolve it.
9. **line 77 — "Method"**: x-update / z-update / dual-update in 3 lines of display math, one
   sentence naming the only quantity crossing the network (a `(T x 2)` trajectory block per
   neighbor per iteration), links to `docs/README_math.md` + derivations.
10. **line 84 — "What this does not do"**: explicit limitations (no obstacle/collision avoidance,
    no terminal set/cost → no recursive-feasibility/stability guarantee, 2nd-order integrator only,
    switching schedule published centrally in the ROS demo while control is distributed).
11. **line 96 — citation hook**: one paragraph tying the ADMM kernel to `transition-viable-swarm`,
    naming the gap from `convergence_proof.tex` §7 (fixed graph + synchronous updates) that
    motivates the AHTD object in the thesis proposal.
12. **line 104 — references**: Boyd et al. 2011, Stellato et al. 2020 (OSQP), plus the
    switching-topology consensus reference. Full citations.

### 3.1 Correct reference numbers (from prior validated work — use these in the README table)

Benchmark medians over 3 seeds, `TOL=1e-4` (ADMM rho=20, DD step=2):

| Config | ADMM iters | ADMM ms | DD iters | DD ms |
|---|---|---|---|---|
| 4 agents cycle | 25 | 689 | 223 | 4470 |
| 4 agents complete | 29 | 812 | 295 | 5630 |
| 8 agents cycle | 25 | 1095 | 221 | 8048 |
| 8 agents path | 23 | 948 | 216 | 6958 |
| 4 agents r_input=0 | 23 | 1097 | 223 | 5635 |

Tuning sweep (cycle4, seed 0, max_iter=500):
- ADMM rho: 0.5→500✗, 1.0→455, 2.0→229, 5.0→93, 10.0→47, 20.0→23
- DD step: 0.2→500✗, 0.5→500✗, 0.7→500✗, 1.0→449, 1.5→298, 2.0→223

Closed-loop / formation numbers from already-correct notebooks:
- `02_4_agent_consensus.ipynb`: open-loop converged=True in 724 iters (rho=1.0, TOL=1e-6);
  closed-loop `mean_admm_iterations=2.38, settling_step=17.0`.
- `03_formation_control.ipynb`: square `final_formation_error=0.00696, settling_step=19.0`;
  8-agent `final_formation_error=0.01387, settling_step=24.0`; cycle lambda_2=2.0 settling=19,
  complete lambda_2=4.0 settling=20.
- `04_switching_topology.ipynb`: switch steps (25, 50); split/merge (20, 40), residual at
  merge = 0.0258 m; switching peak 3.295 m.

---

## 4. Write `FIX_REPORT.md` at repo root

Currently the template still says "nothing implemented". Write a real report, sections **M1–M10,
newest first** (M10 at top). Cover each milestone's fix/implementation, with the M10 section
summarizing the formation-offsets bug and its consequences (62 vs 733 iterations).

---

## 5. Media assets (`media/`)

Confirm these exist and are non-placeholder; generate any that are missing using the corresponding
notebooks/scripts:

- `media/4_agent_formation.gif` (README hero)
- `media/convergence_curves.png` (README results)
- `media/topology_switch.gif` (README switching-topology)
- `media/README.md` — already complete (TODO removed); do not regress it.

Generation is best done by executing the notebook cells that produce these figures (notebook 02/04
and `analysis/` notebooks already have plotting code). Do not hand-fabricate plots.

---

## 6. Final verification (definition of done)

Run and expect **zero** results:

```bash
cd /home/eiman/Documents/ephd/ros-ws/src/distributed-mpc-admm
grep -rn "TODO(deepseek" --exclude-dir=.git --exclude-dir=.deepseek .
```

Also confirm the following files have no remaining `TODO(deepseek …)`:
- root `README.md`
- `docs/README_math.md`, `docs/derivations/{augmented_lagrangian,consensus_admm_derivation,convergence_proof,preamble}.tex`, `docs/derivations/references.bib`
- `docs/COMPARISON_VS_DUAL_DECOMP.md`
- `media/README.md`
- all `python/notebooks/01..05`
- all `analysis/*.ipynb`

If time permits, re-run all notebooks full (without `NB_CI=1`) so committed outputs are real.

---

## 7. Notes on files already done (do not redo)

- `python/distributed_mpc_admm/dual_decomposition.py` — new, validated. `DualDecompositionOptions`,
  `DualDecompositionAgentSolver`, `DualDecomposition`, `DualDecompositionResult`.
- `python/distributed_mpc_admm/__init__.py` — exports `DualDecomposition*` in `__all__`.
- `python/scripts/run_comparison.py` — rewritten; `_make_solvers` passes offsets to constructor.
- `docs/COMPARISON_VS_DUAL_DECOMP.md` — §5 filled with measured numbers (complete).
- `analysis/communication_load_study.ipynb` — patched + re-executed (221714 bytes).
- `analysis/topology_robustness.ipynb` — patched; **re-execution was launched but unconfirmed** —
  verify its outputs show correct (coupled) results before considering it done.
- `python/notebooks/02/03/04` — already correct (offsets passed to constructor).

### ADMM/dual-decomposition math reference (for any doc text you write)

- ADMM scaled-dual form (Boyd 2011 §7.2): x-update (parallel QP), z-update (neighborhood
  averaging with `y + lam`), dual `λ += y_hat − z`. Residuals: primal
  `r_k = sqrt(Σ||y_i^j − z^j||²)`, dual `s_k = ρ·sqrt(Σ||z^j − z^j_{k-1}||²)`.
  Stopping: `eps_pri = sqrt(n_dual)·eps_abs + eps_rel·max(||y||,||z||)`;
  `eps_dual = sqrt(n_dual)·eps_abs + eps_rel·ρ·||lam||`.
- Dual decomposition: x-update with linear dual term `ν·y`, z-update as plain average,
  dual `ν += step·(y−z)`. Residuals have **no ρ multiplier**; tolerances use ν norm, **no ρ**.
- `AgentCostWeights` defaults: `q_position=1.0, q_velocity=0.1, r_input=0.05, r_rate=0.0,
  p_terminal=5.0, w_formation=10.0`.
