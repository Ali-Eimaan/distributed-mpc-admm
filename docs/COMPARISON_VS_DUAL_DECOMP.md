# ADMM vs dual decomposition

This document is the measured, repo-local answer to one question: *given the same coupled
MPC problem, the same communication graph, and the same per-iteration message pattern,
what do we actually lose if we drop ADMM's augmented term and run plain dual
decomposition?* Every number in §5 comes from `python/scripts/run_comparison.py`, run
against the same instances, seeds and tolerances as `notebooks/05_convergence_analysis.ipynb`.

## 1. The question

Both methods decompose the *same* coupled problem

$$
\min_{y_i, z} \; \sum_{i=1}^{N} f_i(y_i)
\quad\text{s.t.}\quad y_i^j = z^j \quad \forall\, i,\ j \in \bar{\mathcal N}_i,
$$

across the same undirected graph, and both need exactly one round of neighbor
communication per iteration (an x-update in parallel, a neighborhood averaging, a dual
step). The *only* structural difference is a single term: ADMM augments the Lagrangian
with $\tfrac{\rho}{2}\sum_{i,j}\lVert y_i^j - z^j\rVert^2$, dual decomposition does not.

That one term is what changes the whole character of the iteration. ADMM's augmented
Lagrangian is strongly convex in every primal block, so each x-update is a well-posed,
*bounded* QP for any $f_i$ that is merely convex, and the update is a firmly-nonexpansive
step whose dual iterates converge linearly. Dual decomposition climbs the ordinary
Lagrangian with a subgradient step, so (i) each x-update needs $f_i$ to be *strictly*
convex or the inner problem is unbounded/multi-valued, and (ii) the step size must be
tuned to a Lipschitz constant of a dual function nobody can compute a priori. These two
facts — not the asymptotic iteration-complexity bounds — are what dominate the measured
behaviour in §5.

## 2. Dual decomposition

Drop the quadratic penalty and dualise only the equality constraints $y_i^j = z^j$ with an
**unscaled** multiplier $\nu_i^j$. One iteration is

| step | ADMM (this repo) | dual decomposition |
| --- | --- | --- |
| x-update | $y_i \leftarrow \arg\min\, f_i(y_i) + \frac{\rho}{2}\sum_{j\in\bar{\mathcal N}_i}\lVert y_i^j - z^j + \lambda_i^j\rVert^2$ | $y_i \leftarrow \arg\min\, f_i(y_i) + \sum_{j\in\bar{\mathcal N}_i}\nu_i^j\!\cdot\! y_i^j$ |
| z-update | $z^j \leftarrow \frac{1}{\lvert C(j)\rvert}\sum_{i\in C(j)}\bigl(y_i^j + \lambda_i^j\bigr)$ | $z^j \leftarrow \frac{1}{\lvert C(j)\rvert}\sum_{i\in C(j)} y_i^j$ |
| dual update | $\lambda_i^j \leftarrow \lambda_i^j + y_i^j - z^j$ | $\nu_i^j \leftarrow \nu_i^j + \eta\,(y_i^j - z^j)$ |

where $\lambda = \nu/\rho$ is the *scaled* dual and $\eta > 0$ is the subgradient step.
The columns are deliberately aligned so the single added term is visible: ADMM keeps
$\tfrac{\rho}{2}\lVert y_i^j - z^j + \lambda_i^j\rVert^2$ in the x-update **and** carries
$\lambda_i^j$ inside the averaging. Dual decomposition has neither — its average is the
plain mean, and its dual update is a fixed-step subgradient ascent.

Two structural weaknesses follow immediately:

1. **Strict convexity is required.** Without the $\tfrac{\rho}{2}\lVert y\rVert^2$ term,
   the x-update objective is linear in every free copy $y_i^j$ ($j \neq i$) except for
   whatever curvature $f_i$ itself supplies. If $f_i$ has a null direction in a copy, the
   subproblem is unbounded below (when $\nu_i^j \neq 0$) or multi-valued (when
   $\nu_i^j = 0$). §4 demonstrates this by zeroing the tracking/effort weights.
2. **The step must be tuned to an unknown Lipschitz constant.** Subgradient ascent
   converges for any sufficiently small $\eta$, but the *useful* range is set by the
   inverse of the dual function's smoothness — a spectral quantity of the coupled problem.
   ADMM's $\rho$ plays the same role, but its good region is far wider (§5), which is the
   practical payoff of the augmented term.

## 3. Baseline implementation

The baseline lives in `python/distributed_mpc_admm/dual_decomposition.py`, a faithful
mirror of `consensus_admm.py`:

- `DualDecompositionAgentSolver` is structurally identical to `CvxpyAgentSolver` (same
  dynamics condensation, same limits, same $f_i$, DPP-compiled once), except the
  augmented-Lagrangian penalty `0.5 * rho * sum_squares(y) - y @ w_p` is replaced by a
  single linear dual term `nu @ y`. The unscaled $\nu_i^j$ is passed through
  `LocalProblemData.lam`.
- `DualDecomposition.solve` runs the loop: x-update → plain-average z-update → dual
  update $\nu \mathrel{+}= \eta\,(y-z)$, with residuals and tolerances that differ from
  ADMM only by the missing $\rho$ factors.
- It reuses `ADMMHistory`, `LocalProblemData` and `LocalSolution`, so both methods are
  benchmarked with the *same* stopping scheme (`eps_abs = eps_rel = 1e-4`) and the same
  iteration/history bookkeeping.

Two implementation notes that matter for honesty. First, the naive "reuse
`PerAgentSolver` with $\rho = 0$" shortcut does **not** produce dual decomposition: at
$\rho = 0$ the `CvxpyAgentSolver` consensus penalty *and* its linear `w_p = rho(z-lam)`
term both vanish, leaving an uncoupled problem with no dual feedback at all. A dedicated
solver with a genuine linear dual term is required. Second, the driver passes formation
`offsets` to each solver's **constructor**, because both solvers compile $f_i$ (including
the formation terms) once at build time from those offsets; passing them only to `.solve()`
silently drops the coupling (and, for dual decomposition, makes the x-update unbounded).

## 4. What to measure

1. **Iterations and wall time to the same tolerance** across five instances: 4 agents on
   a cycle and a complete graph, 8 agents on a cycle and a path, and the `r_input = 0`
   variant. Both methods stop on the same residual/tolerance test, so `iterations` and
   `converged` are directly comparable.
2. **Tuning sensitivity.** Each method is swept on a coarse grid (`rho` vs `step_size`) at
   seed 0 and shown at its *best* value — the fairest treatment of the weaker method.
3. **Loss of strict convexity.** Zero the tracking and effort weights (and drop formation
   offsets) so $f_i$ stops pinning the free copies; the x-update then has a null
   direction and dual decomposition returns `unbounded`, while ADMM's $\rho$-penalty keeps
   its subproblem bounded.
4. **Wall time per iteration.** Dual decomposition's iterations are cheaper (a linear
   dual term instead of a quadratic penalty), so comparing iteration counts alone would
   understate it; the wall-time column accounts for this.
5. **Packet loss.** ADMM's channel/loss model (`LossyChannel`) is exercised in
   `notebooks/04_switching_topology.ipynb`; the baseline has no channel model by design
   (the comparison is on perfect synchronous communication, where dual decomposition is
   already at its most favourable), so loss behaviour is reported for ADMM only and left
   as the honest asymmetry it is.

## 5. Results

Median over 3 seeds, `eps_abs = eps_rel = 1e-4`, `max_iterations = 2000`. Each method is
shown at its best coarse-grid tuning, chosen at seed 0 (`rho` and `step` columns). Measured
on an Intel Core i7-7600U @ 2.80 GHz, Python 3.14.4, CVXPY 1.9.2, OSQP 1.1.3.

| Problem | ADMM rho | DD step | ADMM iters | DD iters | ADMM wall (ms) | DD wall (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 agents, cycle | 20 | 2 | 25 | 223 | 689 | 4470 |
| 4 agents, complete | 20 | 2 | 29 | 295 | 812 | 5630 |
| 8 agents, cycle | 20 | 2 | 25 | 221 | 1095 | 8048 |
| 8 agents, path | 20 | 2 | 23 | 216 | 948 | 6958 |
| 4 agents, r_input = 0 | 20 | 2 | 23 | 223 | 1097 | 5635 |

ADMM is faster in iterations **and** wall time on every instance — by roughly an order of
magnitude — despite dual decomposition's cheaper per-iteration x-update. The final primal
residual at the declared stopping point is ~`1e-3` for both methods, not `1e-4`: the
relative term in `eps_pri = sqrt(n_dual)·eps_abs + eps_rel·max(||y||,||z||)` dominates once
`||y||` is ~10, so "1e-4 tolerance" means the *relative* component, which is exactly what
both methods are held to.

**Tuning sweep (cycle(4), seed 0, `max_iterations = 500`).** Neither method shows a U-curve
over this grid — both improve monotonically toward the right edge — but the *width* of the
usable region is the point:

| ADMM rho | 0.5 | 1.0 | 2.0 | 5.0 | 10.0 | 20.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iters | 500 ✗ | 455 | 229 | 93 | 47 | **23** |

| DD step | 0.2 | 0.5 | 0.7 | 1.0 | 1.5 | 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iters | 500 ✗ | 500 ✗ | 500 ✗ | 449 | 298 | **223** |

ADMM converges for every `rho` from 1.0 up (a ≥20× range, and `rho` can be raised further —
it only degrades the dual residual, which is not the binding constraint at this tolerance).
Dual decomposition needs `step >= 1.0` just to converge inside 500 iterations, and its
usable range sits against the subgradient stability ceiling — raise the step a little more
and it diverges. That asymmetry is the augmented term's practical payoff, and it is more
decisive here than the asymptotic iteration-complexity bounds.

**Strict-convexity demonstration.** Zero the tracking and effort weights
(`q_position = p_terminal = r_input = 0`) and drop the formation offsets, so `f_i` stops
pinning the free copies `y_i^j (j != i)`. Dual decomposition then raises
`RuntimeError: ... non-optimal status 'unbounded'` on the first x-update; the same instance
under ADMM stays bounded (the `rho`-penalty supplies the missing curvature). This is the
concrete failure mode §2 warned about, and it is why the `rho = 0` "shortcut" (§3) cannot
even be benchmarked.

## 6. When dual decomposition still wins

Dual decomposition is not obsolete, and a fair comparison has to name the regimes where it
wins. Its x-update is cheaper per iteration (one linear term instead of a quadratic
penalty, and no $\rho$-scaled `w_p` parameter); its per-iteration memory footprint is
smaller; and when the dual function's Lipschitz constant is *genuinely known* — e.g. from
a fixed problem family solved thousands of times with an oracle-tuned step — it needs no
$\rho$ at all. It also parallelises across *constraints* rather than *agents*, which is
the right decomposition when the coupling graph has a natural bipartite structure (e.g.
facility/worker, producer/consumer) rather than a peer-to-peer consensus structure. None
of those regimes describe this repo: the coupling here is symmetric peer consensus, the
instances are small and re-solved each MPC step, and no oracle step size is available —
which is exactly the regime where the augmented term earns its keep.

## 7. Verdict for this repo

For the peer-consensus coupling and receding-horizon workload in this repository, ADMM is
the right default: it is robust to the same tolerance with dramatically fewer iterations
and no hand-tuned subgradient step (§5). The single augmented term is what buys that
robustness — it keeps every subproblem bounded even when $f_i$ thins out, and it turns a
sublinear, step-size-sensitive ascent into a linear-rate contraction. Dual decomposition
remains the better tool when the coupling is constraint-structured and a Lipschitz
constant is known, but neither condition holds here.

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., and Eckstein, J. (2011). *Distributed
  optimization and statistical learning via the alternating direction method of
  multipliers.* Foundations and Trends in Machine Learning, 3(1):1–122. §§2, 7.
- Everett, H. III (1963). *Generalized Lagrange multiplier method for solving problems of
  optimum allocation of resources.* Operations Research, 11(3):399–417.
- Dantzig, G. B., and Wolfe, P. (1960). *Decomposition principle for linear programs.*
  Operations Research, 8(1):101–111.
- Christofides, P. D., Scattolini, R., Muñoz de la Peña, D., and Liu, J. (2013).
  *Distributed model predictive control: a tutorial review and future research
  directions.* Computers & Chemical Engineering, 51:21–41.
