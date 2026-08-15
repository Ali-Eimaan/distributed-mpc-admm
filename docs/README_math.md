# Mathematical reference

**Convention.** Every symbol used in `docs/derivations/*.tex` is defined here, and the
derivations pull their macros from `docs/derivations/preamble.tex`. The code stores the
*scaled* dual (Boyd 2011 §7.2's $u$), so $\lambda$ below is the scaled dual; the unscaled
multiplier is $\nu = \rho\lambda$.

Single source of truth for notation. If a symbol here disagrees with the code, the code is
wrong.

## 1. Notation

| Symbol | Code | Shape | Meaning |
| --- | --- | --- | --- |
| $N$ | `n_agents` | — | number of agents |
| $T$ | `horizon` | — | MPC prediction horizon |
| $n$ | `n_states` | — | per-agent state dimension (4 in 2D) |
| $m$ | `n_inputs` | — | per-agent input dimension (2 in 2D) |
| $x_i$ | `x0` | $(n,)$ | agent state $[p_x, p_y, v_x, v_y]^\top$ |
| $U_i$ | `inputs` | $(T, m)$ | agent input sequence |
| $\mathcal{N}_i$ | `neighbors(i)` | — | open neighborhood |
| $\bar{\mathcal{N}}_i$ | `closed_neighborhood(i)` | — | $\mathcal{N}_i \cup \{i\}$ |
| $y_i^j$ | `copies[j]` | $(T, 2)$ | $i$'s local copy of $j$'s position trajectory |
| $z^j$ | `consensus[j]` | $(T, 2)$ | agreed trajectory of agent $j$ |
| $\lambda_i^j$ | `duals[i][j]` | $(T, 2)$ | scaled dual for $y_i^j = z^j$ |
| $\rho$ | `rho` | — | penalty parameter |
| $d_{ij}$ | `offsets[j]` | $(2,)$ | desired $p_i - p_j$ |
| $L$ | `laplacian()` | $(N, N)$ | graph Laplacian |
| $\lambda_2$ | `algebraic_connectivity()` | — | Fiedler value |

| $d$ | `dim` | — | spatial dimension (2 in 2D) |
| $A, B$ | `matrices()` | — | discrete double-integrator matrices |
| $\Phi, \Gamma$ | `prediction_matrices()` | — | condensed state prediction $X = \Phi x_0 + \Gamma U$ |
| $\Phi_p, \Gamma_p$ | `position_prediction_matrices()` | — | condensed position prediction $y = \Phi_p x_0 + \Gamma_p U$ |
| $u_i(t)$ | `inputs` | — | acceleration input at prediction step $t$ |
| $p_i(t), v_i(t)$ | — | — | position / velocity at step $t$ |
| $f_i$ | `local_objective` | — | agent $i$'s separable cost (tracking + effort + formation) |
| $\nu_i^j$ | — | $(T, 2)$ | **unscaled** dual for $y_i^j = z^j$; $\nu_i^j = \rho\lambda_i^j$ |
| $\alpha$ | `alpha` | — | over-relaxation parameter |
| $\varepsilon^{\mathrm{abs}}, \varepsilon^{\mathrm{rel}}$ | `eps_abs`, `eps_rel` | — | stopping tolerances |
| $\tau, \mu$ | `tau`, `mu` | — | adaptive-$\rho$ scaling factors |
| $K$ | `n_steps` | — | closed-loop control steps (receding horizon length) |
| $r^k, s^k$ | — | — | primal / dual residual at ADMM iteration $k$ |

> Do not introduce a symbol in a `.tex` file without adding it here first.

## 2. Agent model

The continuous plant is a double integrator in $d$ spatial dimensions:

$$
\dot{p}_i = v_i, \qquad \dot{v}_i = u_i,
$$

with position $p_i \in \mathbb{R}^d$, velocity $v_i \in \mathbb{R}^d$ and acceleration
input $u_i \in \mathbb{R}^d$. The state is $x_i = [p_i^\top,\ v_i^\top]^\top \in
\mathbb{R}^{n}$ with $n = 2d$. Under a zero-order hold of duration $\Delta t$ the
dynamics are

$$
x_i(t+1) = A\, x_i(t) + B\, u_i(t),
\qquad
A = \begin{bmatrix} I & \Delta t\, I \\ 0 & I \end{bmatrix},
\qquad
B = \begin{bmatrix} \tfrac12 \Delta t^2\, I \\ \Delta t\, I \end{bmatrix},
$$

where each identity block is $d\times d$. This is exactly
`DoubleIntegrator.matrices()` in `per_agent_solver.py`.

**Condensed prediction, $t = 1..T$.** Stack the predicted states and inputs as

$$
X_i = \big[x_i(1);\ \dots;\ x_i(T)\big] \in \mathbb{R}^{Tn},
\qquad
U_i = \big[u_i(0);\ \dots;\ u_i(T-1)\big] \in \mathbb{R}^{Tm}.
$$

The horizon starts at $t = 1$: the measured state $x_i(0)$ is the *initial condition* of
the prediction, **not** row 0 of $X_i$. Rolling the recursion forward gives the condensed
form

$$
X_i = \Phi\, x_i(0) + \Gamma\, U_i,
$$

where block row $r$ of $\Phi$ (producing $x_i(r+1)$) is $A^{r+1}$, and block $(r,s)$ of
$\Gamma$ is $A^{r-s}B$ for $s \le r$ and zero for $s > r$ (causality). The code builds
these in `_condensed_prediction` and cross-checks them against `DoubleIntegrator.simulate`
in the tests. Position-only prediction uses $\Phi_p = C_p \Phi$ and
$\Gamma_p = C_p\Gamma$, where $C_p$ selects the position rows of each state
(`position_prediction_matrices`).

> **Off-by-one rule.** $t$ runs $1 \dots T$; $x_i(0)$ is not part of $X_i$; block row $r$
> corresponds to $t = r+1$. The local equality constraint is
> $y_i^i = \Phi_p x_i(0) + \Gamma_p U_i$, never $x_i(0)$ on the right.

## 3. The coupled MPC problem

Write $q_p =$ `q_position`, $q_v =$ `q_velocity`, $r =$ `r_input`, $p_T =$ `p_terminal`,
$w_f =$ `w_formation` for the cost weights. The centralised (open-loop) MPC problem is

$$
\begin{aligned}
\min_{U_1,\dots,U_N} \quad
\sum_{i=1}^N \sum_{t=1}^T &\Big[
  q_p \lVert p_i(t) - p_i^{\mathrm{ref}}(t)\rVert^2
+ q_v \lVert v_i(t)\rVert^2
+ r \lVert u_i(t)\rVert^2
+ p_T \lVert p_i(T) - p_i^{\mathrm{ref}}(T)\rVert^2 \Big] \\
&+ w_f \sum_{(i,j)\in\mathcal E}\ \sum_{t=1}^T
   \big\lVert (p_i(t) - p_j(t)) - d_{ij}\big\rVert^2
\end{aligned}
$$

subject to, for every agent $i$ and step $t$,

$$
x_i(t+1) = A x_i(t) + B u_i(t),
\qquad
|u_i(t)| \le u_{\max},\qquad
|v_i(t)| \le v_{\max},\qquad
p_{\min} \le p_i(t) \le p_{\max},
$$

with the boxes optional and component-wise. Here $\mathcal E$ is the formation-edge set and
$d_{ij}$ the desired displacement $p_i - p_j$ (so $d_{ij} = -d_{ji}$).

After condensing out the states this is a **single convex quadratic program** in the
$N T d$ input variables. Its Hessian is block-structured: a block-diagonal part (tracking,
velocity, effort — one $Td\times Td$ block per agent) plus a sparse formation coupling
whose sparsity pattern is exactly the graph $\mathcal G$. The problem is strictly convex
whenever $r > 0$ or $q_v > 0$; with $r = q_v = 0$ it is merely convex and the solution is
not unique. For $N = 8$, $T = 20$, $d = 2$ the condensed QP has $320$ variables and a
Hessian with $\mathcal O(NTd + |\mathcal E| T d)$ nonzeros — small enough to solve
centrally, but the point is that **no central solver exists** in the deployment this repo
targets.

## 4. Why it splits

The only terms that couple distinct agents are the formation edge costs: they contain
$(p_i - p_j)$, which no regrouping makes separable across $i$. Every tracking, velocity,
effort and box constraint already lives on a single agent.

Introduce, for each agent $i$ and each $j \in \bar{\mathcal N}_i$, a **local copy**
$y_i^j \in \mathbb{R}^{Td}$ of agent $j$'s position trajectory, together with a global
consensus trajectory $z^j \in \mathbb{R}^{Td}$. The copy-based problem is

$$
\min \ \sum_i f_i\big(y_i^i, \{y_i^j\}_{j\in\mathcal N_i}\big)
\quad\text{s.t.}\quad
y_i^j = z^j \quad \forall\, i,\ j\in\bar{\mathcal N}_i,
\qquad
y_i^i = \Phi_p x_i(0) + \Gamma_p U_i,
$$

where $f_i$ collects agent $i$'s tracking, effort and formation costs (the formation term
becomes $w_f\lVert(y_i^i - y_i^j) - d_{ij}\rVert^2$ over $i$'s incident edges). The
equivalence is immediate: at any feasible copy-based point the consensus constraints force
$y_i^j = z^j$ for every copy, so substituting back collapses all copies of $j$ into a
single trajectory and the objective reproduces the coupled problem; conversely any coupled
solution sets $z^j = p_j$ and $y_i^j = p_j$. This one-to-one map is what licenses replacing
the coupled QP by a constrained, separable problem.

## 5. Consensus ADMM

Form the **augmented Lagrangian** of the copy-based problem (unscaled multiplier
$\nu_i^j$ on each consensus constraint):

$$
\mathcal L_\rho = \sum_i f_i(y_i) + \sum_{i}\sum_{j\in\bar{\mathcal N}_i}\Big[
  \nu_i^j \cdot (y_i^j - z^j) + \tfrac{\rho}{2}\lVert y_i^j - z^j\rVert^2 \Big].
$$

ADMM alternates three steps, each holding the other two blocks fixed:

1. **x-update** — minimise over $\{y_i^j, U_i\}$ for every $i$ independently. Because
   $f_i$ and the penalty both separate across $i$, this is $N$ independent QPs
   (`CvxpyAgentSolver`).
2. **z-update** — minimise over $z^j$ for every $j$ independently.
3. **dual update** — $\nu_i^j \leftarrow \nu_i^j + \rho(y_i^j - z^j)$.

**Scaled form.** Substitute $\lambda_i^j = \nu_i^j / \rho$ and complete the square; the
penalty plus linear term become $\tfrac{\rho}{2}\lVert y_i^j - z^j + \lambda_i^j\rVert^2$
up to constants, and the dual update becomes the tidy
$\lambda_i^j \leftarrow \lambda_i^j + (y_i^j - z^j)$. The code stores this **scaled** dual
(`duals[i][j]`).

**The z-update is an average.** Setting $\nabla_{z^j}\mathcal L_\rho = 0$ gives

$$
z^j = \frac{1}{|\{i : j \in \bar{\mathcal N}_i\}|}
       \sum_{i : j \in \bar{\mathcal N}_i} \big(y_i^j + \lambda_i^j\big),
$$

i.e. the average of $y_i^j + \lambda_i^j$ over all agents that hold a copy of $j$ — exactly
the contributors of $j$. The set of contributors is $\bar{\mathcal N}_j$ in an undirected
graph, so **the z-update for $j$ needs only messages from $j$'s neighbors**. That is the
single fact that makes the iteration distributed; it is implemented verbatim as
`ConsensusADMM._z_update`.

## 6. Residuals, tolerances, stopping

With the scaled dual, the two residuals are

$$
r^k = \Big(\sum_{i}\sum_{j\in\bar{\mathcal N}_i}\lVert y_i^j - z^j\rVert^2\Big)^{1/2},
\qquad
s^k = \rho\,\Big(\sum_j \lVert z^j - z^{j}_{k-1}\rVert^2\Big)^{1/2},
$$

where $z^j_{k-1}$ is the previous iterate. $r^k$ is the consensus violation; $s^k$ is the
dual residual written in terms of the consensus *change*, which (by the $z$-update
optimality condition) is equivalent to $\lVert\rho(z^k - z^{k-1})\rVert$ — this is why the
factor $\rho$ appears and why no explicit $\lambda$ term is needed. Stopping uses

$$
\varepsilon^{\mathrm{pri}} = \sqrt{n_{\mathrm{dual}}}\,\varepsilon^{\mathrm{abs}}
  + \varepsilon^{\mathrm{rel}}\max\{\lVert y\rVert, \lVert z\rVert\},
\qquad
\varepsilon^{\mathrm{dual}} = \sqrt{n_{\mathrm{dual}}}\,\varepsilon^{\mathrm{abs}}
  + \varepsilon^{\mathrm{rel}}\,\rho\,\lVert\lambda\rVert,
$$

with $n_{\mathrm{dual}} = \sum_i |\bar{\mathcal N}_i|\, T\, d$ the total number of scalar
consensus constraints. Convergence is declared when
$r^k \le \varepsilon^{\mathrm{pri}}$ **and** $s^k \le \varepsilon^{\mathrm{dual}}$.
These are `ConsensusADMM._residuals` and `ConsensusADMM._tolerances`.

## 7. Adaptive rho

`ADMMOptions.adaptive_rho` implements residual balancing (Boyd 2011 §3.4.1): when the
primal residual dominates, scale $\rho$ up by $\tau$; when the dual dominates, scale down
by $1/\tau$:

$$
\rho_{k+1} =
\begin{cases}
\tau\rho_k & r^k > \mu s^k,\\
\rho_k / \tau & s^k > \mu r^k,\\
\rho_k & \text{otherwise},
\end{cases}
\qquad
\mu \sim 10,\ \tau \sim 2,
$$

clipped to `[rho_min, rho_max]`. **Because the code stores the scaled dual, every change of
$\rho$ must rescale $\lambda$ by the reciprocal factor** (`_update_rho` does
`lam /= (new_rho/old_rho)`); forgetting this is the classic adaptive-$\rho$ bug. Note that
the linear-rate guarantees in section 8 assume a *fixed* $\rho$; adaptive $\rho$ is a
practical heuristic, not covered by that theorem.

## 8. Convergence

The classical guarantees (see `convergence_proof.tex`) require:

- **A1** each $f_i$ closed, proper, convex;
- **A2** the unaugmented Lagrangian has a saddle point;
- **A3** the communication graph is fixed and connected;
- **A4** updates are synchronous;
- **A5** local subproblems are solved exactly.

Under A1–A5 the residuals converge to zero, the objective converges to the optimum, and the
*ergodic* average of the iterates converges at $O(1/k)$. If each $f_i$ is additionally
strongly convex with $L$-Lipschitz gradient, the rate is linear, with the contraction
governed by the graph through $\lambda_2(L)$ and $\lambda_{\max}(L)$; an explicit bound is
derived in `convergence_proof.tex` §5.

**Which assumptions this repo's experiments break:** the switching-topology studies
(notebook 04, `analysis/topology_robustness.ipynb`) violate A3 — the decision variable
changes dimension at a merge event — and the packet-loss studies violate A4. The
fixed-iteration inner solves on hardware also violate A5. The standard theorem therefore
does **not** apply there; this repo *measures the gap* and does not close it. See
`convergence_proof.tex` §7 for the precise statement of what fails.

## 9. Tuning guide

Defaults that work across the notebooks (all available in `ADMMOptions` /
`AgentCostWeights`):

- **$\rho$:** scale it to the ratio of objective curvature to consensus curvature. For the
  formation problem the empirical optimum sits around $\rho \in [1, 10]$; notebook 05 sweeps
  it and finds a wide, shallow basin.
- **$\alpha$:** over-relaxation in $[1.5, 1.8]$ reliably halves the iteration count;
  $\alpha = 1$ is plain ADMM and $\alpha \ge 2$ diverges.
- **$T\,\Delta t$:** must exceed the formation settling time, or the closed loop chases a
  moving target. Longer $T$ at fixed $\Delta t$ costs more per QP but not more per step.
- **$w_f$ vs $q_p$:** the ratio decides which wins a tracking/formation conflict. $w_f
  \gg q_p$ holds the shape and lets the centroid lag; $q_p \gg w_f$ tracks tightly and lets
  the shape distort.
- **Iteration budget in closed loop:** ~10 warm-started iterations per MPC step (using
  `ADMMResult.shifted()` as the initial guess) beats 200 cold iterations, because the
  previous step's consensus is already within the basin of the new one.

## 10. Implementation notes

Both the Python and C++ implementations share one layout: every trajectory is stored
**time-major**, i.e. block $t$ occupies rows $t d \dots (t+1)d - 1$ of the flattened
vector. The local QP has, per agent, $|\bar{\mathcal N}_i|$ trajectory variables
($y_i^j$) plus the input vector $U_i$; its equality block is the single row
$y_i^i = \Phi_p x_0 + \Gamma_p U_i$ and its objective is the tracking/effort/formation
sum plus the DPP-safe expansion
$\tfrac{\rho}{2}\lVert y\rVert^2 - y^\top \rho(z-\lambda)$ (see `CvxpyAgentSolver`).
`cpp_admm/test/test_admm_kernel.cpp` and `python/tests/test_admm_convergence.py` pin the
two implementations to the same numerics on a fixed seed, so the C++ kernel and the Python
reference are interchangeable up to solver tolerance.

## References

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). *Distributed
  Optimization and Statistical Learning via the Alternating Direction Method of
  Multipliers.* Foundations and Trends in Machine Learning, 3(1), 1–122.
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). *OSQP: An
  Operator Splitting Solver for Quadratic Programs.* Mathematical Programming
  Computation, 12(4), 637–672.
- The distributed-MPC survey, linear-rate ADMM result, switching-topology consensus,
  rigidity-theory and hybrid-systems references live in `docs/derivations/references.bib`.
