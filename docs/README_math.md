# Mathematical reference

<!-- TODO(deepseek §13.1): fill every section. Keep notation identical to the code and to
     docs/derivations/*.tex. Every symbol used here must appear in the table below. -->

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

<!-- TODO(deepseek §13.1): extend as needed; do not introduce a symbol in a .tex file without adding it
     here first. -->

## 2. Agent model

<!-- TODO(deepseek §13.1): continuous double integrator, ZOH discretisation, the exact A and B, and the
     condensed prediction X = Phi x0 + Gamma U with the t = 1..T indexing convention
     stated explicitly. That convention is the single most common source of off-by-one
     bugs in this repo -- write it down here and cite this section from the code. -->

## 3. The coupled MPC problem

<!-- TODO(deepseek §13.1): the centralised problem first: sum of per-agent tracking and effort costs plus
     formation edge costs, subject to per-agent dynamics and box constraints. Show that it
     is a single convex QP, and give its size as a function of (N, T) so the reader can see
     why distributing it matters. -->

## 4. Why it splits

<!-- TODO(deepseek §13.1): identify the only coupling (formation edge terms). Introduce local copies,
     state the equivalence between the coupled problem and the copy-based problem with
     consensus constraints, and prove the equivalence in one paragraph (it is immediate --
     but state it, because it is the step that licenses everything else). -->

## 5. Consensus ADMM

<!-- TODO(deepseek §13.1): augmented Lagrangian, the three updates, the scaled-dual substitution
     lambda := u / rho, and the derivation of the z-update as a plain average. Show why
     the z-update needs only neighbor communication -- that is the whole point. -->

## 6. Residuals, tolerances, stopping

<!-- TODO(deepseek §13.1): primal and dual residual definitions, the eps_pri / eps_dual formulas with the
     n_dual count spelled out, and why the dual residual is rho * ||z - z_prev|| rather
     than something involving lambda. -->

## 7. Adaptive rho

<!-- TODO(deepseek §13.1): residual balancing rule, the mandatory dual rescaling, and an explicit warning
     that the linear-rate result assumes fixed rho. -->

## 8. Convergence

<!-- TODO(deepseek §13.1): state the assumptions (closed proper convex f_i, existence of a saddle point,
     fixed graph, synchronous updates, exact local solves). Give the O(1/k) ergodic result
     and the linear rate under strong convexity plus Lipschitz gradient, with the
     lambda_2 dependence. Then state plainly which assumptions the switching-topology and
     packet-loss experiments break. Do not overclaim: this repo demonstrates a gap, it
     does not close it. -->

## 9. Tuning guide

<!-- TODO(deepseek §13.1): practical defaults and the reasoning:
     - rho scaled to the ratio of objective to consensus curvature
     - alpha in [1.5, 1.8]
     - horizon vs dt trade-off (T*dt must exceed the formation settling time)
     - w_formation vs q_position: the ratio that decides whether the formation or the
       reference wins during a conflict
     - iteration budget in closed loop: why ~10 warm-started iterations beat 200 cold ones
-->

## 10. Implementation notes

<!-- TODO(deepseek §13.1): the vec ordering convention (time-major), the QP block layout shared by the
     Python and C++ implementations, and the parity test that pins them together. -->

## References

<!-- TODO(deepseek §13.1): Boyd et al. 2011 (Foundations and Trends in ML 3(1)); Stellato et al. 2020
     (OSQP, Math. Prog. Comp.); the distributed-MPC survey used for the comparison table;
     a rigidity-theory reference for section 3 of formation_constraints. Full citations,
     not just author-year. -->
